"""
Component-rate forecasting for UCL fantasy points.

Idea: forecast the *rates* that generate fantasy points (availability, minutes per appearance, goals/90, assists/90,
clean-sheet rate, goals conceded/90, cards/90) from last season's domestic form, then convert the forecast rates into
expected fantasy points with the game's scoring rules.

Design choices (see rate_forecast_model.ipynb for the reasoning and results):
  * unit of analysis : player-season  (UCL season T  <-  domestic season T-1)
  * regression       : weighted least squares, weights = exposure of the target rate (e.g. UCL 90s played)
  * noisy inputs     : domestic rates are shrunk toward the position mean, (events + k*mu) / (exposure + k)
  * honest selection : the shrinkage constant k (or 'use the training mean') is chosen using only *earlier*
                       evaluation origins, so the evaluation year never votes on the model (Ch 6, look-ahead bias)
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

POSMAP = {"Goalkeeper": "GK", "Defender": "DEF", "Midfield": "MID", "Attack": "FWD"}

# UEFA association coefficient by league and season-start-year; columns = 22/23, 23/24, 24/25, 25/26 (earlier -> 22/23)
COEF = {
    "GB1": [23.000, 17.375, 29.464, 28.680], "IT1": [22.357, 21.000, 21.875, 19.000], "ES1": [16.571, 16.062, 23.892, 22.093],
    "L1": [17.125, 19.357, 18.421, 21.785], "FR1": [12.583, 16.250, 17.928, 18.321], "PO1": [12.500, 11.000, 16.250, 20.500],
    "BE1": [14.200, 14.400, 15.650, 11.400], "NL1": [13.500, 10.000, 15.250, 9.979], "TR1": [11.800, 12.000, 10.300, 11.075],
    "GR1": [2.125, 11.400, 12.687, 14.200], "UKR1": [5.700, 4.100, 3.600, 8.312], "A1": [4.900, 4.800, 9.650, 4.100],
    "NO1": [5.750, 8.000, 11.812, 8.050], "TS1": [6.750, 13.500, 10.550, 11.025],
}

# component -> (numerator, denominator, weight column, positions modelled)
SPEC = {
    "g90": ("goals", "nineties", "nineties", ["DEF", "MID", "FWD"]),
    "a90": ("assists", "nineties", "nineties", ["DEF", "MID", "FWD"]),
    "yc90": ("yc", "nineties", "nineties", ["GK", "DEF", "MID", "FWD"]),
    "apps_share": ("apps", "club_games", "club_games", ["GK", "DEF", "MID", "FWD"]),
    "mpa": ("minutes", "apps", "apps", ["GK", "DEF", "MID", "FWD"]),
    "p60": ("apps60", "apps", "apps", ["GK", "DEF", "MID", "FWD"]),
    "cs_rate": ("cs", "apps60", "apps60", ["GK", "DEF", "MID"]),
    "gc90": ("gc_onpitch", "nineties", "nineties", ["GK", "DEF"]),
}
KGRID = [0, 4, 8, 16, 32, 64, 128]
MEAN = -1  # candidate id for "just use the training mean"
SUB_BASE = ["Goalkeeper", "Centre-Back", "Centre-Forward", "Central Midfield"]

# fantasy scoring weights by UEFA position (notes.md)
GOAL_W = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_W = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}


# --------------------------------------------------------------------------------------------- data
def league_strength(league, dom_season):
    i = min(max(int(dom_season) - 2022, 0), 3)
    return COEF.get(league, [np.nan] * 4)[i]


def prepare_stints(stints: pd.DataFrame):
    st = stints.copy()
    st["pos"] = st.position.map(POSMAP)
    st["nineties"] = st.minutes / 90.0
    dom = st[(st.group == "DOM") & (st.minutes >= 450)].copy()
    cl = st[st.group == "CL"].copy()
    return dom, cl


def build_panel(stints: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """UCL season T (target stint)  <-  domestic season T-1 (inputs), one row per player-season."""
    dom, cl = prepare_stints(stints)
    d = dom.rename(columns=lambda c: c if c == "player_id" else f"d_{c}")
    d["season"] = d.d_season + 1
    panel = cl.merge(d, on=["player_id", "season"], how="inner")
    c = ctx[["club_id", "season", "gf_pg", "ga_pg"]].copy()
    c["season"] += 1  # club's domestic context in T-1, aligned to UCL season T
    panel = panel.merge(c, on=["club_id", "season"], how="left")
    panel["league_strength"] = [league_strength(l, s - 1) for l, s in zip(panel.d_league, panel.season)]
    panel["age2"] = panel.d_age ** 2
    return panel.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- rates
def _den(df, den, prefix=""):
    return df[f"{prefix}nineties"] if den == "nineties" else df[f"{prefix}{den}"]


def target_rate(comp, df):
    num, den, _, _ = SPEC[comp]
    return df[num] / _den(df, den).replace(0, np.nan)


def pop_mu(dom, comp, pos, upto):
    num, den, _, _ = SPEC[comp]
    x = dom[(dom.pos == pos) & (dom.season <= upto)]
    return x[num].sum() / _den(x, den).sum()


def design(df, comp, k, mu, fill):
    num, den, _, _ = SPEC[comp]
    n, d = df[f"d_{num}"], _den(df, den, "d_")
    X = pd.DataFrame({"own": (n + k * mu) / (d + k), "age": df.d_age, "age2": df.age2, "lstr": df.league_strength,
                      "gf": df.gf_pg.fillna(fill["gf"]), "ga": df.ga_pg.fillna(fill["ga"])}, index=df.index)
    sub = pd.get_dummies(df.d_sub_position)
    sub = sub.drop(columns=[c for c in SUB_BASE if c in sub], errors="ignore")
    return pd.concat([X, sub.astype(float)], axis=1)


def fit_predict(train, test, comp, pos, cand, dom, upto, return_fit=False):
    """Fit on `train`, predict `test`. cand = shrinkage k, or MEAN for the training-mean benchmark."""
    wcol = SPEC[comp][2]
    y = target_rate(comp, train)
    w = train[wcol]
    ok = y.notna() & w.notna() & (w > 0)
    tr, y, w = train[ok], y[ok], w[ok]
    base = float(np.average(y, weights=w))
    if cand == MEAN:
        return pd.Series(base, index=test.index), base, None
    mu = pop_mu(dom, comp, pos, upto)
    fill = {"gf": tr.gf_pg.mean(), "ga": tr.ga_pg.mean()}
    both = pd.concat([tr, test]).reset_index(drop=True)  # align train/test rows by position (indices may repeat)
    X = design(both, comp, cand, mu, fill)
    Xtr, Xte = X.iloc[: len(tr)], X.iloc[len(tr):]
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    m = Xtr.notna().all(axis=1).values
    fit = sm.WLS(y.values[m], sm.add_constant(Xtr[m], has_constant="add"), weights=w.values[m]).fit()
    mt = Xte.notna().all(axis=1).values
    pred = pd.Series(np.nan, index=test.index)
    pred[mt] = np.asarray(fit.predict(sm.add_constant(Xte[mt], has_constant="add")))  # .values: labels differ from test.index
    pred = pred.fillna(base)
    return (pred, base, fit) if return_fit else (pred, base, mu)


# --------------------------------------------------------------------------------------------- evaluation
def grid_errors(panel, dom, origins):
    """Weighted MSE of every candidate at every expanding-window origin, plus benchmarks."""
    rows = []
    for comp, (num, den, wcol, poss) in SPEC.items():
        for pos in poss:
            P = panel[panel.pos == pos]
            for T0 in origins:
                train, test = P[P.season < T0], P[P.season == T0]
                if len(train) < 80 or len(test) < 15:
                    continue
                yt, wt = target_rate(comp, test), test[wcol]
                ok = (yt.notna() & wt.notna() & (wt > 0)).values
                if ok.sum() < 10:
                    continue
                te = test[ok]
                y_, w_ = yt[ok].values, wt[ok].values
                nn, dd = te[f"d_{num}"], _den(te, den, "d_")
                raw = (nn / dd.replace(0, np.nan)).values
                for cand in [MEAN] + KGRID:
                    pred, base, _ = fit_predict(train, te, comp, pos, cand, dom, T0 - 1)
                    rows.append(dict(comp=comp, pos=pos, T0=T0, cand=cand, n=len(te),
                                     mse=np.average((y_ - pred.values) ** 2, weights=w_),
                                     mse_mean=np.average((y_ - base) ** 2, weights=w_),
                                     mse_raw=np.average((y_ - np.where(np.isnan(raw), base, raw)) ** 2, weights=w_)))
    return pd.DataFrame(rows)


def choose_candidate(errs, comp, pos, T0, default=16, min_origins=2):
    """Pick the candidate (k or MEAN) with the lowest average error over origins strictly before T0."""
    g = errs[(errs.comp == comp) & (errs.pos == pos) & (errs.T0 < T0)]
    if g.T0.nunique() < min_origins:
        return default
    return int(g.groupby("cand").mse.mean().idxmin())


def honest_table(errs, first_origin=2023):
    out = []
    for (comp, pos), g in errs.groupby(["comp", "pos"]):
        for T0 in sorted(g.T0.unique()):
            cand = choose_candidate(errs, comp, pos, T0)
            r = g[(g.T0 == T0) & (g.cand == cand)]
            if len(r):
                r = r.iloc[0]
                out.append(dict(comp=comp, pos=pos, T0=T0, cand=cand, n=r.n, mse=r.mse, mse_mean=r.mse_mean, mse_raw=r.mse_raw))
    return pd.DataFrame(out)


def summarize(honest, first_origin=2023):
    h = honest[honest.T0 >= first_origin]
    f = lambda g: pd.Series({
        "n": g.n.sum(),
        "R2_model": 1 - np.average(g.mse, weights=g.n) / np.average(g.mse_mean, weights=g.n),
        "R2_raw_domestic": 1 - np.average(g.mse_raw, weights=g.n) / np.average(g.mse_mean, weights=g.n),
        "choice": "mean" if (g.cand == MEAN).mean() > 0.5 else f"k={int(g.cand[g.cand != MEAN].mode().iat[0])}" if (g.cand != MEAN).any() else "mean"})
    return h.groupby(["comp", "pos"]).apply(f).reset_index()


# --------------------------------------------------------------------------------------------- prediction
def predict_components(panel, dom, errs, T0):
    """Predict every component for the rows of season T0 using only data before T0 (honest candidate choice)."""
    test_all = panel[panel.season == T0].copy()
    for comp, (num, den, wcol, poss) in SPEC.items():
        test_all[f"p_{comp}"] = np.nan
        for pos in poss:
            idx = test_all.index[test_all.pos == pos]
            train = panel[(panel.pos == pos) & (panel.season < T0)]
            if len(idx) == 0 or len(train) < 80:
                continue
            cand = choose_candidate(errs, comp, pos, T0)
            pred, _, _ = fit_predict(train, test_all.loc[idx], comp, pos, cand, dom, T0 - 1)
            test_all.loc[idx, f"p_{comp}"] = pred.values
    return test_all


def other_points_per90(official: pd.DataFrame, stints: pd.DataFrame, upto: int):
    """Points per 90 not produced by the forecast components (balls recovered, saves, PotM, penalties, ...), by UEFA position,
    calibrated on official UEFA points for seasons <= upto."""
    cl = stints[stints.group == "CL"][["player_id", "season", "apps", "apps60", "minutes"]]
    o = official[(official.season <= upto) & official.tm_player_id.notna()].merge(
        cl, left_on=["tm_player_id", "season"], right_on=["player_id", "season"])
    modelled = (o.apps + o.apps60 + o.pos.map(GOAL_W) * o.gS + 3 * o.assist + o.pos.map(CS_W) * o.cS
                - np.where(o.pos.isin(["GK", "DEF"]), 0.5 * o.gC, 0) - (o.yC + 3 * o.rC))
    o = o.assign(other=o.totPts - modelled, n90=o.minutes / 90.0)
    return (o.groupby("pos").other.sum() / o.groupby("pos").n90.sum()).to_dict()


def expected_points(df, pos_uefa, club_games, other90):
    """Expected fantasy points from forecast components. df needs p_* columns. pos_uefa/club_games are Series aligned to df."""
    m = df.p_mpa / 90.0
    pts_app = (1.0 + df.p_p60
               + pos_uefa.map(GOAL_W) * df.p_g90.fillna(0) * m
               + 3.0 * df.p_a90.fillna(0) * m
               + pos_uefa.map(CS_W) * df.p_cs_rate.fillna(0) * df.p_p60
               - np.where(pos_uefa.isin(["GK", "DEF"]), 0.5 * df.p_gc90.fillna(0) * m, 0.0)
               - df.p_yc90.fillna(0) * m
               + pos_uefa.map(other90) * m)
    return df.p_apps_share * club_games * pts_app, pts_app
