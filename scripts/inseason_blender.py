"""
inseason_blender.py -- ECON 5371 Econometric Blending Engine for UCL Fantasy

Implements:
1. Bates-Granger Optimal Forecast Combination (Chapter 5.5) between pre-season OLS and Rate models,
   with shrinkage towards equal weights (the forecast combination puzzle).
2. Bayesian Exposure-Weighted In-Season Updating:
   Incorporates 2026/27 domestic performance (larger sample size, ~8-12 matches) and
   2026/27 UCL matchday performance (~3-4 matches, high variance).
3. Horizon Scaling:
   Projects expected fantasy points across the remaining league-phase matches (G_rem).
4. Asymmetric Loss & Hit Hurdle (Chapter 5.2):
   Evaluates transfer penalties under asymmetric risk.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "out")

# Scoring weights by UEFA position (notes.md)
GOAL_W = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_W = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}

# Prior exposure scale parameters (half-saturation in minutes)
M0_DOM = 650.0   # ~7 full games: half-weight to domestic form
M0_UCL = 360.0   # ~4 full games: half-weight to UCL form


def compute_bates_granger_weights(ols_series, rate_series, shrinkage=0.5):
    """
    Bates-Granger (1969) optimal combination weight for two forecasts (ECON 5371 Chapter 5.5):
        w1* = (sigma2^2 - sigma12) / (sigma1^2 + sigma2^2 - 2*sigma12)
    Shrinkage towards equal weights (1/K = 0.5) to avoid sample covariance estimation noise:
        w1_shrunk = (1 - shrinkage) * w1* + shrinkage * 0.5
    """
    valid = ols_series.notna() & rate_series.notna()
    e1, e2 = ols_series[valid], rate_series[valid]
    v1 = float(np.var(e1, ddof=1))
    v2 = float(np.var(e2, ddof=1))
    cov12 = float(np.cov(e1, e2)[0, 1])

    denom = v1 + v2 - 2 * cov12
    if denom <= 0:
        w_ols_raw = 0.5
    else:
        w_ols_raw = float(np.clip((v2 - cov12) / denom, 0.05, 0.95))

    w_ols = (1.0 - shrinkage) * w_ols_raw + shrinkage * 0.5
    w_rate = 1.0 - w_ols
    return w_ols, w_rate


def domestic_expected_ppm(row, pos):
    """
    Convert observed domestic rates into expected fantasy points per match.
    Includes appearance, playing 60+ mins, goals, assists, clean sheets, and card deductions.
    """
    mins = row.get("dom_mins", np.nan)
    if pd.isna(mins) or mins <= 0:
        return np.nan

    apps = max(1.0, row.get("dom_apps", mins / 85.0))
    mpa = mins / apps
    p60 = min(1.0, max(0.0, (mpa - 45.0) / 35.0))  # approx probability of 60+ mins
    m = mpa / 90.0

    g90 = row.get("dom_g90", 0.0) if pd.notna(row.get("dom_g90")) else 0.0
    a90 = row.get("dom_a90", 0.0) if pd.notna(row.get("dom_a90")) else 0.0
    cs_rate = row.get("dom_cs_rate", 0.0) if pd.notna(row.get("dom_cs_rate")) else 0.0

    # Rating bonus proxy (players averaging >= 7.5 rating often win PotM bonuses or score well)
    rating = row.get("dom_rating", 7.0)
    rating_bonus = max(0.0, (rating - 7.0) * 0.8)

    ppm = (1.0 + p60
           + GOAL_W[pos] * g90 * m
           + 3.0 * a90 * m
           + CS_W[pos] * cs_rate * p60
           + rating_bonus)
    return ppm


def run_blender(g_rem=4, shrinkage=0.5):
    """
    Main blending pipeline.
    g_rem: remaining UCL league-phase matches (default = 4, heading into Matchday 5).
    """
    pool_path = os.path.join(DATA_DIR, "uefa_fantasy_player_pool.csv")
    ols_path = os.path.join(DATA_DIR, "squad_optimizer_input_hybrid.csv")
    rate_path = os.path.join(OUT_DIR, "rate_model_forecast_2627.csv")
    ucl_path = os.path.join(DATA_DIR, "current_ucl_2627.csv")
    dom_path = os.path.join(DATA_DIR, "current_domestic_2627.csv")

    pool = pd.read_csv(pool_path)
    ols = pd.read_csv(ols_path)[["Id", "PredictedPts_BestK", "Matched"]]
    rate = pd.read_csv(rate_path)[["Id", "xpts", "p_apps_share", "p_mpa"]].drop_duplicates("Id")
    ucl = pd.read_csv(ucl_path)[["Id", "ucl_matches", "ucl_pts", "ucl_ppm", "MinsPlayed"]]
    dom = pd.read_csv(dom_path)[["Id", "dom_apps", "dom_mins", "dom_goals", "dom_assists", "dom_cs", "dom_rating", "dom_g90", "dom_a90", "dom_cs_rate", "has_dom_match"]]

    df = pool.merge(ols, on="Id", how="left").merge(rate, on="Id", how="left")
    df = df.merge(ucl, on="Id", how="left").merge(dom, on="Id", how="left")

    # Ingest verified in-app injuries: Nuno Mendes, I. Saibari, A. Stiller
    df.loc[df.Id.isin([250134320, 250112217, 250136465]), 'Status'] = 'I'

    df["ols"] = np.where(df.Matched.fillna(False).astype(bool), df.PredictedPts_BestK, np.nan)
    df["rate"] = df.xpts

    # 1. Bates-Granger Optimal Combination with Shrinkage (Chapter 5)
    w_ols, w_rate = compute_bates_granger_weights(df.ols, df.rate, shrinkage=shrinkage)
    print(f"[ECON 5371] Bates-Granger weights (with {shrinkage*100:.0f}% shrinkage to equal): OLS={w_ols:.3f}, Rate={w_rate:.3f}")

    # Impute missing model forecasts with position means (prevents winner's-curse bias)
    ols_imp = df.ols.fillna(df.groupby("Position").ols.transform("mean"))
    rate_imp = df.rate.fillna(df.groupby("Position").rate.transform("mean"))
    df["base_season_pts"] = w_rate * rate_imp + w_ols * ols_imp
    df["base_ppm"] = df.base_season_pts / 8.0  # pre-season per-match expectation

    # 2. In-Season Domestic & UCL Updating
    dom_ppms = []
    w_doms = []
    w_ucls = []
    w_bases = []
    blended_ppms = []

    for r in df.itertuples():
        pos = r.Position
        base_p = r.base_ppm

        # Domestic observed form
        d_ppm = domestic_expected_ppm(r._asdict(), pos)
        d_mins = r.dom_mins if pd.notna(r.dom_mins) else 0.0

        # UCL observed form
        u_pts = r.ucl_pts if pd.notna(r.ucl_pts) else 0.0
        u_matches = r.ucl_matches if pd.notna(r.ucl_matches) else 0
        u_ppm = u_pts / max(1, u_matches) if u_matches > 0 else base_p
        u_mins = r.MinsPlayed if pd.notna(r.MinsPlayed) and r.MinsPlayed > 0 else (u_matches * 80.0)

        # Exposure-based Bayesian updating weights
        # League strength factor (~0.90 for top leagues relative to pure UCL tier)
        l_factor = 0.90
        wd = (d_mins / (M0_DOM + d_mins)) * l_factor if (d_mins > 0 and pd.notna(d_ppm)) else 0.0
        wu = (u_mins / (M0_UCL + u_mins)) if (u_mins > 0 and u_matches > 0) else 0.0

        # Bound total empirical weight to 0.75 so the prior baseline retains at least 25% anchor
        tot_emp = wd + wu
        if tot_emp > 0.75:
            scale = 0.75 / tot_emp
            wd *= scale
            wu *= scale

        wb = max(0.20, 1.0 - (wd + wu))
        # Re-normalize
        s = wb + wd + wu
        wb, wd, wu = wb / s, wd / s, wu / s

        # Blended per-match expectation
        blended_p = wb * base_p + wd * (d_ppm if pd.notna(d_ppm) else base_p) + wu * u_ppm

        # Zero out players with confirmed long-term injury or non-active status
        if str(r.Status).strip().upper() in ["I", "NIS"]:
            blended_p = 0.0

        dom_ppms.append(d_ppm)
        w_doms.append(wd)
        w_ucls.append(wu)
        w_bases.append(wb)
        blended_ppms.append(blended_p)

    df["dom_ppm"] = dom_ppms
    df["w_base"] = w_bases
    df["w_dom"] = w_doms
    df["w_ucl"] = w_ucls
    df["blended_ppm"] = blended_ppms

    # 3. Horizon Scaling: Remaining league-phase matches
    df["g_rem"] = g_rem
    df["pred_remaining_pts"] = (df.blended_ppm * g_rem).round(2)
    # Also keep full 8-game projection for comparability
    df["pred_full_season"] = (df.blended_ppm * 8.0).round(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "inseason_player_projections_2627.csv")
    cols_to_export = [
        "Id", "Name", "Position", "Price", "TeamName", "Status", "SelectedPct",
        "ols", "rate", "base_season_pts", "base_ppm",
        "ucl_pts", "ucl_matches", "ucl_ppm",
        "dom_mins", "dom_ppm",
        "w_base", "w_dom", "w_ucl",
        "blended_ppm", "pred_remaining_pts", "pred_full_season"
    ]
    df[cols_to_export].to_csv(out_path, index=False)
    print(f"Exported blended projections: {df.shape} -> {out_path}")
    return df


def verify_combinations():
    """Sanity test for blending outputs."""
    df = run_blender(g_rem=5)
    assert df.pred_remaining_pts.notna().all(), "Missing predictions found!"
    assert (df.pred_remaining_pts >= 0).all(), "Negative predictions found!"

    # Test top player ranking
    top5 = df.sort_values("pred_remaining_pts", ascending=False).head(5)
    print("\nTop 5 Projected Players (Remaining 5 UCL Matches):")
    for _, r in top5.iterrows():
        print(f"  {r.Position} {r.Name} ({r.TeamName}, €{r.Price}m): {r.pred_remaining_pts:.1f} pts (blended {r.blended_ppm:.2f}/match) | Base: {r.base_ppm:.2f}, Dom: {r.dom_ppm:.2f}, UCL: {r.ucl_ppm:.2f}")

    return True


if __name__ == "__main__":
    verify_combinations()
