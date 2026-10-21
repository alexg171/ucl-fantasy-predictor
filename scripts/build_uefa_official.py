"""
Download UEFA Fantasy's archived player feeds for past seasons and map each player to a Transfermarkt player_id.

Feed pattern : https://gaming.uefa.com/en/uclfantasy/services/feeds/players/players_<ID>_en_1.json
Season ids   : 60 = 2023/24, 70 = 2024/25, 80 = 2025/26 (final season totals incl. bonuses). 90 = current season.
               (30/40/50 look like early-matchday snapshots, not full seasons -> not used.)

Mapping to Transfermarkt: within a season, UEFA players with minutes are matched to that season's TM Champions League
stints (data/tm_stints.csv) by normalised name, using minutes to break ties (UEFA and TM minutes agree to within a few
minutes). Unmatched players fall back to fuzzy name similarity among stints with near-identical minutes.

Output: data/uefa_official_ucl.csv
Requires: data/tm_stints.csv (run build_rate_data.py first)
"""
import difflib
import json
import os
import re
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

SEASONS = {60: 2023, 70: 2024, 80: 2025}
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def fetch(feed_id: int) -> list:
    url = f"https://gaming.uefa.com/en/uclfantasy/services/feeds/players/players_{feed_id}_en_1.json"
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-ucl-personal-research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))["data"]["value"]["playerList"]


def main() -> None:
    st = pd.read_csv(os.path.join(DATA, "tm_stints.csv"))
    cl = st[st.group == "CL"].copy()
    cl["nn"] = cl.name.map(norm)
    rows = []
    for feed_id, season in SEASONS.items():
        pl = fetch(feed_id)
        tm = cl[cl.season == season]
        by_name = {k: g for k, g in tm.groupby("nn")}
        used = set()
        recs = []
        for p in pl:
            if not p.get("minsPlyd"):
                continue
            mins = p["minsPlyd"]
            matches = round(p["totPts"] / p["avgPlayerPts"]) if p.get("avgPlayerPts") else np.nan
            rec = dict(season=season, uefa_id=int(p["id"]), name=p.get("pFName") or p["pDName"], short_name=p["pDName"],
                       club_uefa_id=int(p["tId"]), pos=POS[p["skill"]], mins=mins, matches=matches, totPts=p["totPts"],
                       gS=p["gS"], assist=p["assist"], cS=p["cS"], gC=p["gC"], bR=p["bR"], saves=p["saves"], yC=p["yC"],
                       rC=p["rC"], oG=p["oG"], mOMPts=p["mOMPts"], pE=p["pE"], pC=p["pC"], pM=p["pM"],
                       gOB=p.get("gOB", 0), selPer=p.get("selPer"), tm_player_id=np.nan, match_method="")
            cands = None
            for key in (norm(p.get("pFName")), norm(p.get("latinName"))):
                if key and key in by_name:
                    cands = by_name[key]
                    break
            if cands is not None:
                c = cands.assign(d=(cands.minutes - mins).abs()).sort_values("d").iloc[0]
                if c.d <= 60:
                    rec.update(tm_player_id=int(c.player_id), match_method="name+minutes")
            recs.append(rec)
        df = pd.DataFrame(recs)
        used = set(df.tm_player_id.dropna().astype(int))
        # fallback: fuzzy name among unused stints with near-identical minutes
        pool = tm[~tm.player_id.isin(used)]
        # token containment (e.g. UEFA "Nacho" vs TM "Nacho Fernandez") with near-identical minutes, only if unique
        for i in df.index[df.tm_player_id.isna()]:
            r = df.loc[i]
            toks = set(norm(r["name"]).split())
            hit = pool[((pool.minutes - r.mins).abs() <= 3) &
                       pool.nn.map(lambda n: bool(toks) and (toks <= set(n.split()) or set(n.split()) <= toks))]
            if len(hit) == 1:
                df.loc[i, ["tm_player_id", "match_method"]] = [int(hit.iloc[0].player_id), "tokens+minutes"]
                pool = pool[pool.player_id != hit.iloc[0].player_id]
        # fallback: fuzzy name among unused stints with near-identical minutes
        for i in df.index[df.tm_player_id.isna()]:
            r = df.loc[i]
            near = pool[(pool.minutes - r.mins).abs() <= 15]
            best, best_ratio = None, 0.0
            for _, c in near.iterrows():
                ratio = max(difflib.SequenceMatcher(None, norm(r["name"]), c.nn).ratio(),
                            difflib.SequenceMatcher(None, norm(r["short_name"]).split()[-1] if norm(r["short_name"]) else "",
                                                    c.nn.split()[-1] if c.nn else "").ratio() * 0.9)
                if ratio > best_ratio:
                    best, best_ratio = c, ratio
            if best is not None and best_ratio >= 0.62:
                df.loc[i, ["tm_player_id", "match_method"]] = [int(best.player_id), f"fuzzy({best_ratio:.2f})"]
                pool = pool[pool.player_id != best.player_id]
        rows.append(df)
        print(season, "UEFA players with minutes:", len(df), "| matched:", int(df.tm_player_id.notna().sum()))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(DATA, "uefa_official_ucl.csv"), index=False)
    print("wrote", out.shape)


if __name__ == "__main__":
    main()
