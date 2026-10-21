"""
Build compact player-season "stint" tables from the Transfermarkt open dataset.

Source (CC0): https://github.com/dcaribou/transfermarkt-datasets
Download   : https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip  (~241 MB)
Unzip it to any folder (the tables are *.csv.gz) and pass that folder as --tm-dir.

Outputs (written to data/):
  tm_stints.csv            one row per player x season x club x competition-group (DOM = domestic league, CL = Champions League)
  tm_club_dom_context.csv  one row per club x season: domestic goals for/against per game, points per game

Per-appearance definitions (approximations, see the notebook for caveats):
  apps60      appearance with >= 60 minutes (fantasy "60 minutes" point)
  cs          clean sheet = team conceded 0 in the game AND player played >= 60 minutes
  gc_onpitch  goals conceded by the team in the game, scaled by minutes/90 (pro-rata, no minute-of-goal info used)

Usage:  python scripts/build_rate_data.py --tm-dir <folder with the csv.gz tables>
"""
import argparse
import os

import numpy as np
import pandas as pd

DOMESTIC = ["GB1", "L1", "ES1", "FR1", "IT1", "NL1", "TR1", "PO1", "BE1", "GR1", "UKR1", "A1", "NO1", "TS1"]
UCL = "CL"
MIN_SEASON, MAX_SEASON = 2017, 2025  # season = start year (2025 -> 2025/26)


def main(tm_dir: str, out_dir: str) -> None:
    rd = lambda name, **kw: pd.read_csv(os.path.join(tm_dir, name), **kw)

    games = rd("games.csv.gz", usecols=["game_id", "competition_id", "season"])
    games = games[games.competition_id.isin(DOMESTIC + [UCL]) & games.season.between(MIN_SEASON, MAX_SEASON)]

    cg = rd("club_games.csv.gz", usecols=["game_id", "club_id", "own_goals", "opponent_goals", "is_win"])
    cg = cg.merge(games, on="game_id", how="inner")
    cg["group"] = np.where(cg.competition_id == UCL, "CL", "DOM")
    cg["pts"] = np.where(cg.own_goals > cg.opponent_goals, 3, np.where(cg.own_goals == cg.opponent_goals, 1, 0))

    # ---- club x season domestic context ----
    dom = cg[cg.group == "DOM"]
    club_ctx = (dom.groupby(["club_id", "season", "competition_id"])
                .agg(games=("game_id", "nunique"), gf=("own_goals", "sum"), ga=("opponent_goals", "sum"), pts=("pts", "sum"))
                .reset_index())
    club_ctx["gf_pg"] = club_ctx.gf / club_ctx.games
    club_ctx["ga_pg"] = club_ctx.ga / club_ctx.games
    club_ctx["ppg"] = club_ctx.pts / club_ctx.games
    club_ctx.rename(columns={"competition_id": "league"}, inplace=True)

    # ---- club games per season x group (denominator for availability) ----
    club_games = (cg.groupby(["club_id", "season", "group"]).game_id.nunique().rename("club_games").reset_index())

    # ---- appearances ----
    ap = rd("appearances.csv.gz", usecols=["game_id", "player_id", "player_club_id", "competition_id", "yellow_cards",
                                           "red_cards", "goals", "assists", "minutes_played"])
    ap = ap[ap.competition_id.isin(DOMESTIC + [UCL])]
    ap = ap.merge(games[["game_id", "season"]], on="game_id", how="inner")
    ap = ap.merge(cg[["game_id", "club_id", "opponent_goals"]], left_on=["game_id", "player_club_id"],
                  right_on=["game_id", "club_id"], how="left").drop(columns="club_id")
    ap["group"] = np.where(ap.competition_id == UCL, "CL", "DOM")
    ap["apps60"] = (ap.minutes_played >= 60).astype(int)
    ap["cs"] = ((ap.opponent_goals == 0) & (ap.minutes_played >= 60)).astype(int)
    ap["gc_onpitch"] = ap.opponent_goals.fillna(0) * ap.minutes_played / 90.0

    stints = (ap.groupby(["player_id", "season", "group", "player_club_id"])
              .agg(league=("competition_id", lambda s: s.mode().iat[0]),
                   apps=("game_id", "nunique"), minutes=("minutes_played", "sum"), apps60=("apps60", "sum"),
                   goals=("goals", "sum"), assists=("assists", "sum"), yc=("yellow_cards", "sum"),
                   rc=("red_cards", "sum"), cs=("cs", "sum"), gc_onpitch=("gc_onpitch", "sum"))
              .reset_index().rename(columns={"player_club_id": "club_id"}))
    stints = stints.merge(club_games, on=["club_id", "season", "group"], how="left")
    # primary stint = club where the player logged most minutes in that season & group
    stints["is_primary"] = (stints.groupby(["player_id", "season", "group"]).minutes.transform("max") == stints.minutes)
    stints = stints.sort_values(["player_id", "season", "group", "minutes"], ascending=[True, True, True, False])
    stints = stints.drop_duplicates(["player_id", "season", "group", "is_primary"]).query("is_primary").drop(columns="is_primary")

    # ---- player attributes ----
    pl = rd("players.csv.gz", usecols=["player_id", "name", "position", "sub_position", "date_of_birth"])
    pl["date_of_birth"] = pd.to_datetime(pl.date_of_birth, errors="coerce")
    stints = stints.merge(pl, on="player_id", how="left")
    season_start = pd.to_datetime(stints.season.astype(str) + "-08-01")
    stints["age"] = ((season_start - stints.date_of_birth).dt.days / 365.25).round(2)

    os.makedirs(out_dir, exist_ok=True)
    stints.drop(columns="date_of_birth").to_csv(os.path.join(out_dir, "tm_stints.csv"), index=False)
    club_ctx.to_csv(os.path.join(out_dir, "tm_club_dom_context.csv"), index=False)
    print("stints:", stints.shape, "| by group:", stints.group.value_counts().to_dict())
    print("club context:", club_ctx.shape)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--tm-dir", required=True)
    ap_.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    a = ap_.parse_args()
    main(a.tm_dir, a.out_dir)
