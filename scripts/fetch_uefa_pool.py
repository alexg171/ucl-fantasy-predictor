"""Refresh data/uefa_fantasy_player_pool.csv from UEFA Fantasy's live players feed (the same call the site makes).

Feed: https://gaming.uefa.com/en/uclfantasy/services/feeds/players/players_90_en_1.json  (90 = current season)
Columns kept from the old file (Id, Name, Position, Price, TotalPts, Team = UEFA club id) plus TeamName, TeamCode, Status, SelectedPct.
"""
import json, os, urllib.request
import pandas as pd

URL = "https://gaming.uefa.com/en/uclfantasy/services/feeds/players/players_90_en_1.json"
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "uefa_fantasy_player_pool.csv")

req = urllib.request.Request(URL, headers={"User-Agent": "fantasy-ucl-personal-research/1.0"})
pl = json.loads(urllib.request.urlopen(req, timeout=60).read())["data"]["value"]["playerList"]
df = pd.DataFrame([dict(Id=int(p["id"]), Name=p["pDName"], Position=POS[p["skill"]], Price=p["value"], TotalPts=p["totPts"],
                        Team=int(p["tId"]), TeamName=p.get("tName"), TeamCode=p.get("cCode"), Status=p.get("pStatus"),
                        SelectedPct=p.get("selPer")) for p in pl])
old = pd.read_csv(OUT) if os.path.exists(OUT) else None
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print("wrote", df.shape)
if old is not None:
    m = old.merge(df, on="Id", suffixes=("_old", "_new"))
    ch = m[m.Price_old != m.Price_new]
    print("players:", len(old), "->", len(df), "| price changes:", len(ch), "| points changed:", int((m.TotalPts_old != m.TotalPts_new).sum()))
    print(ch[["Name_new", "Price_old", "Price_new"]].head(15).to_string(index=False))
