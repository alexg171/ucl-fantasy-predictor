import pandas as pd
import json

df = pd.read_csv('out/inseason_player_projections_2627.csv')

# Candidates
names = ['V. Gyökeres', 'L. Suárez', 'T. Douvikas', 'Ferran Torres', 'S. Guirassy', 'D. Malen', 'N. Tresoldi', 'B. Mbeumo']

subset = df[df['Name'].isin(names)].copy()

# Add fixture info
fixtures = {
    'V. Gyökeres': ('Arsenal', 'Dortmund (H)', 'Positive correlation with Gabriel (ARS DEF); Arsenal heavy favorite at home'),
    'L. Suárez': ('Sporting CP', 'Man Utd (H)', 'Targets struggling Man Utd in Lisbon; 29 UCL pts, 4 goals, primary penalty taker'),
    'T. Douvikas': ('Como', 'AEK Athens (H)', 'Direct receiver of Nico Paz assists; 37 UCL pts (6 goals); highest scorer in bracket'),
    'Ferran Torres': ('Paris', 'Roma (H)', 'Stacks with Vitinha (PSG MID) at home; 20 UCL pts (3 goals in 188 mins)'),
    'S. Guirassy': ('B. Dortmund', 'Arsenal (A)', 'CLASH: Plays against Gabriel (Arsenal DEF) away at Emirates; severe negative covariance'),
    'D. Malen': ('Roma', 'Paris (A)', 'Roma away at Parc des Princes against PSG defense; user not sold'),
    'N. Tresoldi': ('Club Brugge', 'Liverpool (H)', 'CLASH: Plays against Ronald Araújo (Liverpool DEF); negative covariance'),
    'B. Mbeumo': ('Man Utd', 'Sporting CP (A)', 'CURRENT OWNERSHIP: Man Utd horrific form, tough away fixture in Lisbon; SELL candidate')
}

rows = []
for name, (team, fix, note) in fixtures.items():
    p = df[df['Name'] == name]
    if len(p) > 0:
        row = p.iloc[0]
        rows.append({
            'Name': name,
            'Team': team,
            'Price (€m)': row['Price'],
            'UCL Pts': row['ucl_pts'],
            'MD5 Fixture': fix,
            'Pred MD5-8 Pts': round(row['pred_remaining_pts'], 1),
            'Pred Full Pts': round(row['pred_full_season'], 1),
            'Tactical Evaluation': note
        })

res = pd.DataFrame(rows)
print(res.to_string(index=False))
