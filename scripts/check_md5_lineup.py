import urllib.request, json, sys
sys.stdout.reconfigure(encoding='utf-8')

URL = 'https://gaming.uefa.com/en/uclfantasy/services/feeds/players/players_90_en_1.json'
req = urllib.request.Request(URL, headers={'User-Agent': 'fantasy-ucl-personal-research/1.0'})
raw = json.loads(urllib.request.urlopen(req, timeout=30).read())
pl = raw['data']['value']['playerList']

# Find Paz and Douvikas IDs
paz_id = None
douv_id = None
for p in pl:
    if 'Paz' in p.get('pDName', ''):
        paz_id = p['id']
    if 'Douvikas' in p.get('pDName', ''):
        douv_id = p['id']

print(f"Nico Paz ID: {paz_id}, Douvikas ID: {douv_id}")

squad_ids = {
    '250016833': 'H. Kane (BAY)',
    '250109003': 'Troy Parrott (BET)',
    str(douv_id): 'T. Douvikas (COM)',
    '250112880': 'Raphinha (BAR)',
    '250101444': 'Vitinha (PSG)',
    '250078886': 'Bruno Fernandes (MUN)',
    str(paz_id): 'Nico Paz (COM)',
    '250112217': 'A. Stiller (STU) - INJ',
    '250131901': 'Gabriel (ARS)',
    '250134856': 'N. Schlotterbeck (BVB)',
    '250042422': 'A. Grimaldo (ATM)',
    '250056189': 'F. Dimarco (INT)',
    '250136465': 'Nuno Mendes (PSG) - INJ',
    '97923': 'M. Neuer (BAY) - DOUBT',
    '250135068': 'L. Chevalier (LIL)'
}

rows = []
for p in pl:
    pid = str(p['id'])
    if pid in squad_ids:
        up = p.get('upcomingMatchesList', [{}])[0]
        dt = up.get('matchDate', '')
        day = 'Tuesday (Nov 24)' if '11/24' in dt else 'Wednesday (Nov 25)' if '11/25' in dt else 'Unknown'
        rows.append({
            'Player': squad_ids[pid],
            'Pos': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}[p['skill']],
            'Price': p['value'],
            'TotPts': p['totPts'],
            'Day': day,
            'Time': dt,
            'Match': f"{up.get('tLoc')} vs {up.get('vsTSCode')} ({up.get('vsTLoc')})"
        })


import pandas as pd
df_squad = pd.DataFrame(rows).sort_values(['Day', 'Pos'], ascending=[True, False])
print(df_squad.to_string(index=False))
