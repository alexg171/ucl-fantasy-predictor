import json, urllib.request, sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

URL = 'https://gaming.uefa.com/en/uclfantasy/services/feeds/players/players_90_en_1.json'
req = urllib.request.Request(URL, headers={'User-Agent': 'fantasy-ucl-personal-research/1.0'})
raw = json.loads(urllib.request.urlopen(req, timeout=30).read())
pl = raw['data']['value']['playerList']

# Map team fixture
team_fixtures = {}
for p in pl:
    t = p.get('tName')
    if t and t not in team_fixtures:
        up = p.get('upcomingMatchesList', [])
        if up:
            m = up[0]
            team_fixtures[t] = {
                'loc': m.get('tLoc'),
                'opp': m.get('vsTSCode'),
                'opp_code': m.get('vsCCode'),
                'opp_loc': m.get('vsTLoc'),
                'date': m.get('matchDate')
            }

print('=== ALL MD5 FIXTURES ===')
seen_matches = set()
for t, f in sorted(team_fixtures.items()):
    match_str = f"{t} ({f['loc']}) vs {f['opp']} ({f['opp_loc']}) on {f['date']}"
    # Deduplicate H vs A
    pair = tuple(sorted([t, f['opp']]))
    if pair not in seen_matches:
        seen_matches.add(pair)
        print(match_str)

# Now check candidate forwards
POS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
fwds = []
for p in pl:
    if p.get('skill') == 4: # FWD
        price = p.get('value', 0)
        t = p.get('tName')
        fix = team_fixtures.get(t, {})
        fwds.append({
            'id': p.get('id'),
            'name': p.get('pDName'),
            'team': t,
            'price': price,
            'totPts': p.get('totPts', 0),
            'status': p.get('pStatus'),
            'selPer': p.get('selPer', 0),
            'md5_fix': f"{fix.get('opp', '?')} ({fix.get('loc', '?')})",
            'md5_opp': fix.get('opp'),
            'md5_loc': fix.get('loc'),
            'date': fix.get('date')
        })

df_fwd = pd.DataFrame(fwds)

# Merge with our blended projections if available
try:
    proj = pd.read_csv('out/inseason_player_projections_2627.csv')
    df_fwd['id'] = df_fwd['id'].astype(int)
    proj['Id'] = proj['Id'].astype(int)
    df_fwd = df_fwd.merge(proj[['Id', 'blended_ppm', 'pred_remaining_pts', 'pred_full_season']], left_on='id', right_on='Id', how='left')
except Exception as e:
    print('Proj error:', e)

print('\n=== ALL FORWARDS BUDGET <= 9.5M (Sorted by pred_remaining_pts) ===')
cand = df_fwd[(df_fwd['price'] <= 9.5) & (df_fwd['totPts'] >= 5)].sort_values(by=['pred_remaining_pts', 'totPts'], ascending=False)
print(cand[['name', 'team', 'price', 'totPts', 'md5_fix', 'status', 'pred_remaining_pts', 'pred_full_season']].head(40).to_string(index=False))

