import urllib.request, json, sys

sys.stdout.reconfigure(encoding='utf-8')

# Let's inspect the feed for upcoming fixtures or check matchdays
URL = 'https://gaming.uefa.com/en/uclfantasy/services/feeds/players/players_90_en_1.json'
req = urllib.request.Request(URL, headers={'User-Agent': 'fantasy-ucl-personal-research/1.0'})
raw = json.loads(urllib.request.urlopen(req, timeout=30).read())
pl = raw['data']['value']['playerList']

# Check what upcoming matches are listed per player
# Do they list multiple matches or just next match?
for p in pl[:10]:
    if p.get('upcomingMatchesList'):
        print(p['pDName'], p['tName'], len(p['upcomingMatchesList']), [m.get('vsTSCode') for m in p['upcomingMatchesList']])
        break
