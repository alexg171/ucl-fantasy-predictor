import json, urllib.request, sys
sys.stdout.reconfigure(encoding='utf-8')
URL = 'https://gaming.uefa.com/en/uclfantasy/services/feeds/players/players_90_en_1.json'
req = urllib.request.Request(URL, headers={'User-Agent': 'fantasy-ucl-personal-research/1.0'})
raw = json.loads(urllib.request.urlopen(req, timeout=30).read())
pl = raw['data']['value']['playerList']
targets = ['Gyökeres', 'Suárez', 'Torres', 'Douvikas', 'Tresoldi', 'Guirassy', 'Malen', 'Šeško', 'Kane', 'David', 'Adeyemi', 'Thuram', 'Šturm']
for p in pl:
    for t in targets:
        if t in p.get('pDName', ''):
            print(f"{p['pDName']:15s} | {p['tName']:12s} | Price: €{p['value']}m | TotPts: {p['totPts']:2d} | Mins: {p.get('minsPlyd',0):3d} | G: {p.get('gS',0)} | A: {p.get('assist',0)} | Status: {p.get('pStatus')}")
