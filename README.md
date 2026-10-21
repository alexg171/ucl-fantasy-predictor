# UEFA Champions League Fantasy — Analysis & Squad Optimizer

A data-driven pipeline for the 2026/27 UEFA Champions League Fantasy game: pulls real player statistics (current-season and historical), fits a position-specific OLS model to predict UCL fantasy points from domestic-league form, and uses that model to optimize a 15-player squad under the game's real budget/position/club constraints.

## Repo layout

```
fantasy ucl/
├── README.md                      — this file
├── notes.md                       — UEFA Fantasy rules & scoring reference (squad rules, transfers, chips, full scoring table)
├── ols_fantasy_model.ipynb        — builds and validates the prediction model
├── squad_optimizer.ipynb          — applies the model to pick an optimal squad (ILP)
├── data/                          — inputs the notebooks read (raw/joined datasets, reference tables)
└── out/                           — results the notebooks write (not re-read by anything downstream)
```

**Run notebooks from the project root** (`fantasy ucl/`) — they use relative `data/`/`out/` paths.

## The two notebooks

### 1. `ols_fantasy_model.ipynb`
Builds four separate OLS regressions (one per position: GK/DEF/MID/FWD) predicting a player's **UEFA Champions League fantasy points** from their **domestic-league performance**. Covers:
- Why the target has to be *reconstructed* (UEFA doesn't publish historical fantasy points — see Methodology section in the notebook)
- The lag correction: domestic season *S* predicts UCL season *S+1* — not same-season, which was tried first and shown to be mostly timing coincidence, not real signal
- Full coefficient tables (β, robust SE, p-values) per position
- Whether more historical seasons help — answer: it depends on position (DEF/FWD improve with more history, MID/GK get worse)
- The resulting 2026/27 predictions for every current player

### 2. `squad_optimizer.ipynb`
Takes those predictions and picks the actual best 15-man squad via Integer Linear Programming (ILP, solved with `PuLP`/CBC) — maximizing total predicted points subject to:
- €100m budget
- Exactly 2 GK / 5 DEF / 5 MID / 3 FWD
- Max 3 players from any one club

Uses a **hybrid** version of the model: each position scored by whichever trailing-history window (1 season vs. up to 4, averaged) actually fits that position best, per the first notebook's findings.

## Data sources
- **UEFA Fantasy's own feed** (`gaming.uefa.com/.../services/feeds/players/...json`) — the live player pool, prices, positions, current-season points.
- **Sofascore's statistics API** — domestic-league and UCL match stats, current and historical (2018/19–2025/26), across 16 leagues covering every club in the 2026/27 competition.
- **UEFA's own coefficient rankings** (`uefa.com/nationalassociations/uefarankings/country`) — used as a league-strength control in the model.

## Known limitations (see each notebook's own "Limitations" section for detail)
- Fantasy points are reconstructed from match stats, not official — missing Player of the Match, penalty, and card bonuses.
- No age, domestic-cup, current-season, fixture-difficulty, or transfer/new-club adjustment in the model.
- Coverage isn't 100% — roughly 230 of 1,163 current players have no clean historical match and are excluded rather than scored as zero.

## Requirements
`pandas`, `numpy`, `statsmodels`, `matplotlib`, `pulp` — all standard, installable via `pip install pandas numpy statsmodels matplotlib pulp`.

## Rate-forecast model (`rate_forecast_model.ipynb`)
Forecasts per-game rates (goals/90, assists/90, minutes per appearance, availability, ...) for a player's UCL season from his domestic-league season, then converts them to expected fantasy points. Expanding-window out-of-sample evaluation, validation against official UEFA season totals, and a 2026/27 forecast (`out/rate_model_forecast_2627.csv`).

- `scripts/build_rate_data.py` builds `data/tm_stints.csv` and `data/tm_club_dom_context.csv` from the Transfermarkt open dataset (CC0; download link in the script docstring).
- `scripts/build_uefa_official.py` builds `data/uefa_official_ucl.csv` (UEFA archived season feeds mapped to Transfermarkt ids).
- `scripts/ratemodel.py` is the modelling module; `scripts/make_rate_notebook.py` regenerates the notebook.

## Transfer optimizer (`transfer_optimizer.ipynb`)
Averages the OLS hybrid and rate-model forecasts (missing forecasts imputed with the position mean), then solves an ILP for the best transfers from your current squad (`MY_SQUAD` in the notebook) under budget, position, club and transfer-cost constraints. Regenerate with `scripts/make_transfer_notebook.py`.
