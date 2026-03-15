# March Madness Model

Utilities for generating March Madness bracket prediction submissions from historical game results.

## What this repo provides

- An Elo-style rating model trained on regular-season game outcomes.
- A CLI to generate submission files in the competition format:
  - `MTourneyPredictions.csv`
  - `WTourneyPredictions.csv`

The output CSV always has two columns: `WTeamID` and `LTeamID`, and includes one prediction for every unordered pair of teams.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data expectations

The script expects Kaggle-style compact regular season results files containing at least:

- `Season`
- `DayNum`
- `WTeamID`
- `LTeamID`

For teams, provide a CSV with a `TeamID` column.

## Usage

### Men's predictions

```bash
python generate_submission.py \
  --regular-season data/MRegularSeasonCompactResults.csv \
  --teams data/MTeams.csv \
  --output MTourneyPredictions.csv
```

### Women's predictions

```bash
python generate_submission.py \
  --regular-season data/WRegularSeasonCompactResults.csv \
  --teams data/WTeams.csv \
  --output WTourneyPredictions.csv
```

Optional flags:

- `--season 2026` to train only up to a target season.
- `--k-factor 20` to control Elo update sensitivity.
- `--home-advantage 0` if location data is unavailable.

## Notes

- If a team has no history, it starts at the base Elo (`1500`).
- Ties in predicted ratings are broken by smaller team id.
