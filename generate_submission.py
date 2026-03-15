"""Generate MTourneyPredictions.csv / WTourneyPredictions.csv style files."""

from __future__ import annotations

import argparse

import pandas as pd

from predictor import EloConfig, EloPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pairwise NCAA matchup predictions.")
    parser.add_argument("--regular-season", required=True, help="Path to compact regular-season results CSV.")
    parser.add_argument("--teams", required=True, help="Path to teams CSV with TeamID column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--season", type=int, default=None, help="Train with seasons <= this value.")
    parser.add_argument("--k-factor", type=float, default=20.0, help="Elo K-factor.")
    parser.add_argument("--home-advantage", type=float, default=80.0, help="Home-court Elo bonus.")
    parser.add_argument("--base-rating", type=float, default=1500.0, help="Initial Elo per team.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    games = pd.read_csv(args.regular_season)
    teams = pd.read_csv(args.teams)

    if "TeamID" not in teams.columns:
        raise ValueError("Teams file must contain a TeamID column")

    if args.season is not None:
        if "Season" not in games.columns:
            raise ValueError("--season was provided but games file has no Season column")
        games = games[games["Season"] <= args.season]

    model = EloPredictor(
        EloConfig(
            base_rating=args.base_rating,
            k_factor=args.k_factor,
            home_advantage=args.home_advantage,
        )
    )
    model.fit(games)

    pred_df = model.build_pairwise_predictions(teams["TeamID"].tolist())
    pred_df.to_csv(args.output, index=False)

    print(f"Wrote {len(pred_df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
