"""Elo-based winner prediction for NCAA matchups."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Tuple

import pandas as pd


@dataclass
class EloConfig:
    base_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 80.0


class EloPredictor:
    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self.ratings: Dict[int, float] = {}

    def _ensure_team(self, team_id: int) -> None:
        if team_id not in self.ratings:
            self.ratings[team_id] = self.config.base_rating

    def _expected_win_prob(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def fit(self, games: pd.DataFrame) -> "EloPredictor":
        required = {"DayNum", "WTeamID", "LTeamID"}
        missing = required.difference(games.columns)
        if missing:
            raise ValueError(f"games is missing required columns: {sorted(missing)}")

        if "WLoc" in games.columns:
            sort_cols = [c for c in ["Season", "DayNum"] if c in games.columns]
        else:
            sort_cols = [c for c in ["Season", "DayNum"] if c in games.columns]

        ordered_games = games.sort_values(sort_cols or ["DayNum"])

        for row in ordered_games.itertuples(index=False):
            winner = int(row.WTeamID)
            loser = int(row.LTeamID)
            self._ensure_team(winner)
            self._ensure_team(loser)

            winner_rating = self.ratings[winner]
            loser_rating = self.ratings[loser]

            if hasattr(row, "WLoc"):
                wloc = getattr(row, "WLoc")
                if wloc == "H":
                    winner_rating_adj = winner_rating + self.config.home_advantage
                    loser_rating_adj = loser_rating
                elif wloc == "A":
                    winner_rating_adj = winner_rating
                    loser_rating_adj = loser_rating + self.config.home_advantage
                else:
                    winner_rating_adj = winner_rating
                    loser_rating_adj = loser_rating
            else:
                winner_rating_adj = winner_rating
                loser_rating_adj = loser_rating

            exp_win = self._expected_win_prob(winner_rating_adj, loser_rating_adj)
            delta = self.config.k_factor * (1.0 - exp_win)

            self.ratings[winner] += delta
            self.ratings[loser] -= delta

        return self

    def predict_winner(self, team_a: int, team_b: int) -> Tuple[int, int]:
        self._ensure_team(team_a)
        self._ensure_team(team_b)
        rating_a = self.ratings[team_a]
        rating_b = self.ratings[team_b]

        if rating_a > rating_b:
            return team_a, team_b
        if rating_b > rating_a:
            return team_b, team_a
        return (team_a, team_b) if team_a < team_b else (team_b, team_a)

    def build_pairwise_predictions(self, team_ids: Iterable[int]) -> pd.DataFrame:
        rows = []
        for team_a, team_b in combinations(sorted(set(int(t) for t in team_ids)), 2):
            winner, loser = self.predict_winner(team_a, team_b)
            rows.append({"WTeamID": winner, "LTeamID": loser})
        return pd.DataFrame(rows, columns=["WTeamID", "LTeamID"])
