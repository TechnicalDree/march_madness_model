import pandas as pd

from predictor import EloPredictor


def test_pairwise_predictions_count_and_columns():
    games = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "DayNum": [10, 20, 30],
            "WTeamID": [1, 2, 1],
            "LTeamID": [2, 3, 3],
            "WLoc": ["N", "N", "N"],
        }
    )

    model = EloPredictor().fit(games)
    preds = model.build_pairwise_predictions([1, 2, 3])

    assert list(preds.columns) == ["WTeamID", "LTeamID"]
    assert len(preds) == 3


def test_unseen_team_uses_base_rating_and_tie_breaker():
    games = pd.DataFrame(
        {
            "DayNum": [1],
            "WTeamID": [10],
            "LTeamID": [20],
        }
    )

    model = EloPredictor().fit(games)
    winner, loser = model.predict_winner(30, 31)

    assert (winner, loser) == (30, 31)
