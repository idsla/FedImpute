import numpy as np


def test_allk_feature_selection_picks_most_correlated_remaining_columns():
    seed = 102020
    rng = np.random.default_rng(seed)
    base = rng.normal(size=30)
    data = rng.normal(size=(30, 5))
    data[:, 0] = base
    data[:, 2] = base + rng.normal(scale=0.01, size=30)
    data[:, 4] = -base + rng.normal(scale=0.01, size=30)

    col = 0
    candidate_cols = [1, 2, 3, 4]
    X = np.concatenate([data[:, col].reshape(-1, 1), data[:, candidate_cols]], axis=1)
    correlations = np.abs(np.corrcoef(X, rowvar=False)[0])
    selected = np.argsort(correlations)[::-1][1:3]

    selected_original_cols = [candidate_cols[idx - 1] for idx in selected]
    assert set(selected_original_cols) == {2, 4}
