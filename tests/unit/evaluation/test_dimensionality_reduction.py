import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from fedimpute.evaluation import Evaluator
from fedimpute.evaluation.dimensionality_reduction import (
    eval_pca,
    eval_tsne,
    eval_umap,
    visualize_embeddings,
)

pytestmark = pytest.mark.unit


def _sample_client_data():
    origins = [
        pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 2.0]}),
        pd.DataFrame({"a": [2.0, 3.0], "b": [3.0, 4.0]}),
    ]
    imputed = [
        pd.DataFrame({"a": [0.1, 1.1], "b": [0.9, 2.1]}),
        pd.DataFrame({"a": [1.9, 3.1], "b": [3.2, 3.9]}),
    ]
    return origins, imputed


def _assert_valid_embedding_result(embedding, colors, n_original, n_imputed):
    assert embedding.shape == (8, 2)
    assert colors == ["red"] * 4 + ["blue"] * 4
    assert n_original == 4
    assert n_imputed == 4
    assert np.isfinite(embedding).all()


def test_eval_pca_embeds_original_and_imputed_client_data():
    origins, imputed = _sample_client_data()

    embedding, colors, n_original, n_imputed = eval_pca(origins, imputed)

    _assert_valid_embedding_result(embedding, colors, n_original, n_imputed)


def test_eval_tsne_embeds_original_and_imputed_client_data():
    origins, imputed = _sample_client_data()

    embedding, colors, n_original, n_imputed = eval_tsne(
        origins,
        imputed,
        perplexity=2,
        max_iter=250,
        n_iter_without_progress=50,
        n_jobs=1,
    )

    _assert_valid_embedding_result(embedding, colors, n_original, n_imputed)


@pytest.mark.filterwarnings(
    "ignore:using precomputed metric; inverse_transform will be unavailable:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:n_jobs value .* overridden to 1 by setting random_state.*:UserWarning"
)
def test_eval_umap_embeds_original_and_imputed_client_data():
    origins, imputed = _sample_client_data()

    embedding, colors, n_original, n_imputed = eval_umap(
        origins,
        imputed,
        n_neighbors=2,
        n_epochs=10,
        low_memory=True,
    )

    _assert_valid_embedding_result(embedding, colors, n_original, n_imputed)


def test_visualize_embeddings_rejects_unknown_method():
    origins, imputed = _sample_client_data()

    with pytest.raises(ValueError, match="Invalid dimensionality reduction method"):
        visualize_embeddings(X_imps=imputed, X_origins=origins, method="bad-method")


def test_visualize_embeddings_saves_pca_plot(tmp_path):
    origins, imputed = _sample_client_data()
    save_path = tmp_path / "pca.png"

    visualize_embeddings(
        X_imps=imputed,
        X_origins=origins,
        method="pca",
        sampling_size=2,
        save_path=str(save_path),
        fontsize=8,
        method_params={"n_components": 1},
    )

    assert save_path.exists()


def test_evaluator_exposes_dimensionality_visualization_methods():
    evaluator = Evaluator()

    assert callable(evaluator.tsne_visualization)
    assert callable(evaluator.pca_visualization)
    assert callable(evaluator.umap_visualization)
    assert callable(evaluator.dimensionality_visualization)
