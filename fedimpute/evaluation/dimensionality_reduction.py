import os
from typing import Dict, List, Tuple, Union

import gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DataInput = Union[
    pd.DataFrame, pd.Series, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]
]
MethodParams = Dict[str, object]

DEFAULT_METHOD_PARAMS = {
    "tsne": {
        "n_components": 2,
        "perplexity": 40,
        "max_iter": 1000,
        "init": "random",
        "n_jobs": -1,
    },
    "pca": {
        "n_components": 2,
        "whiten": False,
        "svd_solver": "auto",
    },
    "umap": {
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "spread": 1.0,
    },
}


def eval_tsne(
    origin_datas: DataInput,
    imputed_datas: DataInput,
    seed: int = 0,
    n_components: int = 2,
    perplexity: float = 40,
    max_iter: int = 1000,
    init: str = "random",
    n_jobs: int = -1,
    **kwargs,
    # learning_rate: Union[str, float] = "auto",
    # early_exaggeration: float = 12.0,
    # n_iter_without_progress: int = 300,
    # n_jobs: int = -1,
) -> Tuple[np.ndarray, List[str], int, int]:
    """
    Embed original and imputed data using t-SNE.

    Parameters
    ----------
    origin_datas : DataInput
        Original ground-truth data. This can be a single data object or a list
        of client-specific data objects.
    imputed_datas : DataInput
        Imputed data aligned with ``origin_datas``.
    seed : int, default=0
        Random seed used by t-SNE.
    n_components : int, default=2
        Number of t-SNE embedding dimensions.
    perplexity : float, default=40
        t-SNE perplexity. It is clamped to a valid value for small datasets.
    max_iter : int, default=1000
        Maximum number of t-SNE optimization iterations.
    init : str, default="random"
        t-SNE initialization method. Passed directly to scikit-learn's TSNE.
    n_jobs : int, default=-1
        Number of parallel jobs for t-SNE optimization. Passed directly to

    Returns
    -------
    tuple
        ``(embedding, colors, n_original, n_imputed)`` where ``embedding`` has
        at least two columns for plotting, ``colors`` labels original and
        imputed rows, and the counts split the embedded rows by source.
    """
    plot_data, colors, n_original, n_imputed = _prepare_embedding_data(
        origin_datas, imputed_datas
    )

    # Gower distances let t-SNE consume mixed numeric/categorical tables through
    # scikit-learn's precomputed metric path.
    distance_matrix = np.clip(gower.gower_matrix(plot_data), 0, 1)
    tsne = TSNE(
        metric="precomputed",
        n_components=n_components,
        verbose=0,
        max_iter=max_iter,
        perplexity=_valid_tsne_perplexity(perplexity, plot_data.shape[0]),
        init=init,
        n_jobs=n_jobs,
        random_state=seed,
        **kwargs
    )
    embedding = tsne.fit_transform(distance_matrix)
    return _ensure_two_columns(embedding), colors, n_original, n_imputed


def eval_pca(
    origin_datas: DataInput,
    imputed_datas: DataInput,
    seed: int = 0,
    n_components: int = 2,
    whiten: bool = False,
    svd_solver: str = "auto",
    **kwargs,
) -> Tuple[np.ndarray, List[str], int, int]:
    """
    Embed original and imputed data using principal component analysis.

    The ``seed`` argument is accepted for API consistency with stochastic
    methods, but PCA itself is deterministic unless method-specific kwargs
    change that behavior.
    """
    plot_data, colors, n_original, n_imputed = _prepare_embedding_data(
        origin_datas, imputed_datas
    )
    feature_matrix = _encode_feature_matrix(plot_data)
    fitted_components = min(
        n_components, feature_matrix.shape[0], feature_matrix.shape[1]
    )
    if fitted_components < 1:
        raise ValueError("PCA requires at least one sample and one feature.")

    pca = PCA(
        n_components=fitted_components,
        whiten=whiten,
        svd_solver=svd_solver,
        random_state=seed if svd_solver == "randomized" else None,
        **kwargs
    )
    embedding = pca.fit_transform(feature_matrix)
    return _ensure_two_columns(embedding), colors, n_original, n_imputed


def eval_umap(
    origin_datas: DataInput,
    imputed_datas: DataInput,
    seed: int = 0,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    **kwargs,
) -> Tuple[np.ndarray, List[str], int, int]:
    """
    Embed original and imputed data using UMAP.

    ``n_neighbors`` is clamped to a valid value for small datasets.
    """
    plot_data, colors, n_original, n_imputed = _prepare_embedding_data(
        origin_datas, imputed_datas
    )

    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "UMAP visualization requires the optional dependency `umap-learn`."
        ) from exc

    distance_matrix = np.clip(gower.gower_matrix(plot_data), 0, 1)
    reducer = UMAP(
        metric="precomputed",
        n_components=n_components,
        n_neighbors=_valid_umap_neighbors(n_neighbors, plot_data.shape[0]),
        min_dist=min_dist,
        spread=spread,
        random_state=seed,
        n_jobs=1,   # UMAP's parallelism is not suported when using random state for reproducibility
        **kwargs
    )
    embedding = reducer.fit_transform(distance_matrix)
    return _ensure_two_columns(embedding), colors, n_original, n_imputed


def plot_embedding(
    embedding: np.ndarray,
    n_original: int,
    n_imputed: int,
    alpha: float = 0.5,
    ax=None,
    color_mapping: dict = None,
):
    """
    Draw original and imputed rows from an existing two-dimensional embedding.

    Parameters
    ----------
    embedding : np.ndarray
        Embedding matrix. The first ``n_original`` rows are original data, and
        the next ``n_imputed`` rows are imputed data.
    n_original : int
        Number of original rows in the embedding.
    n_imputed : int
        Number of imputed rows in the embedding.
    alpha : float, default=0.5
        Scatter marker transparency.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new axes is created when omitted.
    color_mapping : dict, optional
        Mapping with ``"original"`` and ``"imputed"`` color values.
    """
    if ax is None:
        _, ax = plt.subplots()

    if color_mapping is None:
        color_mapping = {
            "original": "red",
            "imputed": "blue",
        }

    ax.scatter(
        embedding[:n_original, 0],
        embedding[:n_original, 1],
        c=color_mapping["original"],
        label="original",
        alpha=alpha,
    )
    ax.scatter(
        embedding[n_original : n_original + n_imputed, 0],
        embedding[n_original : n_original + n_imputed, 1],
        c=color_mapping["imputed"],
        label="imputed",
        alpha=alpha,
    )
    return ax

###############################################################################################
# Main Visualization Function
###############################################################################################

def visualize_embeddings(
    X_imps: List[pd.DataFrame],
    X_origins: List[pd.DataFrame],
    method: str,
    method_params: MethodParams = None,
    fontsize: int = 20,
    alpha: float = 0.5,
    sampling_size: int = None,
    overall: bool = False,
    seed: int = 0,
    save_path: str = None,
):
    """
    Plot client-wise dimensionality reduction views of imputed and original data.

    Parameters
    ----------
    X_imps : list[pd.DataFrame]
        Client-specific imputed datasets.
    X_origins : list[pd.DataFrame]
        Client-specific original datasets aligned with ``X_imps``.
    method : str
        Dimensionality reduction method: ``"tsne"``, ``"pca"``, or ``"umap"``.
    method_params : dict, optional
        Parameters forwarded to the selected embedding method. These values
        override ``DEFAULT_METHOD_PARAMS`` for the requested method.
    fontsize : int, default=20
        Font size for subplot titles and legend.
    alpha : float, default=0.5
        Scatter marker transparency.
    sampling_size : int, optional
        Maximum number of rows sampled per client before embedding.
    overall : bool, default=False
        Whether to append an overall plot using all clients combined.
    seed : int, default=0
        Random seed for sampling and stochastic embedding methods.
    save_path : str, optional
        Path where the figure should be saved. If omitted, the plot is shown.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """

    ###################################################################
    # Input validation and preparation
    ###################################################################
    X_imps = [_to_frame(item) for item in X_imps]
    X_origins = [_to_frame(item) for item in X_origins]

    if len(X_imps) != len(X_origins):
        raise ValueError("X_imps and X_origins must have the same number of clients.")
    if len(X_imps) == 0:
        raise ValueError("At least one client dataset is required for visualization.")

    if overall:
        X_imps = X_imps + [pd.concat(X_imps, axis=0, ignore_index=True)]
        X_origins = X_origins + [pd.concat(X_origins, axis=0, ignore_index=True)]
        titles = [f"Client {i + 1}" for i in range(len(X_imps))]
        titles[-1] = "Overall"
    else:
        titles = [f"Client {i + 1}" for i in range(len(X_imps))]

    if sampling_size is not None:
        rng = np.random.default_rng(seed)
        for i, (X_imp, X_origin) in enumerate(zip(X_imps, X_origins)):
            max_index = min(len(X_imp), len(X_origin))
            sample_count = min(sampling_size, max_index)
            if sample_count < 1:
                raise ValueError(
                    "sampling_size must select at least one row per client."
                )
            # Sample matching rows from original and imputed data so each point
            # pair still refers to the same source record.
            indices = rng.choice(max_index, sample_count, replace=False)
            X_imps[i] = X_imp.iloc[indices].reset_index(drop=True)
            X_origins[i] = X_origin.iloc[indices].reset_index(drop=True)

    n_clients = len(X_imps)
    n_cols = min(5, n_clients)
    n_rows = n_clients // n_cols + (n_clients % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = np.asarray(axs).reshape(-1)
    
    ###########################################################################
    # Validate Dimensionality Reduction Method and Parameters
    ############################################################################
    if method not in {"tsne", "pca", "umap"}:
        raise ValueError(f"Invalid dimensionality reduction method: {method}")
    
    method = method.lower()
    params = DEFAULT_METHOD_PARAMS[method].copy()
    if method_params is not None:
        params.update(method_params)
    method_params = params

    color_mapping = {
        "original": "red",
        "imputed": "blue",
    }

    method_name = method.upper()
    for i in range(n_clients):
        print(f"Evaluating {method_name} for {titles[i]} ...")

        #################################################################
        # Dimensionality reduction
        #################################################################
        if method == "tsne":
            embedding, _, n_original, n_imputed = eval_tsne(
                X_origins[i], X_imps[i], seed=seed, **method_params
            )
        elif method == "pca":
            embedding, _, n_original, n_imputed = eval_pca(
                X_origins[i], X_imps[i], seed=seed, **method_params
            )
        else:
            embedding, _, n_original, n_imputed = eval_umap(
                X_origins[i], X_imps[i], seed=seed, **method_params
            )

        #################################################################
        # Plotting
        #################################################################
        plot_embedding(
            embedding,
            n_original,
            n_imputed,
            alpha=alpha,
            ax=axs[i],
            color_mapping=color_mapping,
        )
        axs[i].set_title(titles[i], fontsize=fontsize, fontweight="bold")
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    #######################################################################
    # Adjust layout and add legend
    #######################################################################
    for i in range(n_clients, len(axs)):
        axs[i].set_visible(False)

    marker_size = max(fontsize - 3, 1)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_mapping["original"],
            markersize=marker_size,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_mapping["imputed"],
            markersize=marker_size,
        ),
    ]

    fig.legend(
        legend_elements,
        ["Original", "Imputed"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.1),
        prop={"weight": "bold", "size": fontsize},
        frameon=False,
    )
    plt.subplots_adjust(wspace=0.0)
    plt.tight_layout()
    if save_path is not None:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig


###############################################################################################
# Helpers Functions
###############################################################################################


def _prepare_embedding_data(
    origin_datas: DataInput,
    imputed_datas: DataInput,
) -> Tuple[pd.DataFrame, List[str], int, int]:
    """Concatenate data and build row labels shared by all embedding methods."""
    origin_data = _concat_data(origin_datas)
    imputed_data = _concat_data(imputed_datas)
    plot_data = pd.concat([origin_data, imputed_data], axis=0, ignore_index=True)
    n_original = origin_data.shape[0]
    n_imputed = imputed_data.shape[0]
    colors = ["red" for _ in range(n_original)] + ["blue" for _ in range(n_imputed)]

    return plot_data, colors, n_original, n_imputed


def _concat_data(datas: DataInput) -> pd.DataFrame:
    """Convert one or more tabular inputs into a single dataframe."""
    if isinstance(datas, (pd.DataFrame, pd.Series, np.ndarray)):
        return _to_frame(datas)

    return pd.concat([_to_frame(item) for item in datas], axis=0, ignore_index=True)


def _to_frame(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> pd.DataFrame:
    """Convert a supported tabular object to a row-indexed dataframe."""
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=True)
    if isinstance(data, pd.Series):
        return data.to_frame().reset_index(drop=True)
    return pd.DataFrame(np.asarray(data)).reset_index(drop=True)


def _encode_feature_matrix(plot_data: pd.DataFrame) -> np.ndarray:
    """One-hot encode, impute, and scale data before PCA."""
    # PCA needs numeric feature vectors rather than a precomputed distance
    # matrix, so categorical columns are expanded before scaling.
    feature_data = pd.get_dummies(plot_data, dummy_na=True)
    feature_data = feature_data.apply(pd.to_numeric, errors="coerce")
    feature_data = feature_data.fillna(feature_data.mean(numeric_only=True)).fillna(0.0)

    scaler = StandardScaler()
    return scaler.fit_transform(feature_data)


def _ensure_two_columns(embedding: np.ndarray) -> np.ndarray:
    """Pad one-dimensional embeddings so downstream plotting can use x/y axes."""
    if embedding.shape[1] >= 2:
        return embedding

    padding = np.zeros((embedding.shape[0], 2 - embedding.shape[1]))
    return np.concatenate([embedding, padding], axis=1)


def _valid_tsne_perplexity(perplexity: float, n_samples: int) -> float:
    """Clamp t-SNE perplexity to a value valid for the sample size."""
    if n_samples < 2:
        raise ValueError("t-SNE requires at least two samples.")

    return min(perplexity, max(1, n_samples - 1))


def _valid_umap_neighbors(n_neighbors: int, n_samples: int) -> int:
    """Clamp UMAP neighbor count to a value valid for the sample size."""
    if n_samples < 2:
        raise ValueError("UMAP requires at least two samples.")

    return min(n_neighbors, n_samples - 1)
