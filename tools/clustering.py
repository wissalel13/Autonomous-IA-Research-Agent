import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from schemas.clustering import ClusterCandidate, ClusteringResult

ALGORITHMS = ["kmeans", "agglomerative", "gmm"]
ALGO_ORDER = {a: i for i, a in enumerate(ALGORITHMS)}  # préférence : le plus simple d'abord
MAX_ROWS = 5000
STABILITY_MIN = 0.6      # heuristique, à calibrer dans le benchmark
STRUCTURE_MARGIN = 0.10  # écart minimal silhouette réelle - silhouette nulle (idem)
MIN_SPLIT_ROWS = 200  # en dessous, un segment n'est pas subdivisé (verdict trop instable)
MIN_SUB_ROWS = 50     # taille minimale d'un sous-segment (heuristique)
MIN_SUB_SHARE = 0.10  # part minimale d'un sous-segment dans son segment parent (heuristique)
CAT_WEIGHT = 0.3  # poids des colonnes one-hot dans la distance (provisoire, à calibrer dans le benchmark)


def _search(X: np.ndarray, k_range, seed: int):
    """Évalue tous les candidats. Retourne (candidats, meilleur, silhouette nulle, structure) ou None."""
    cands: list[ClusterCandidate] = []
    for name in ALGORITHMS:
        for k in k_range:
            labels = fit_labels(name, k, X, seed)
            if len(set(labels)) != k:  # le modèle a fusionné des groupes : candidat ignoré
                continue
            cands.append(
                ClusterCandidate(
                    algorithm=name,
                    n_clusters=k,
                    silhouette=round(_silhouette(X, labels, seed), 4),
                    davies_bouldin=round(float(davies_bouldin_score(X, labels)), 4),
                    calinski_harabasz=round(float(calinski_harabasz_score(X, labels)), 1),
                    stability_ari=round(_stability(name, k, X, seed), 4),
                )
            )
    if not cands:
        return None
    best = select_best(cands)
    null = _null_silhouette(best.algorithm, best.n_clusters, X, seed)
    has_structure = (best.silhouette - null) >= STRUCTURE_MARGIN and best.stability_ari >= STABILITY_MIN
    return cands, best, null, bool(has_structure)


def _refine_once(X: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, list[str]]:
    """Une passe : subdivise un segment seulement s'il contient lui-même de la structure."""
    new = labels.copy()
    next_id = int(labels.max()) + 1
    notes: list[str] = []
    for k in sorted(set(labels)):
        idx = np.where(labels == k)[0]
        if len(idx) < MIN_SPLIT_ROWS:
            continue
        found = _search(X[idx], range(2, 5), seed)
        if found is None or not found[3]:
            continue
        _, b, null, _ = found
        sub = fit_labels(b.algorithm, b.n_clusters, X[idx], seed)
        smallest = int(np.bincount(sub).min())
        if smallest < max(MIN_SUB_ROWS, int(MIN_SUB_SHARE * len(idx))):
            notes.append(
                f"segment {k} : subdivision écartée "
                f"(sous-groupe de {smallest} lignes, trop petit pour un segment)"
            )
            continue
        for j in range(1, b.n_clusters):
            new[idx[sub == j]] = next_id
            next_id += 1
        notes.append(
            f"segment {k} subdivisé en {b.n_clusters} "
            f"(silhouette {b.silhouette} contre {round(null, 3)} pour des données sans groupes)"
        )
    return new, notes


def prepare_matrix(
    df: pd.DataFrame, max_cardinality: int = 50, cat_weight: float | None = None) -> tuple[np.ndarray, list[str]]:
    """Imputation, standardisation, one-hot. Écarte constantes et colonnes de type identifiant."""
    dropped, cols = [], []
    for c in df.columns:
        s = df[c]
        if s.nunique(dropna=True) <= 1:
            dropped.append(c)
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique() > max_cardinality:
            dropped.append(c)
        else:
            cols.append(c)

    X = df[cols].copy()
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in cols if c not in num]

    prep = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ]
    )
    Z = np.asarray(prep.fit_transform(X), dtype=float)
    Z[:, len(num):] *= CAT_WEIGHT if cat_weight is None else cat_weight
    return Z, dropped


def fit_labels(name: str, k: int, X: np.ndarray, seed: int) -> np.ndarray:
    if name == "kmeans":
        model = KMeans(n_clusters=k, n_init=5, random_state=seed)
    elif name == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    elif name == "gmm":
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    else:
        raise ValueError(f"Algorithme inconnu : {name}")
    return model.fit_predict(X)


def _silhouette(X: np.ndarray, labels: np.ndarray, seed: int) -> float:
    if len(set(labels)) < 2:
        return -1.0
    try:
        return float(
            silhouette_score(X, labels, sample_size=min(len(X), 2000), random_state=seed)
        )
    except ValueError:
        return -1.0


def _stability(name: str, k: int, X: np.ndarray, seed: int, repeats: int = 2) -> float:
    """ARI moyen entre deux clusterings obtenus sur des sous-échantillons de 80 %."""
    rng = np.random.default_rng(seed)
    n, m = len(X), int(0.8 * len(X))
    scores = []
    for r in range(repeats):
        a = np.sort(rng.choice(n, m, replace=False))
        b = np.sort(rng.choice(n, m, replace=False))
        common = np.intersect1d(a, b)
        la = fit_labels(name, k, X[a], seed + r)
        lb = fit_labels(name, k, X[b], seed + r + 100)
        scores.append(
            adjusted_rand_score(la[np.searchsorted(a, common)], lb[np.searchsorted(b, common)])
        )
    return float(np.mean(scores))


def _null_silhouette(name: str, k: int, X: np.ndarray, seed: int, repeats: int = 10) -> float:
    """Silhouette de deux jeux sans groupes ; on garde le plus élevé (nul le plus exigeant).

    1. colonnes mélangées indépendamment : garde les distributions, détruit les corrélations ;
    2. gaussienne de même moyenne et même covariance : garde les corrélations, aucun groupe.
    """
    rng = np.random.default_rng(seed)
    mean, cov = X.mean(axis=0), np.cov(X, rowvar=False)
    perm_vals, gauss_vals = [], []
    for r in range(repeats):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        perm_vals.append(_silhouette(Xp, fit_labels(name, k, Xp, seed + r), seed))
        Xg = rng.multivariate_normal(mean, cov, size=len(X), check_valid="ignore", method="eigh")
        gauss_vals.append(_silhouette(Xg, fit_labels(name, k, Xg, seed + r), seed))
    return float(max(np.mean(perm_vals), np.mean(gauss_vals)))


def select_best(cands: list[ClusterCandidate]) -> ClusterCandidate:
    """Retient, parmi les candidats stables dont la silhouette est à moins de 0,02 du meilleur,
    celui de l'algorithme le plus prioritaire (kmeans, puis agglomerative, puis gmm), puis,
    à algorithme égal, le plus petit k. L'ordre est donc : algorithme d'abord, k ensuite."""
    stable = [c for c in cands if c.stability_ari >= STABILITY_MIN] or cands
    top = max(c.silhouette for c in stable)
    near = [c for c in stable if c.silhouette >= top - 0.02]
    return min(near, key=lambda c: (ALGO_ORDER[c.algorithm], c.n_clusters))

def run_clustering(
    df: pd.DataFrame, k_range: range = range(2, 9), seed: int = 42
) -> ClusteringResult:
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=seed)
    X, dropped = prepare_matrix(df)
    if X.shape[1] < 2:
        raise ValueError("Au moins 2 variables exploitables sont nécessaires pour le clustering.")

    found = _search(X, k_range, seed)
    if found is None:
        raise ValueError("Aucun clustering valide trouvé.")
    cands, best, null, has_structure = found

    labels = fit_labels(best.algorithm, best.n_clusters, X, seed)
    refinements: list[str] = []
    if has_structure:
        labels, refinements = _refine_once(X, labels, seed)

    return ClusteringResult(
        candidates=cands,
        best=best,
        null_silhouette=round(null, 4),
        has_structure=has_structure,
        n_rows_used=len(df),
        dropped_columns=dropped,
        row_index=[int(i) for i in df.index],
        labels=[int(x) for x in labels],
        n_clusters=int(len(set(labels))),
        refinements=refinements,
        cat_weight=CAT_WEIGHT,
    )


