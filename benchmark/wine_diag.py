from benchmark.cluster_bench import CASES
from tools.clustering import _search, prepare_matrix

if __name__ == "__main__":
    df, _ = CASES["wine"][0]()
    X, _ = prepare_matrix(df)
    for seed in range(6):
        cands, best, null, _ = _search(X, range(2, 9), seed)
        top = max(c.silhouette for c in cands)
        near = sorted([c for c in cands if c.silhouette >= top - 0.03], key=lambda c: -c.silhouette)
        print(
            f"\ngraine {seed} : retenu = {best.algorithm} k={best.n_clusters} "
            f"(silhouette {best.silhouette}, nulle {null:.3f}, marge {best.silhouette - null:+.3f})"
        )
        for c in near:
            print(f"    {c.algorithm:<14} k={c.n_clusters}  sil={c.silhouette:.3f}  stab={c.stability_ari:.2f}")