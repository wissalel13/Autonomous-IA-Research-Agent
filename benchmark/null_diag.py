from benchmark.cluster_bench import CASES
from tools.clustering import STABILITY_MIN, STRUCTURE_MARGIN, _null_silhouette, _search, prepare_matrix

if __name__ == "__main__":
    print(f"seuils : marge >= {STRUCTURE_MARGIN}, stabilité >= {STABILITY_MIN}")
    for name in ("wine", "moons"):
        df, _ = CASES[name][0]()
        X, _ = prepare_matrix(df)
        print(f"\n=== {name} ===")
        print("graine | silhouette | nulle(3) | marge(3) | nulle(10) | marge(10) | stabilité")
        for seed in range(6):
            _, best, null3, _ = _search(X, range(2, 9), seed)
            null10 = _null_silhouette(best.algorithm, best.n_clusters, X, seed, repeats=10)
            print(
                f"{seed:>6} | {best.silhouette:.3f}      | {null3:.3f}    | {best.silhouette - null3:+.3f}   "
                f"| {null10:.3f}     | {best.silhouette - null10:+.3f}    | {best.stability_ari:.2f}"
            )