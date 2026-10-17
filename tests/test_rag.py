from tools.rag import EMBED_MODEL, build_index, search

IN_DOMAIN = [
    ("que faire quand il y a très peu de positifs ?", "desequilibre"),
    ("pourquoi l'accuracy est trompeuse", "metriques"),
    ("quand utiliser le sous-échantillonnage", "desequilibre"),
    ("faut-il imputer par la moyenne ou la médiane", "valeurs_manquantes"),
    ("deux colonnes très corrélées, laquelle supprimer", "variables_correlees"),
    ("comment interpréter SHAP avec des variables corrélées", "explicabilite_shap"),
    ("pourquoi le score de test est instable", "validation_croisee"),
    ("quel modèle choisir sur peu de lignes", "choix_modeles"),
]
OFF_TOPIC = ["recette de cuisine", "résultats du match de football", "histoire de la Rome antique"]

if __name__ == "__main__":
    print(f"Modèle d'embeddings : {EMBED_MODEL}")
    print(f"{build_index()} passages indexés\n")

    top1 = top2 = 0
    in_dists = []
    for q, expected in IN_DOMAIN:
        hits = search(q, k=2)
        ok1 = hits[0]["source"] == expected
        ok2 = ok1 or hits[1]["source"] == expected
        top1 += ok1
        top2 += ok2
        in_dists.append(hits[0]["distance"])
        print(f"{'OK ' if ok1 else 'KO '} {q}")
        print(f"     -> {hits[0]['id']} ({hits[0]['section']}) dist={hits[0]['distance']}  attendu={expected}")

    print(f"\nTop-1 : {top1}/{len(IN_DOMAIN)}   Top-2 : {top2}/{len(IN_DOMAIN)}")

    off_dists = []
    print("\nHors sujet :")
    for q in OFF_TOPIC:
        h = search(q, k=1)[0]
        off_dists.append(h["distance"])
        print(f"  {q:<35} dist={h['distance']}  ({h['id']})")

    print(f"\nDistance top-1 en domaine : {min(in_dists)} - {max(in_dists)}")
    print(f"Distance top-1 hors sujet : {min(off_dists)} - {max(off_dists)}")