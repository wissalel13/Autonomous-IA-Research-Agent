# Classes déséquilibrées

## Pourquoi c'est un problème
Quand une classe représente moins d'environ 10 % des exemples, un modèle peut obtenir une accuracy élevée en prédisant toujours la classe majoritaire, sans rien apprendre d'utile.

## Pondération des classes (class_weight)
La pondération augmente le coût des erreurs sur la classe minoritaire sans modifier les données. Elle n'introduit aucun risque de fuite de données et constitue un bon premier choix, surtout quand il y a peu d'exemples.

## SMOTE
SMOTE génère des exemples synthétiques par interpolation entre voisins de la classe minoritaire. Il doit être appliqué uniquement sur les données d'entraînement de chaque fold, à l'intérieur d'un pipeline, jamais avant la validation croisée, sinon les scores sont gonflés. Il peut dégrader les résultats quand il y a très peu d'exemples minoritaires ou beaucoup de variables catégorielles.

## Sous-échantillonnage
Le sous-échantillonnage supprime des exemples de la classe majoritaire. Il est rapide mais jette de l'information : il n'est adapté que lorsqu'il reste assez d'exemples minoritaires, de l'ordre du millier.

## Bonne pratique
Utiliser un découpage stratifié pour conserver la même proportion de classes dans chaque fold et dans le jeu de test.