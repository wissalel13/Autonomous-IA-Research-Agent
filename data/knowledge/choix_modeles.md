# Choix des modèles sur données tabulaires

## Toujours commencer par une baseline
Un modèle linéaire (régression logistique ou ridge) est rapide, interprétable et difficile à surapprendre. Si un modèle complexe ne fait pas nettement mieux, la baseline est préférable.

## Régression logistique et ridge
Efficaces quand la relation est proche du linéaire et que les données sont peu nombreuses. Ils demandent des variables mises à l'échelle. Avec des variables fortement corrélées, les coefficients deviennent instables, mais la régularisation atténue ce problème.

## Random forest
Robuste, peu sensible aux échelles et aux valeurs extrêmes, avec peu de réglages. Bon choix par défaut quand les relations sont non linéaires.

## Gradient boosting (XGBoost, LightGBM)
Très performants sur des volumes moyens à grands. Sur peu de lignes, ils surapprennent facilement : il faut limiter la profondeur, réduire le taux d'apprentissage et régulariser.

## Comparer honnêtement
Quand l'écart entre deux modèles est inférieur à l'écart-type de la validation croisée, ils sont statistiquement indiscernables. On retient alors le plus simple.


## Extra trees
Les extra trees ressemblent à la forêt aléatoire mais choisissent les seuils de découpage au hasard, ce qui les rend plus rapides et parfois plus stables sur des données bruitées. Ils sont un bon second modèle d'arbres à comparer à la forêt.

## Gradient boosting natif de scikit-learn
HistGradientBoosting regroupe les valeurs en classes avant d'entraîner les arbres, ce qui le rend rapide et capable de gérer les valeurs manquantes. Il fait partie des modèles de gradient boosting et supporte la pondération des classes.

## SVM
Un SVM à noyau cherche une frontière lisse entre les classes et fonctionne bien sur des jeux de données de petite taille. Son temps d'entraînement augmente très vite avec le nombre de lignes, et il exige des variables mises à l'échelle.

## k plus proches voisins
Le k-NN prédit à partir des exemples les plus proches et ne demande aucun entraînement. Il est sensible aux variables non pertinentes, au bruit et aux grandes dimensions, et il ne supporte pas la pondération des classes.