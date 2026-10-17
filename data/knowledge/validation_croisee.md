# Validation croisée et fuite de données

## Principe
La validation croisée découpe les données d'entraînement en k parties : le modèle est entraîné sur k-1 parties et évalué sur la dernière, puis on fait la moyenne. L'écart-type des scores indique la stabilité de l'estimation.

## Fuite de données (data leakage)
Il y a fuite quand de l'information du jeu de validation ou de test influence l'entraînement. Les causes fréquentes : normaliser ou imputer avant le découpage, rééquilibrer (SMOTE) avant la validation croisée, ou sélectionner des variables sur l'ensemble des données. Les scores deviennent alors trop optimistes.

## Pipeline
Mettre le prétraitement, le rééquilibrage et le modèle dans un seul pipeline garantit que chaque étape est ajustée uniquement sur les données d'entraînement de chaque fold.

## Découpage stratifié
Pour la classification, un découpage stratifié conserve la proportion de classes dans chaque fold. C'est indispensable quand la classe positive est rare.

## Jeu de test
Le jeu de test sert à une seule évaluation finale. Avec peu d'exemples positifs, le score de test est très variable : il faut le lire avec prudence et privilégier la moyenne de validation croisée accompagnée de son écart-type.

## Optimisation des hyperparamètres
Le score de validation croisée d'un modèle dont les hyperparamètres ont été choisis sur ces mêmes folds est légèrement optimiste. Une validation croisée imbriquée (nested) donne une estimation plus honnête, au prix d'un temps de calcul plus élevé.