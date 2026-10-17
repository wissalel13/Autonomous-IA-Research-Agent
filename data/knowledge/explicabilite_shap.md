# Explicabilité avec SHAP

## Principe
Les valeurs SHAP attribuent à chaque variable une contribution à chaque prédiction, de façon additive : la somme des contributions explique l'écart entre la prédiction et la valeur moyenne.

## Importance globale
La moyenne des valeurs absolues de SHAP par variable donne un classement de l'importance globale. La corrélation entre la valeur d'une variable et sa contribution donne une indication de la direction de l'effet.

## Limite 1 : le modèle, pas la réalité
SHAP explique ce que le modèle utilise, pas ce qui cause le phénomène. Une variable importante n'est pas forcément une cause.

## Limite 2 : variables corrélées
Quand deux variables sont corrélées, le crédit est réparti entre elles, et leurs effets individuels peuvent sembler contradictoires. Il faut les interpréter ensemble.

## Limite 3 : échelles différentes
Les valeurs SHAP d'une régression logistique sont exprimées en log-odds, celles d'une forêt en probabilité. Il faut comparer les classements entre modèles, pas les magnitudes.

## Limite 4 : jeu de référence
Le calcul dépend d'un jeu de données de référence. Pour les arbres, TreeSHAP est exact ; pour les autres modèles, il s'appuie sur un échantillon et reste une approximation.