# Choix des métriques

## Accuracy
L'accuracy est trompeuse quand les classes sont déséquilibrées : prédire toujours la classe majoritaire donne un score élevé.

## ROC AUC
L'ROC AUC peut être trop optimiste quand la classe positive est rare, car le taux de faux positifs est dilué par le grand nombre de négatifs.

## Average precision
L'average precision (aire sous la courbe précision-rappel) reflète mieux la qualité sur la classe rare. Un classifieur aléatoire obtient une valeur égale à la proportion de positifs, ce qui sert de repère pour juger un score.

## F1
Le F1 combine précision et rappel, mais dépend du seuil de décision choisi.

## Régression
Le RMSE pénalise fortement les grosses erreurs, le MAE est plus robuste aux valeurs extrêmes, et le R² mesure la part de variance expliquée.


## Balanced accuracy
La balanced accuracy est la moyenne du rappel de chaque classe : elle donne le même poids à chaque classe, quelle que soit sa taille, et évite l'effet trompeur de l'accuracy quand les classes sont déséquilibrées.

## Multiclasse
Avec plus de deux classes, le F1 est calculé par classe puis moyenné. La moyenne macro donne le même poids à chaque classe et révèle les faiblesses sur les classes rares, alors que la moyenne pondérée favorise les classes fréquentes. L'average precision n'a pas de version native multiclasse : on utilise le F1 macro ou la balanced accuracy.