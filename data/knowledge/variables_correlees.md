# Colonnes constantes, corrélées et à forte cardinalité

## Colonnes constantes
Une colonne qui prend une seule valeur n'apporte aucune information : elle peut être supprimée sans risque.

## Variables fortement corrélées
Deux variables quasi identiques (corrélation proche de 1) sont redondantes. En garder une suffit. Elles posent surtout problème pour les modèles linéaires (coefficients instables) et pour l'interprétation : l'importance se répartit entre elles. Il ne faut jamais supprimer les deux d'une même paire.

## Corrélation modérée
Une corrélation élevée mais inférieure à 1 n'implique pas que l'une des variables soit inutile : la différence entre deux variables corrélées peut être informative. Il vaut mieux tester que supprimer d'office.

## Forte cardinalité
Une variable catégorielle avec de très nombreuses modalités produit, avec un encodage one-hot, beaucoup de colonnes éparses et favorise le surapprentissage. Les identifiants uniques, par exemple, n'ont aucune valeur prédictive. Des alternatives : regrouper les modalités rares ou utiliser un encodage cible, appliqué à l'intérieur de la validation croisée.