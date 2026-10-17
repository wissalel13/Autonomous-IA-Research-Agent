# Valeurs manquantes

## Comprendre l'origine
Les valeurs peuvent manquer au hasard, à cause d'une autre variable observée, ou à cause de la valeur manquante elle-même. Dans ce dernier cas, le fait qu'une valeur soit absente est lui-même informatif.

## Imputation numérique
La médiane est préférable à la moyenne quand la variable est asymétrique ou contient des valeurs extrêmes, car elle y est peu sensible. La moyenne convient pour des distributions à peu près symétriques.

## Imputation catégorielle
On remplace en général par la modalité la plus fréquente, ou par une catégorie « inconnu » si l'absence est informative.

## Indicateur de valeur manquante
Ajouter une colonne binaire indiquant si la valeur était absente permet au modèle d'exploiter cette information.

## Éviter la fuite
L'imputation doit être ajustée uniquement sur les données d'entraînement, à l'intérieur d'un pipeline. Calculer la médiane sur tout le jeu de données avant le découpage introduit une fuite.

## Colonnes très incomplètes
Quand une colonne est manquante en grande proportion, par exemple plus de 60 %, l'imputation invente l'essentiel de l'information. La supprimer est souvent plus sûr, sauf si l'absence elle-même est informative.