- Vous trouverez un dossier **_booking-cancel-prediction_with_EWA_** où il y a le code pour le problème de classification. 
Le **_EWA_** est ajouté pour distinguer ce dossier d'un autre dossier de classification.
Le rapport sur la classification **_Rapport_data_challenge_EWA.pdf_** contient ce qui a été fait sur le problème de classification
avec une agrégation d'experts avec l'algorthme EWA.

- - Vous trouverez un dossier **Classification_SVM** où il y a le code pour le problème de classification.
  - Le fichier qui permet d’obtenir les soumissions est le fichier **simple_predict**. Les différents modèles sont dans le dossier **SVM_models**
  - Pour choisir un modèle, il faut décommenter l’impmort souhaité dans **simple_predict** et commenter les autres
  - Pour importer les données, il faut modifier la variable **DATA_DIR** ligne 23 du fichier **simple_predict**.

- Vous trouverez un dossier **Data_challenge_de_régression_Notebook_final.py** où il y a le code pour le problème de régression.
Les données ont été chargées à l'aide de la fonction files() de google.colab. Mais elles peuvent etre chargées à l'aide d'une autre bibliothèque si nécessaire.
L'important est de charger les 4 fichiers train_data.csv, test_data.csv, data.csv et dataset.csv au début du script.
Les fichiers data.csv et dataset.csv ont été récupérés sur la plateforme Kaggle, les liens pour les récupérés sont détaillées dans le rapport.
J'ai ajouté le fichier dataset.csv sur le repo mais le fichier data.csv est trop volumineux, il faut donc aller le chercher sur la plateforme Kaggle.
Le script est éxécutable sans ce fichier, la précision final baissera légèrement.
Pour finir la soumission est enregistrer dans un dossier submissions via le chemin d'accès "submissions/submission_moe_clusters_fast.csv".
En annexe sont ajoutés tout les modèles testées et hyperparamètres retenues aux cours du challenge.
La partie du rapport sur la régression contient ce qui a été fait sur le problème de régression.
