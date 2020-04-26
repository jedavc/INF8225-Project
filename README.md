# INF8225-Project
## Exploration de diverses méthodes de segmentation d'images médicales.
### Plusieurs ensembles de données sont utilisés soient BraTS-18, ISIC-2017, DRIVE et CHAOS.
### Les modèles explorés sont un encodeur automatique régularisé, un SegAN, une dérivée de U-Net (BCDU-Net) et un modèle multi-scale self-guided. Chacun de ces modèles a un fichier main personnalisé.
### Pour BCDU-Net:
Implémentation et pré-traitement inspiré de https://github.com/rezazad68/BCDU-Net. Voici les répertoires qu'il faut créer pour rouler le code. 
Il faut créer les répertoires "BCDU_models", "Preprocessed_Images" et "Tests" dans le répertoire "BCDU-net" pour avoir l'ordre suivant:
architectures/BCDU_net/Preprocessed_Images/... Aussi il faut ajouter le répertoire "Patches" dans "Preprocessed_Images".

Pour démarrer l'entraînement, allez dans le fichier "INF8225-Project" et ouvrez une console à partir de cette destination. Tapez la commande "python mains\train_bcdunet --train --eval" pour démarrer l'entraînement et effectuer l'évaluation du modèle par la suite. 


