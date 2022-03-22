# <div align="center"> Projet IA sur la classification de musique<div>

Il s'agit d'un projet étudiant visant à créer un modèle d'IA capable de reconnaitre le genre d'une musique à partir du spectrogramme Mel issu d'un extrait de cette musique.

## Participants
- BOISSAY Eve 
- DIOMANDE Lansana 
- LE NET Laurine
- TOUERESS Sophia

## Dataset + generation des spectrogrammes mel

Notre projet utilise le dataset disponible <a href="http://marsyas.info/downloads/datasets.html">ici</a>.
Il contient 1000 musiques réparties équitablement sur 10 genres de musiques. 
En utilisant les fichier **generation_dataset_method_***, ​vous pouvez découper les musiques en 10 morceaux de 3 secondes et utiliser les extraits afin d'obtenir des spectrogrammes mel. Ainsi vous obtiendrez un dataset de 10000 spectrogrammes répartis équitablement sur 10 genres musicaux.

- **generation_dataset_method_1** : Permet de générer les spectrogrammes mel en utilisant la méthode 1 d'entrainement
- **generation_dataset_method_2** :  Permet de générer les spectrogrammes mel en utilisant la méthode 2 d'entrainement


## Entrainement des modèles

Dans le dossier training, vous trouverez les différents scripts qui nous ont permis de générer nos divers modèles.
Vous pouvez les importer dans les fichiers train_model_method_1  et train_model_method_2 et utiliser la méthode `getModel` pour obtenir le modèle à entrainer.

**Utilisation de réseaux convolutif 2D**
Il y a deux fichiers disponibles pour l'entrainement du modèle. 
- train_model_method_1 : Celui-ci utilise ImageDataGenerator pour donner les images au modèle afin de permettre l'entrainement et le test. L'avantage est que celui-ci n'est pas très consommateur en ressource. Le désavantage est que celui-ci prend plus de temps.
- train_model_method_2 : Celui-ci utilise image_dataset_from_directory pour récupérer les images qui seront ensuite transmises au modèle afin de permettre son entrainement et le test. Il est plus rapide mais aussi plus consommateur en ressource. Parfois, celui-ci ne permettra pas d'entrainer un modèle si la taille des images est trop grande.

## Modèles entrainés

Vous trouverez tous les modèles que nous avons entrainés ans le dossier téléchargeable <a href="https://epfedu-my.sharepoint.com/:f:/g/personal/sophia_toueress_epfedu_fr/ElRh2dUkGFxCqNUjD4BI4ScB2_50YIQnO45ahe8ZMForvg?e=55gSZ4"> ici </a> 

Ceux ci sont nommés ainsi **model_pourcentage_imageShape_***
Exemple:
- model_60_150x200x3_n3.h5 => accuracy: 60% , Image: 150x200 en rgb
- model_78_288x432x4_n2.h5 => accuracy: 78% , Image: 288x432 en rgba

## Application
Le fichier `./app/app.py` permet de lancer une application développée avec le framework streamlit. Cette application permet à l'utilisateur de glisser un morceaux en mp3 et lui retourne les 2 genres musicaux qui ont le plus de chance de correspondre à celui du morceau. Une version est en ligne au lien <a href="http://music-classif.partage2passion.fr/">suivant</a> (lien indisponible. En effet il la période d'essai avec Azure est fini 🙂)

### <div align="center"> Lancement avec Docker <div>

**Prérequis:**
- Docker 
- docker-compose

**Configuration du fichier `app.py` :**
Il suffit d'ouvrir le fichier dans un éditeur de texte et d'adapter la section `Configuration` à votre modèle.

**Lancement:** Se placer dans le dossier `./app` et exécuter la commande `docker-compose up --build`


**Lien pour l'application:** `http://localhost:8501/`

### <div align="center"> Lancement sans Docker <div>

**Prérequis:**
- python 3 + pip
- ffmpeg

**Installation des packages python:**
 Se placer dans le dossier `./app` et exécuter la commande `pip install -r requirement.txt`

**Configuration du fichier app.py :**
Il suffit d'ouvrir le fichier dans un éditeur de texte et d'adapter la section `Configuration` à votre modèle

**Lancement:** Se placer dans le dossier `./app` et exécuter la commande `streamlit run app.py`


**Lien pour l'application:** `http://localhost:8501/`


## Remerciements

Merci à **Léonard BENEDETTI** pour toute l'aide apportée.
