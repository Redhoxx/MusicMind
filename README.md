# AI Project 2024

**Bruce L'horset, Lucas Jendzio-Verdasca, Guillaume Cappe de Baillon, Thibaud Barberon**

Ce projet vise à reconnaître le genre musical d'un fichier audio. Pour ce faire, nous créons une IA qui analyse le signal audio d'une chanson pour en extraire des indicateurs pertinents.

## Fonctionnement

L'IA utilise la bibliothèque `librosa` et des fonctions personnalisées pour calculer les caractéristiques suivantes :

*   MFCCs (Mel Frequency Cepstral Coefficients)
*   Caractéristiques chromatiques
*   Tempo (estimation des battements)
*   Taux de passage à zéro (Zero Crossing Rate - ZCR)

Chaque caractéristique joue un rôle spécifique dans la description du contenu audio :

*   Les MFCCs capturent les caractéristiques spectrales du signal audio sur l'échelle de Mel, qui se rapproche de la perception humaine de la hauteur et de la fréquence.
*   Les caractéristiques chromatiques représentent les 12 demi-tons (gamme chromatique) de la musique et capturent le contenu harmonique et tonal du signal audio.
*   L'extraction du tempo identifie les battements par minute (BPM) de l'audio, ce qui est crucial pour l'analyse rythmique.
*   Le ZCR mesure la vitesse à laquelle le signal audio traverse l'axe d'amplitude zéro, ce qui permet de distinguer les sons voisés et non voisés.

Ces caractéristiques sont les entrées de notre IA pour chaque chanson. Grâce aux données d'apprentissage, le modèle est entraîné à reconnaître les valeurs de ces variables et à les associer à un genre musical.

## Apprentissage automatique

Il s'agit d'un projet d'apprentissage supervisé : nous fournissons à l'IA une étiquette (genre musical) pour chaque ensemble de variables en entrée. Le modèle a donc 4 entrées (MFCCs, caractéristiques chromatiques, tempo, ZCR) et 1 sortie (genre musical).

## Architecture du modèle

Le modèle utilise un réseau de neurones LSTM (Long Short-Term Memory) avec des couches Dense et Dropout pour la classification.

## Entraînement et réentraînement

Le modèle est entraîné sur un ensemble de données de musiques étiquetées. Un workflow GitHub automatise l'entraînement et le réentraînement du modèle. Le réentraînement peut être déclenché manuellement pour intégrer de nouvelles musiques à l'ensemble de données.

## Interface utilisateur

Une interface utilisateur Gradio permet de télécharger un fichier audio et d'obtenir une prédiction du genre musical en temps réel.

## Bibliothèques utilisées

*   `librosa` : pour l'analyse audio et l'extraction de caractéristiques.
*   `tensorflow` : pour la construction et l'entraînement du modèle de réseau de neurones.
*   `sklearn` : pour le prétraitement des données (StandardScaler, LabelEncoder).
*   `gradio` : pour l'interface utilisateur.
*   `pydub` : pour la gestion des fichiers audio.
*   `soundfile` : pour la lecture et l'écriture de fichiers audio.
*   `pandas` : pour la manipulation des données.

## Utilisation

**Installation des dépendances:**

```bash
pip install -r requirements.txt
```

**Entraînement du modèle (entraîne de 0 le modèle):**
```bash
python train_model.py
```

**Réentraînement du modèle (reprend le modèle enregistré):**
```bash
python train_model.py --retrain
```

**Lancement de l'interface utilisateur:**
```bash
python main.py
```

## Utilisation comme base pour un autre projet:
Ce projet peut servir de base pour d'autres projets de classification audio. Vous pouvez :

Modifier les caractéristiques audio extraites.
Utiliser un autre type de modèle d'apprentissage automatique.
Adapter l'interface utilisateur à vos besoins.
Intégrer le code dans une application plus complexe.

## Remarques:
Assurez-vous que le fichier Excel "MusicMind - musics used for training.xlsx" est présent dans le dossier "data/raw" et qu'il contient une colonne nommée "Nom du fichier".
Le workflow GitHub est configuré pour se déclencher manuellement. Vous pouvez le modifier pour qu'il se déclenche automatiquement sur les push vers la branche principale.

## Améliorations futures
Augmenter la taille et la diversité de l'ensemble de données d'entraînement.
Explorer d'autres architectures de modèles et d'autres caractéristiques audio.
Améliorer l'interface utilisateur pour la rendre plus interactive et informative.
Déployer le modèle comme une application web ou une API.
