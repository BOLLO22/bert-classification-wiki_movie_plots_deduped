# BERT Classification - tweet vs type

Devoir Pratique n°3 - NLP avec PyTorch : Fine-tuning de BERT pour la classification de texte.

**Binôme :** Jürgen BOLLO KONDABEKA  & Cheikh MBALLO

## 1. Présentation du dataset

- **Source** : (https://drive.google.com/file/d/1hMKWd8BAymWMo1YmhHeXYbEaKEL-YrGF/view) (`Train.csv`)
- **Tâche** : classification du texte de la colonne `tweet`  selon la colonne `type` .
- **Nombre total d'exemples** : _à compléter_
- **Nombre de classes** : _à compléter_ (liste des valeurs uniques de `type`)
- **Distribution des classes** : _largement par rapport aux autres origines)
  - **Stratégie adoptée** : _à compléter_ (ex. regroupement des classes minoritaires en "Other", sous-échantillonnage des classes majoritaires, ou poids de classes dans la loss)
- **Longueur des textes (tokens)** : max = _..._,
  - Justification du `max_length` choisi : _à compléter_


## 2. Modèle et choix techniques

- **Modèle pré-entraîné** : `bert-base-uncased`
- **Tokenizer** : `AutoTokenizer` / `BertTokenizer` de Hugging Face (`bert-base-uncased`)
- **Tête de classification** : couche `Dropout` + `Linear(hidden_size, num_labels)` appliquée sur le `pooler_output` (représentation du token `[CLS]`)
- **max_length** : 128 (ou 256, selon la statistique de longueur des `tweet`)
- **Hyperparamètres**
  - Learning rate : 2e-5 à 5e-5
  - Batch size : 16 ou 32
  - Epochs : 3 à 5
  - Optimiseur : `Adam` (weight_decay=0.01)
  - Scheduler : linéaire avec warmup (optionnel)
  - Loss : `CrossEntropyLoss`
  - Seed : 42 (fixée pour `random`, `numpy`, `torch`)
- **Split** : train/validation stratifié 80/20 (`load_and_split_data` dans `dataset.py`)

## 3. Étapes de réalisation et difficultés rencontrées

- _à compléter_ : étapes suivies (exploration des données, préparation du Dataset, écriture de la boucle d'entraînement, sauvegarde du meilleur modèle, création de la démo Gradio)
- _à compléter_ : difficultés rencontrées (déséquilibre des classes, temps d'entraînement, gestion de la mémoire GPU, etc.) et solutions apportées

## 4. Résultats

- **Courbes loss/accuracy** : _insérer captures d'écran (ex. `results/loss_curve.png`, `results/accuracy_curve.png`)_
- **Métriques finales** (sur le jeu de validation) :
  - Accuracy : _à compléter_
  - F1-score (macro/weighted) : _à compléter_
- **Matrice de confusion** : _insérer capture d'écran (ex. `results/confusion_matrix.png`)_

## 5. Démo Gradio

- **Captures d'écran de la démo** : _insérer captures (ex. `results/demo_screenshot.png`)_
- Description de l'interface : saisie d'un texte (tweet), affichage de la classe d'origine prédite et des probabilités par classe, avec exemples pré-remplis.

## 6. Installation et exécution

### Prérequis

```bash
pip install -r requirements.txt
```

### Préparer les données

Placer le fichier `Train.csv` dans le dossier `data/` :

```
data/Train.csv
```

### Entraînement

```bash
python train.py
```

Le meilleur modèle (selon `val_loss`) est sauvegardé dans le dossier de checkpoints (ex. `checkpoints/best_model.pt`).

### Lancer la démo

```bash
python demo.py
```

L'interface Gradio est accessible sur `http://127.0.0.1:7860` (lien affiché dans le terminal).

## 7. Structure du projet

```
bert-classification-wiki-movie-tweets/
├── data/                  # dataset (non versionné, voir .gitignore)
│   └── Train.csv
├── dataset.py             # tweetOriginEthnicityDataset + load_and_split_data
├── model.py               # BerttweetOriginEthnicityClassifier + create_model
├── train.py               # boucles train_epoch / eval_epoch + main
├── demo.py                # interface Gradio
├── utils.py                # métriques, seed, visualisations
├── requirements.txt
└── README.md
```

## 8. Répartition du travail

| Membre                 | Tâches réalisées                                           |
|------------------------|------------------------------------------------------------|
| Jûrgen BOLLO KONDABEKA |  exploration des données, dataset.py, model.py, README.md  |
| Cheikh MBALLO          |  train.py, demo.py, utils.py, requirements.txt             |

## 9. Versionnement Git

Le projet est versionné fichier par fichier avec un historique de commits progressif (pas un seul commit final). Exemple de séquence de commits recommandée :

```bash
git init
git add .gitignore
git commit -m "chore: ajoute .gitignore"

git add requirements.txt
git commit -m "chore: ajoute requirements.txt"

git add dataset.py
git commit -m "feat: ajoute tweetOriginEthnicityDataset et load_and_split_data"

git add model.py
git commit -m "feat: ajoute BerttweetOriginEthnicityClassifier"

git add utils.py
git commit -m "feat: ajoute fonctions utilitaires (seed, métriques, visualisations)"

git add train.py
git commit -m "feat: implémente la boucle d'entraînement et d'évaluation PyTorch"

git add demo.py
git commit -m "feat: ajoute l'interface Gradio de démo"

git add README.md
git commit -m "docs: complète le README avec le rapport détaillé"

git remote add origin <https://github.com/BOLLO22/bert-classification-wiki_movie_tweets_deduped.git>
git push -u origin main
```

Chaque membre du binôme doit committer ses propres contributions afin que les deux historiques de commits soient visibles sur GitHub.
