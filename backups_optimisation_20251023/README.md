# Portfolio Data Science - Loïck Dernoncourt

Ce dépôt contient mon site portfolio de data science, construit avec [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## 🚀 Site déployé

Le site est accessible à l'adresse: [https://loick-dernoncourt.github.io/portfolio-site/](https://loick-dernoncourt.github.io/portfolio-site/)

## 💻 Développement local

### Prérequis

- Python 3.11+
- pip

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/loick-dernoncourt/portfolio-site.git
cd portfolio-site

# Installer les dépendances
pip install mkdocs-material
```

### Lancer le serveur de développement

```bash
mkdocs serve
```

Naviguez vers `http://localhost:8000` pour voir le site en développement.

### Construire le site

```bash
mkdocs build
```

Les fichiers HTML générés seront dans le répertoire `site/`.

## 📝 Ajouter un projet

Pour ajouter un nouveau projet au portfolio:

1. Créez un fichier Markdown dans le dossier `docs/projects/`
2. Ajoutez les métadonnées en haut du fichier:

```markdown
---
tags:
  - machine-learning
  - data-analysis
  - python
  - visualisation
  - nlp
  - computer-vision
  - deep-learning
  - pytorch
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - jupyter
  - git
  - docker
  - aws
  - sql
  - tableau
  - power-bi
---

# Titre du projet

![Badge de statut](https://img.shields.io/badge/statut-terminé-success)
![Badge Technologies](https://img.shields.io/badge/python-3.11-blue)
![Badge Framework](https://img.shields.io/badge/pytorch-2.0-orange)
![Badge Performance](https://img.shields.io/badge/accuracy-95%25-green)

## 🎯 Contexte et Objectifs

Description claire du problème à résoudre et des objectifs métier...

## 📊 Données et Sources

- **Source** : [Nom de la source]
- **Format** : CSV/JSON/Parquet
- **Taille** : X millions d'enregistrements
- **Période** : 2020-2024
- **Qualité** : 95% de complétude

## 🔬 Méthodologie

### Préprocessing
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Exemple de code de preprocessing
def preprocess_data(df):
    # Nettoyage des données
    df = df.dropna()
    # Normalisation
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled
```

### Modélisation
```python
import torch
import torch.nn as nn

class MonModele(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MonModele, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## 📈 Résultats et Métriques

| Métrique | Valeur | Baseline |
|----------|--------|----------|
| Accuracy | 95.2% | 78.5% |
| Precision | 94.8% | 76.2% |
| Recall | 95.1% | 77.8% |
| F1-Score | 94.9% | 77.0% |

## 🚀 Déploiement

- **Environnement** : Docker + AWS
- **API** : FastAPI
- **Monitoring** : MLflow
- **CI/CD** : GitHub Actions

## 🔗 Liens

- [Code source](https://github.com/loick-dernoncourt/projet-exemple)
- [Démo interactive](https://demo.example.com)
- [Rapport technique](https://rapport.example.com)
- [Présentation](https://slides.example.com)
```

3. Ajoutez votre projet dans la navigation en modifiant `mkdocs.yml`:

```yaml
nav:
  - Accueil: index.md
  - Projets:
      - Aperçu: projects/index.md
      - ... autres projets ...
      - Votre Nouveau Projet: projects/votre_projet.md
```

4. Assurez-vous d'inclure dans votre projet:
   - Contexte et objectifs
   - Données utilisées (sources, format)
   - Méthodologie (avec extraits de code)
   - Résultats et métriques
   - Liens vers le code source et démos

## 🏗️ Structure du projet

```
portfolio-site/
├── docs/
│   ├── index.md
│   ├── about.md
│   ├── projects/
│   │   ├── index.md
│   │   ├── template.md
│   │   └── exemples/
│   ├── skills/
│   │   ├── index.md
│   │   ├── machine-learning.md
│   │   ├── deep-learning.md
│   │   └── data-engineering.md
│   └── contact.md
├── mkdocs.yml
├── requirements.txt
└── README.md
```

## 🎨 Personnalisation

### Thème et couleurs
Le site utilise le thème Material avec une palette de couleurs personnalisée pour la data science.

### Plugins
- **mkdocs-material** : Thème principal
- **mkdocs-git-revision-date-localized-plugin** : Dates de modification
- **mkdocs-minify-plugin** : Optimisation des assets

## 📊 Analytics

Le site intègre Google Analytics pour le suivi des performances et l'analyse du trafic.

## 🤝 Contribution

Pour contribuer à ce portfolio :
1. Fork le repository
2. Créez une branche feature
3. Committez vos changements
4. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
