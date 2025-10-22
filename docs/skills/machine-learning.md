# 🤖 Machine Learning

Expertise en machine learning avec 5+ années d'expérience et 20+ projets réalisés.

## 🎯 Compétences principales

### Algorithmes supervisés
- **Classification** : Random Forest, SVM, XGBoost, LightGBM
- **Régression** : Linear Regression, Ridge, Lasso, Elastic Net
- **Ensemble Methods** : Bagging, Boosting, Stacking
- **Feature Selection** : Recursive Feature Elimination, L1 Regularization

### Algorithmes non-supervisés
- **Clustering** : K-Means, DBSCAN, Hierarchical Clustering
- **Dimensionality Reduction** : PCA, t-SNE, UMAP
- **Anomaly Detection** : Isolation Forest, One-Class SVM
- **Association Rules** : Apriori, FP-Growth

### Optimisation et validation
- **Hyperparameter Tuning** : Grid Search, Random Search, Bayesian Optimization
- **Cross-Validation** : K-Fold, Stratified, Time Series Split
- **Model Selection** : AIC, BIC, Cross-Validation
- **Feature Engineering** : Polynomial Features, Interaction Terms

## 🛠️ Stack technique

### Frameworks principaux
```python
# Scikit-learn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# LightGBM
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
```

### Outils de développement
- **Jupyter Notebooks** : Exploration et prototypage
- **MLflow** : Gestion des expériences
- **Optuna** : Optimisation des hyperparamètres
- **SHAP** : Explicabilité des modèles

## 📊 Projets réalisés

### Classification d'images médicales
**Technologies** : Random Forest, Feature Engineering  
**Résultat** : 95.2% d'accuracy sur 10K images  
**Impact** : Réduction de 40% du temps de diagnostic

### Prédiction de prix immobiliers
**Technologies** : XGBoost, Feature Engineering  
**Résultat** : RMSE 0.15, R² 0.87  
**Impact** : Amélioration de 30% de la précision

### Segmentation de clients
**Technologies** : K-Means, PCA, Feature Engineering  
**Résultat** : 4 segments identifiés avec 85% de cohérence  
**Impact** : +25% de conversion marketing

## 🔬 Méthodologie

### 1. Analyse exploratoire
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement et exploration
df = pd.read_csv('data.csv')
print(df.info())
print(df.describe())

# Visualisation des distributions
plt.figure(figsize=(12, 8))
df.hist(bins=50, figsize=(12, 8))
plt.show()

# Matrice de corrélation
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```

### 2. Préprocessing
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Nettoyage des données
df = df.dropna()
df = df.drop_duplicates()

# Encodage des variables catégorielles
le = LabelEncoder()
df['categorical_var'] = le.fit_transform(df['categorical_var'])

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 3. Feature Engineering
```python
# Création de nouvelles features
df['feature_ratio'] = df['var1'] / df['var2']
df['feature_interaction'] = df['var1'] * df['var2']
df['feature_polynomial'] = df['var1'] ** 2

# Sélection des features
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Importance des features
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importance = rf.feature_importances_
```

### 4. Modélisation
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Modèles à comparer
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Entraînement et évaluation
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.3f}")
```

### 5. Optimisation des hyperparamètres
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Grille de paramètres pour XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Recherche par grille
grid_search = GridSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Meilleurs paramètres:", grid_search.best_params_)
print("Meilleur score:", grid_search.best_score_)
```

### 6. Évaluation et validation
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

# Métriques de performance
y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion')
plt.show()

# Validation croisée
cv_scores = cross_val_score(
    grid_search.best_estimator_, X, y, cv=5, scoring='f1_weighted'
)
print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## 📈 Métriques de performance

### Classification
| Métrique | Valeur | Description |
|----------|--------|-------------|
| Accuracy | 94.2% | Précision globale |
| Precision | 93.8% | Précision par classe |
| Recall | 94.1% | Rappel par classe |
| F1-Score | 93.9% | Score F1 harmonique |
| AUC-ROC | 0.96 | Aire sous la courbe ROC |

### Régression
| Métrique | Valeur | Description |
|----------|--------|-------------|
| RMSE | 0.15 | Racine de l'erreur quadratique |
| MAE | 0.12 | Erreur absolue moyenne |
| R² | 0.87 | Coefficient de détermination |
| MAPE | 15.2% | Erreur absolue en pourcentage |

## 🎯 Bonnes pratiques

### Préprocessing
- **Gestion des valeurs manquantes** : Imputation intelligente
- **Normalisation** : StandardScaler pour la plupart des algorithmes
- **Encodage** : LabelEncoder pour les variables ordinales
- **Feature Engineering** : Création de features métier

### Modélisation
- **Validation croisée** : Toujours utiliser CV pour l'évaluation
- **Hyperparameter tuning** : Optimisation systématique
- **Ensemble methods** : Combinaison de modèles
- **Feature selection** : Réduction de la dimensionnalité

### Évaluation
- **Métriques appropriées** : Choix selon le problème
- **Validation temporelle** : Pour les données temporelles
- **Test A/B** : Validation en conditions réelles
- **Monitoring** : Suivi des performances en production

## 🚀 Déploiement

### MLOps
```python
import mlflow
import mlflow.sklearn
import joblib

# Sauvegarde du modèle
mlflow.sklearn.log_model(
    grid_search.best_estimator_, 
    "model",
    registered_model_name="best_model"
)

# Sauvegarde des préprocesseurs
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
```

### API de prédiction
```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Chargement du modèle
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.post("/predict")
async def predict(data: dict):
    # Préprocessing
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    
    # Prédiction
    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)
    
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0].max())
    }
```

## 📚 Ressources d'apprentissage

### Cours recommandés
- **Coursera** : Machine Learning (Stanford)
- **Udacity** : Machine Learning Engineer Nanodegree
- **Fast.ai** : Practical Machine Learning for Coders

### Livres essentiels
- **Hands-On Machine Learning** (Aurélien Géron)
- **The Elements of Statistical Learning** (Hastie, Tibshirani, Friedman)
- **Pattern Recognition and Machine Learning** (Christopher Bishop)

### Pratique
- **Kaggle** : Compétitions et datasets
- **Scikit-learn** : Documentation officielle
- **Papers with Code** : Recherche et implémentations

---

*Dernière mise à jour : {{ git_revision_date_localized }}*
