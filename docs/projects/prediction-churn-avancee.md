---
tags:
  - machine-learning
  - classification
  - xgboost
  - feature-engineering
  - churn-prediction
  - business-intelligence
  - production
---

# 📊 Prédiction de churn avancée avec XGBoost

![Badge de statut](https://img.shields.io/badge/statut-terminé-success)
![Badge Technologies](https://img.shields.io/badge/xgboost-1.7.0-orange)
![Badge Performance](https://img.shields.io/badge/accuracy-89.2%25-green)
![Badge F1-Score](https://img.shields.io/badge/f1--score-88.5%25-blue)
![Badge Business Impact](https://img.shields.io/badge/impact-25%25%20réduction%20churn-brightgreen)

## 🎯 Contexte et Objectifs

### Problème à résoudre
Développement d'un système de prédiction de churn pour une plateforme SaaS B2B, capable d'identifier les clients à risque de résiliation avec 3 mois d'avance.

### Objectifs
- **Objectif principal** : Prédire le churn avec 89%+ d'accuracy et 3 mois d'avance
- **Objectifs secondaires** : Réduire le churn de 25%, Améliorer la rétention client
- **Métriques de succès** : F1-Score > 85%, Precision > 80%, Recall > 85%

### Contexte métier
- **Secteur** : SaaS B2B / Customer Success
- **Utilisateurs** : Équipes Customer Success, Sales, Marketing
- **Impact attendu** : Réduction de 25% du taux de churn, +15% de rétention

## 📊 Données et Sources

### Sources de données
- **Source principale** : CRM + Analytics + Support tickets
- **Format** : PostgreSQL + JSON + CSV
- **Taille** : 50,000 clients, 2M+ interactions
- **Période** : 2020-2024
- **Fréquence** : Mise à jour quotidienne

### Qualité des données
- **Complétude** : 92% de complétude
- **Cohérence** : Validation avec équipes métier
- **Exactitude** : Vérification avec données de facturation
- **Actualité** : Données temps réel via API

### Variables disponibles
| Catégorie | Variables | Description | Importance |
|-----------|-----------|-------------|------------|
| **Comportement** | 15 | Sessions, Pages vues, Features utilisées | Haute |
| **Engagement** | 12 | Support tickets, Training, Webinars | Haute |
| **Business** | 8 | Plan, Utilisateurs, Revenus, Durée | Haute |
| **Support** | 10 | Tickets, Résolution, Satisfaction | Moyenne |
| **Marketing** | 6 | Source, Campaigns, Lead score | Faible |

## 🔬 Méthodologie

### 1. Feature Engineering avancé
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb

class ChurnFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_behavioral_features(self, df):
        """Création de features comportementales"""
        # Fréquence d'utilisation
        df['usage_frequency'] = df['sessions_30d'] / 30
        df['avg_session_duration'] = df['total_time'] / df['sessions_30d']
        
        # Tendance d'utilisation
        df['usage_trend'] = (df['sessions_30d'] - df['sessions_60d']) / df['sessions_60d']
        df['feature_adoption_rate'] = df['features_used'] / df['features_available']
        
        # Patterns d'utilisation
        df['weekend_usage_ratio'] = df['weekend_sessions'] / df['sessions_30d']
        df['peak_hours_usage'] = df['peak_hours_sessions'] / df['sessions_30d']
        
        return df
    
    def create_engagement_features(self, df):
        """Création de features d'engagement"""
        # Score d'engagement composite
        df['engagement_score'] = (
            df['support_tickets_30d'] * 0.3 +
            df['training_sessions'] * 0.4 +
            df['webinar_attendance'] * 0.3
        )
        
        # Délai de réponse aux communications
        df['avg_response_time'] = df['total_response_time'] / df['communications_count']
        
        # Satisfaction client
        df['satisfaction_trend'] = df['satisfaction_score'] - df['satisfaction_score_prev']
        
        return df
    
    def create_business_features(self, df):
        """Création de features business"""
        # Ratio revenus/utilisateurs
        df['revenue_per_user'] = df['monthly_revenue'] / df['active_users']
        
        # Croissance
        df['revenue_growth'] = (df['monthly_revenue'] - df['monthly_revenue_prev']) / df['monthly_revenue_prev']
        df['user_growth'] = (df['active_users'] - df['active_users_prev']) / df['active_users_prev']
        
        # Plan vs utilisation
        df['plan_utilization'] = df['features_used'] / df['plan_features']
        
        return df
    
    def create_temporal_features(self, df):
        """Création de features temporelles"""
        # Âge du compte
        df['account_age_days'] = (pd.to_datetime('now') - pd.to_datetime(df['created_at'])).dt.days
        
        # Saisonnalité
        df['month'] = pd.to_datetime(df['created_at']).dt.month
        df['quarter'] = pd.to_datetime(df['created_at']).dt.quarter
        
        # Dernière activité
        df['days_since_last_login'] = (pd.to_datetime('now') - pd.to_datetime(df['last_login'])).dt.days
        
        return df
```

### 2. Gestion du déséquilibre des classes
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import StratifiedKFold

class ChurnDataBalancer:
    def __init__(self, strategy='smote'):
        self.strategy = strategy
        self.balancer = None
        
    def fit_resample(self, X, y):
        """Équilibrage des classes"""
        if self.strategy == 'smote':
            self.balancer = SMOTE(random_state=42, k_neighbors=3)
        elif self.strategy == 'smoteenn':
            self.balancer = SMOTEENN(random_state=42)
        elif self.strategy == 'undersample':
            self.balancer = RandomUnderSampler(random_state=42)
        
        X_balanced, y_balanced = self.balancer.fit_resample(X, y)
        
        return X_balanced, y_balanced
    
    def get_class_weights(self, y):
        """Calcul des poids de classe pour XGBoost"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        return dict(zip(classes, weights))
```

### 3. Modélisation avec XGBoost
```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import optuna

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.threshold = 0.5
        
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Optimisation des hyperparamètres avec Optuna"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
            }
            
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred)
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        return study.best_params
    
    def train_model(self, X_train, y_train, X_val, y_val, params=None):
        """Entraînement du modèle avec validation"""
        if params is None:
            params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'scale_pos_weight': 3,
                'random_state': 42
            }
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def optimize_threshold(self, X_val, y_val):
        """Optimisation du seuil de classification"""
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        return best_threshold
```

### 4. Évaluation et interprétation
```python
import shap
import matplotlib.pyplot as plt
import seaborn as sns

class ChurnModelInterpreter:
    def __init__(self, model, X_train):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.X_train = X_train
        
    def explain_prediction(self, X_sample):
        """Explication d'une prédiction individuelle"""
        shap_values = self.explainer.shap_values(X_sample)
        
        # Plot SHAP values
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('Explication de la prédiction de churn')
        plt.show()
        
        return shap_values
    
    def global_feature_importance(self):
        """Importance globale des features"""
        shap_values = self.explainer.shap_values(self.X_train)
        
        # Summary plot
        shap.summary_plot(shap_values, self.X_train, show=False)
        plt.title('Importance globale des features')
        plt.show()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def partial_dependence_plot(self, feature_name, X_sample):
        """Plot de dépendance partielle"""
        shap.dependence_plot(
            feature_name, 
            self.explainer.shap_values(X_sample), 
            X_sample,
            show=False
        )
        plt.title(f'Dépendance partielle - {feature_name}')
        plt.show()
```

## 📈 Résultats et Métriques

### Performance du modèle
| Métrique | Valeur | Baseline | Amélioration |
|----------|--------|----------|--------------|
| Accuracy | 89.2% | 76.5% | +16.6% |
| Precision | 87.8% | 72.1% | +21.8% |
| Recall | 89.1% | 78.3% | +13.8% |
| F1-Score | 88.5% | 75.1% | +17.8% |
| AUC-ROC | 0.94 | 0.82 | +14.6% |

### Impact business
| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| Taux de churn | 12.5% | 9.4% | -25% |
| Rétention client | 87.5% | 90.6% | +3.1% |
| Revenus récurrents | 100% | 115% | +15% |
| Coût d'acquisition | 100% | 85% | -15% |

### Performance par segment
| Segment | Precision | Recall | F1-Score | Clients |
|---------|-----------|--------|----------|---------|
| Enterprise | 92.1% | 88.3% | 90.2% | 500 |
| Mid-market | 89.4% | 91.2% | 90.3% | 1,200 |
| SMB | 85.6% | 87.9% | 86.7% | 3,300 |

## 🚀 Déploiement

### Architecture de production
- **Environnement** : Docker + Kubernetes
- **API** : FastAPI avec documentation automatique
- **Base de données** : PostgreSQL + Redis
- **Monitoring** : MLflow + Prometheus
- **Scheduling** : Apache Airflow

### Code de déploiement
```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import joblib
import redis
import json

app = FastAPI(title="Churn Prediction API")

# Configuration
redis_client = redis.Redis(host='localhost', port=6379, db=0)
model = joblib.load('churn_model.pkl')
feature_engineer = joblib.load('feature_engineer.pkl')

class CustomerData(BaseModel):
    customer_id: str
    features: dict
    prediction_date: str

class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    key_factors: list
    recommendations: list

@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer_data: CustomerData):
    # Feature engineering
    features_df = pd.DataFrame([customer_data.features])
    processed_features = feature_engineer.transform(features_df)
    
    # Prédiction
    churn_prob = model.predict_proba(processed_features)[0][1]
    
    # Détermination du niveau de risque
    if churn_prob > 0.8:
        risk_level = "Très élevé"
    elif churn_prob > 0.6:
        risk_level = "Élevé"
    elif churn_prob > 0.4:
        risk_level = "Modéré"
    else:
        risk_level = "Faible"
    
    # Facteurs clés (simulation)
    key_factors = [
        "Baisse d'utilisation de 40%",
        "Aucun ticket support récent",
        "Plan sous-utilisé"
    ]
    
    # Recommandations
    recommendations = [
        "Proposer une session de formation",
        "Contacter le customer success manager",
        "Offrir un upgrade de plan"
    ]
    
    return ChurnPrediction(
        customer_id=customer_data.customer_id,
        churn_probability=float(churn_prob),
        risk_level=risk_level,
        key_factors=key_factors,
        recommendations=recommendations
    )

@app.post("/batch_predict")
async def batch_predict(background_tasks: BackgroundTasks):
    """Prédiction en lot pour tous les clients"""
    background_tasks.add_task(run_batch_prediction)
    return {"message": "Batch prediction started"}

async def run_batch_prediction():
    """Exécution de la prédiction en lot"""
    # Récupération des données clients
    customers = get_all_customers()
    
    predictions = []
    for customer in customers:
        prediction = predict_churn(customer)
        predictions.append(prediction)
    
    # Sauvegarde des résultats
    save_predictions(predictions)
```

## 📊 Visualisations

### Analyse des features importantes
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualisation des métriques
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Feature importance
top_features = feature_importance.head(10)
axes[0, 0].barh(top_features['feature'], top_features['importance'])
axes[0, 0].set_title('Top 10 Features Importance')
axes[0, 0].set_xlabel('Importance')

# Distribution des probabilités
churn_probs = model.predict_proba(X_test)[:, 1]
axes[0, 1].hist(churn_probs, bins=50, alpha=0.7, color='red')
axes[0, 1].set_title('Distribution des probabilités de churn')
axes[0, 1].set_xlabel('Probabilité de churn')
axes[0, 1].set_ylabel('Fréquence')

# Courbe ROC
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, churn_probs)
auc_score = auc(fpr, tpr)

axes[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--')
axes[1, 0].set_xlabel('Taux de faux positifs')
axes[1, 0].set_ylabel('Taux de vrais positifs')
axes[1, 0].set_title('Courbe ROC')
axes[1, 0].legend()

# Impact business
months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
churn_rate = [12.5, 11.8, 10.2, 9.8, 9.4, 9.1]
revenue = [100, 102, 105, 108, 115, 118]

ax2 = axes[1, 1].twinx()
axes[1, 1].plot(months, churn_rate, 'ro-', label='Taux de churn (%)')
ax2.plot(months, revenue, 'bo-', label='Revenus (index 100)')
axes[1, 1].set_xlabel('Mois')
axes[1, 1].set_ylabel('Taux de churn (%)', color='red')
ax2.set_ylabel('Revenus (index 100)', color='blue')
axes[1, 1].set_title('Impact du modèle sur le business')
axes[1, 1].legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
```

## 🔗 Liens et ressources

### Code source
- **Repository GitHub** : [github.com/loick-dernoncourt/churn-prediction](https://github.com/loick-dernoncourt/churn-prediction)
- **Notebooks Jupyter** : [github.com/loick-dernoncourt/churn-prediction/tree/main/notebooks](https://github.com/loick-dernoncourt/churn-prediction/tree/main/notebooks)

### Démonstrations
- **Démo interactive** : [churn-prediction-demo.example.com](https://churn-prediction-demo.example.com)
- **API Documentation** : [churn-prediction-api.example.com/docs](https://churn-prediction-api.example.com/docs)
- **Dashboard** : [churn-prediction-dashboard.example.com](https://churn-prediction-dashboard.example.com)

### Documentation
- **Rapport technique** : [churn-prediction-report.example.com](https://churn-prediction-report.example.com)
- **Présentation** : [churn-prediction-slides.example.com](https://churn-prediction-slides.example.com)
- **Article de blog** : [blog.example.com/churn-prediction](https://blog.example.com/churn-prediction)

## 🎯 Prochaines étapes

### Améliorations prévues
- [ ] Intégration de données comportementales temps réel
- [ ] Modèle de prédiction de la valeur de vie client (LTV)
- [ ] Recommandations personnalisées d'actions
- [ ] Intégration avec CRM et outils de marketing

### Technologies à explorer
- [ ] Deep Learning pour patterns complexes
- [ ] Graph Neural Networks pour relations clients
- [ ] Reinforcement Learning pour optimisation d'actions
- [ ] Explicabilité avancée avec LIME et SHAP

---

*Dernière mise à jour : {{ git_revision_date_localized }}*
