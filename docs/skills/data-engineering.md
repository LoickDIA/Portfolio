# 🗄️ Data Engineering

Expertise en data engineering avec 3+ années d'expérience et 8+ projets réalisés.

## 🎯 Compétences principales

### Technologies de données
- **Big Data** : Apache Spark, Hadoop, Hive
- **Databases** : PostgreSQL, MongoDB, Redis, Elasticsearch
- **Cloud** : AWS (S3, EC2, ECS, Lambda), Google Cloud, Azure
- **Streaming** : Kafka, Apache Flink, Apache Storm
- **Orchestration** : Airflow, Prefect, Dagster

### Outils de développement
- **Conteneurisation** : Docker, Kubernetes
- **CI/CD** : GitHub Actions, GitLab CI, Jenkins
- **Monitoring** : Prometheus, Grafana, ELK Stack
- **Versioning** : Git, DVC (Data Version Control)

### Langages et frameworks
- **Python** : Pandas, NumPy, PySpark, FastAPI
- **SQL** : PostgreSQL, MySQL, BigQuery
- **Scala** : Apache Spark
- **Bash** : Scripting et automation

## 🛠️ Stack technique

### Frameworks principaux
```python
# Apache Spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
```

### Outils de développement
- **Docker** : Conteneurisation des applications
- **Kubernetes** : Orchestration des conteneurs
- **Terraform** : Infrastructure as Code
- **Ansible** : Configuration management

## 📊 Projets réalisés

### Pipeline ETL pour e-commerce
**Technologies** : Apache Spark, PostgreSQL, Airflow  
**Résultat** : Traitement de 10M+ enregistrements/jour  
**Impact** : Réduction de 60% du temps de traitement

### API de données en temps réel
**Technologies** : FastAPI, Redis, Kafka  
**Résultat** : 1000 requêtes/seconde, 99.9% de disponibilité  
**Impact** : Amélioration de 40% des performances

### Data Lake sur AWS
**Technologies** : S3, Glue, Athena, Redshift  
**Résultat** : 100TB+ de données stockées  
**Impact** : Réduction de 50% des coûts de stockage

## 🔬 Méthodologie

### 1. Architecture de données
```python
# Configuration Spark
spark = SparkSession.builder \
    .appName("DataPipeline") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Lecture des données
df = spark.read \
    .format("parquet") \
    .option("path", "s3://bucket/data/") \
    .load()

# Transformation des données
df_transformed = df \
    .withColumn("date", to_date(col("timestamp"))) \
    .withColumn("hour", hour(col("timestamp"))) \
    .filter(col("amount") > 0) \
    .groupBy("date", "hour") \
    .agg(sum("amount").alias("total_amount"))
```

### 2. Pipeline ETL
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

def extract_data():
    """Extraction des données depuis la source"""
    import pandas as pd
    import requests
    
    # Extraction depuis API
    response = requests.get("https://api.example.com/data")
    data = response.json()
    
    # Sauvegarde en local
    df = pd.DataFrame(data)
    df.to_parquet("/tmp/raw_data.parquet")
    
    return "Data extracted successfully"

def transform_data():
    """Transformation des données"""
    import pandas as pd
    import numpy as np
    
    # Lecture des données
    df = pd.read_parquet("/tmp/raw_data.parquet")
    
    # Nettoyage et transformation
    df = df.dropna()
    df['amount'] = df['amount'].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculs métier
    df['amount_category'] = np.where(
        df['amount'] > 1000, 'high', 
        np.where(df['amount'] > 100, 'medium', 'low')
    )
    
    # Sauvegarde
    df.to_parquet("/tmp/transformed_data.parquet")
    
    return "Data transformed successfully"

def load_data():
    """Chargement des données vers la destination"""
    import pandas as pd
    import psycopg2
    
    # Lecture des données transformées
    df = pd.read_parquet("/tmp/transformed_data.parquet")
    
    # Connexion à la base de données
    conn = psycopg2.connect(
        host="localhost",
        database="analytics",
        user="user",
        password="password"
    )
    
    # Insertion des données
    df.to_sql('transactions', conn, if_exists='append', index=False)
    
    return "Data loaded successfully"

# Définition du DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='Pipeline ETL quotidien',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Définition des tâches
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

# Définition des dépendances
extract_task >> transform_task >> load_task
```

### 3. API de données
```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import redis
import json
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

app = FastAPI(title="Data API", version="1.0.0")

# Configuration Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Configuration base de données
engine = create_engine('postgresql://user:password@localhost/analytics')

class DataRequest(BaseModel):
    start_date: str
    end_date: str
    filters: dict = {}

class DataResponse(BaseModel):
    data: list
    total_records: int
    execution_time: float

@app.post("/data", response_model=DataResponse)
async def get_data(request: DataRequest):
    import time
    start_time = time.time()
    
    # Vérification du cache
    cache_key = f"data:{hash(str(request.dict()))}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        result = json.loads(cached_result)
        result['execution_time'] = time.time() - start_time
        return DataResponse(**result)
    
    # Requête à la base de données
    query = f"""
    SELECT * FROM transactions 
    WHERE date BETWEEN '{request.start_date}' AND '{request.end_date}'
    """
    
    # Application des filtres
    if request.filters:
        for key, value in request.filters.items():
            query += f" AND {key} = '{value}'"
    
    # Exécution de la requête
    df = pd.read_sql(query, engine)
    
    # Préparation de la réponse
    result = {
        'data': df.to_dict('records'),
        'total_records': len(df),
        'execution_time': time.time() - start_time
    }
    
    # Mise en cache
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return DataResponse(**result)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}
```

### 4. Monitoring et alertes
```python
import logging
from prometheus_client import Counter, Histogram, generate_latest
from flask import Flask, Response

# Métriques Prometheus
REQUEST_COUNT = Counter('data_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('data_request_duration_seconds', 'Request latency')
ERROR_COUNT = Counter('data_errors_total', 'Total errors')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        REQUEST_COUNT.inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        return response
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 📈 Métriques de performance

### Pipeline ETL
| Métrique | Valeur | Description |
|----------|--------|-------------|
| Throughput | 10M records/day | Volume traité par jour |
| Latence | 2 minutes | Temps de traitement |
| Disponibilité | 99.9% | Uptime du pipeline |
| Erreurs | < 0.1% | Taux d'erreur |

### API de données
| Métrique | Valeur | Description |
|----------|--------|-------------|
| QPS | 1000 req/s | Requêtes par seconde |
| Latence | 50ms | Temps de réponse |
| Disponibilité | 99.9% | Uptime de l'API |
| Cache Hit Rate | 85% | Taux de cache |

## 🎯 Bonnes pratiques

### Architecture
- **Scalabilité** : Design horizontal
- **Résilience** : Gestion des erreurs
- **Monitoring** : Métriques et alertes
- **Documentation** : Code et architecture

### Développement
- **Versioning** : Git et DVC
- **Testing** : Tests unitaires et d'intégration
- **CI/CD** : Automatisation des déploiements
- **Code Review** : Validation par les pairs

### Opérations
- **Monitoring** : Prometheus, Grafana
- **Logging** : ELK Stack
- **Alerting** : PagerDuty, Slack
- **Backup** : Stratégies de sauvegarde

## 🚀 Déploiement

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: data-api
  template:
    metadata:
      labels:
        app: data-api
    spec:
      containers:
      - name: data-api
        image: data-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://user:password@db:5432/analytics"
        - name: REDIS_URL
          value: "redis://redis:6379"
```

## 📚 Ressources d'apprentissage

### Cours recommandés
- **Udacity** : Data Engineering Nanodegree
- **Coursera** : Big Data Specialization
- **edX** : Data Engineering with Python

### Livres essentiels
- **Designing Data-Intensive Applications** (Martin Kleppmann)
- **Data Engineering Handbook** (Data Engineering Team)
- **Building Real-Time Data Pipelines** (Ben Stopford)

### Pratique
- **Apache Spark** : Documentation officielle
- **Airflow** : Documentation officielle
- **Docker** : Documentation officielle
- **Kubernetes** : Documentation officielle

---

*Dernière mise à jour : {{ git_revision_date_localized }}*
