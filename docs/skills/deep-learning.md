# 🧠 Deep Learning

Expertise en deep learning avec 3+ années d'expérience et 10+ projets réalisés.

## 🎯 Compétences principales

### Architectures de réseaux
- **CNN** : ResNet, VGG, EfficientNet, MobileNet
- **RNN/LSTM** : Séries temporelles, NLP
- **Transformers** : BERT, GPT, Vision Transformer
- **GANs** : DCGAN, StyleGAN, CycleGAN
- **Autoencoders** : Variational Autoencoders (VAE)

### Frameworks et outils
- **PyTorch** : Framework principal (3 ans)
- **TensorFlow/Keras** : Framework secondaire (2 ans)
- **Hugging Face** : Transformers, Datasets
- **Weights & Biases** : Expérimentation
- **ONNX** : Optimisation et déploiement

### Domaines d'application
- **Computer Vision** : Classification, Détection, Segmentation
- **NLP** : Sentiment Analysis, Text Classification, NER
- **Time Series** : Prédiction, Anomaly Detection
- **Reinforcement Learning** : Q-Learning, Policy Gradient

## 🛠️ Stack technique

### Frameworks principaux
```python
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Hugging Face
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer

# Computer Vision
import torchvision.models as models
import cv2
from PIL import Image
```

### Outils de développement
- **Jupyter Lab** : Développement interactif
- **Weights & Biases** : Suivi des expériences
- **MLflow** : Gestion des modèles
- **Docker** : Conteneurisation

## 📊 Projets réalisés

### Classification d'images médicales
**Technologies** : ResNet50, Transfer Learning, PyTorch  
**Résultat** : 95.2% d'accuracy sur 10K images  
**Impact** : Réduction de 40% du temps de diagnostic

### Analyse de sentiment en temps réel
**Technologies** : BERT, Transformers, FastAPI  
**Résultat** : 94.5% d'accuracy, 50ms de latence  
**Impact** : API déployée avec 99.9% de disponibilité

### Détection d'anomalies industrielles
**Technologies** : CNN, OpenCV, YOLO  
**Résultat** : 99.5% de précision, 40% de réduction des défauts  
**Impact** : Amélioration significative de la qualité

## 🔬 Méthodologie

### 1. Préparation des données
```python
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transformations pour l'augmentation de données
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 2. Architecture de modèle
```python
import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(CustomCNN, self).__init__()
        
        # Backbone pré-entraîné
        if pretrained:
            self.backbone = models.resnet50(pretrained=True)
            # Geler les premières couches
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Dégeler les dernières couches
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        
        # Classification head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Initialisation du modèle
model = CustomCNN(num_classes=10, pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### 3. Entraînement
```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return running_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

# Configuration de l'entraînement
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# Boucle d'entraînement
num_epochs = 50
best_val_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Sauvegarde du meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    print('-' * 50)
```

### 4. Évaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

# Évaluation finale
predictions, targets, probabilities = evaluate_model(model, test_loader, device)
print(classification_report(targets, predictions))
```

## 📈 Métriques de performance

### Computer Vision
| Métrique | Valeur | Description |
|----------|--------|-------------|
| Accuracy | 95.2% | Précision globale |
| Precision | 94.8% | Précision par classe |
| Recall | 95.1% | Rappel par classe |
| F1-Score | 94.9% | Score F1 harmonique |
| AUC-ROC | 0.98 | Aire sous la courbe ROC |

### NLP
| Métrique | Valeur | Description |
|----------|--------|-------------|
| Accuracy | 94.5% | Précision globale |
| Precision | 94.1% | Précision par classe |
| Recall | 94.8% | Rappel par classe |
| F1-Score | 94.4% | Score F1 harmonique |
| Latence | 50ms | Temps de prédiction |

## 🎯 Bonnes pratiques

### Architecture
- **Transfer Learning** : Utiliser des modèles pré-entraînés
- **Regularization** : Dropout, Batch Normalization
- **Data Augmentation** : Rotation, Flip, Color Jitter
- **Ensemble Methods** : Combinaison de modèles

### Entraînement
- **Learning Rate Scheduling** : ReduceLROnPlateau
- **Early Stopping** : Éviter le surapprentissage
- **Gradient Clipping** : Stabiliser l'entraînement
- **Mixed Precision** : Accélérer l'entraînement

### Évaluation
- **Cross-Validation** : Validation robuste
- **Ablation Studies** : Analyse des composants
- **Error Analysis** : Compréhension des erreurs
- **Visualization** : Grad-CAM, t-SNE

## 🚀 Déploiement

### Optimisation
```python
import torch.onnx
import onnx
import onnxruntime

# Conversion ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    export_params=True, opset_version=11,
    do_constant_folding=True,
    input_names=['input'], output_names=['output']
)

# Validation ONNX
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

### API de prédiction
```python
from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI(title="Deep Learning Prediction API")

# Chargement du modèle
model = torch.load('best_model.pth', map_location='cpu')
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lecture de l'image
    image = Image.open(io.BytesIO(await file.read()))
    
    # Préprocessing
    image_tensor = transform(image).unsqueeze(0)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "probabilities": probabilities[0].tolist()
    }
```

## 📚 Ressources d'apprentissage

### Cours recommandés
- **Fast.ai** : Practical Deep Learning for Coders
- **Coursera** : Deep Learning Specialization (Andrew Ng)
- **Udacity** : Deep Learning Nanodegree

### Livres essentiels
- **Deep Learning** (Ian Goodfellow)
- **Hands-On Machine Learning** (Aurélien Géron)
- **Pattern Recognition and Machine Learning** (Christopher Bishop)

### Pratique
- **Papers with Code** : Recherche et implémentations
- **Hugging Face** : Modèles et datasets
- **PyTorch** : Documentation officielle
- **Weights & Biases** : Expérimentation

---

*Dernière mise à jour : {{ git_revision_date_localized }}*
