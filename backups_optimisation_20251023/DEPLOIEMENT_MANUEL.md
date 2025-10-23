# 🚀 Guide de Déploiement Manuel

## 📋 **Étapes de Déploiement**

### 1. **Créer un Repository GitHub**

1. **Aller sur GitHub.com** et créer un nouveau repository
2. **Nom suggéré** : `portfolio` ou `data-science-portfolio`
3. **Visibilité** : Public (pour GitHub Pages gratuit)
4. **Initialiser** : Ne pas cocher "Add a README file"

### 2. **Configurer le Repository Local**

```bash
# Ajouter le remote (remplacer USERNAME par votre nom d'utilisateur GitHub)
git remote add origin https://github.com/USERNAME/portfolio.git

# Pousser le code
git push -u origin master
```

### 3. **Activer GitHub Pages**

1. **Aller dans Settings** du repository
2. **Scroller vers "Pages"** dans le menu de gauche
3. **Source** : Sélectionner "GitHub Actions"
4. **Sauvegarder** les paramètres

### 4. **Déploiement Automatique**

Le workflow GitHub Actions est déjà configuré dans `.github/workflows/deploy.yml` et se déclenchera automatiquement.

### 5. **Vérifier le Déploiement**

- **URL du site** : `https://USERNAME.github.io/portfolio`
- **Temps de déploiement** : 2-5 minutes
- **Vérification** : Aller dans l'onglet "Actions" du repository

## 🔧 **Configuration Alternative**

### **Déploiement avec MkDocs**

Si vous préférez déployer manuellement :

```bash
# Installation de mkdocs-gh-deploy
pip install mkdocs-gh-deploy

# Déploiement direct
mkdocs gh-deploy
```

### **Déploiement avec GitHub CLI**

```bash
# Installation de GitHub CLI
# macOS: brew install gh
# Windows: winget install GitHub.cli

# Authentification
gh auth login

# Création du repository
gh repo create portfolio --public

# Déploiement
git push -u origin master
```

## 📊 **Vérification du Déploiement**

### **Tests à Effectuer**

1. **Site accessible** : Vérifier que l'URL fonctionne
2. **Navigation** : Tester tous les liens
3. **Responsive** : Vérifier sur mobile
4. **Performance** : Temps de chargement < 3 secondes
5. **SEO** : Vérifier les métadonnées

### **Outils de Vérification**

- **PageSpeed Insights** : [pagespeed.web.dev](https://pagespeed.web.dev)
- **GTmetrix** : [gtmetrix.com](https://gtmetrix.com)
- **W3C Validator** : [validator.w3.org](https://validator.w3.org)

## 🎯 **Personnalisation Post-Déploiement**

### **1. Mettre à Jour les Informations**

```yaml
# Dans mkdocs.yml, remplacer :
site_author: "Votre Nom"
site_url: "https://USERNAME.github.io/portfolio"
analytics:
  property: "G-VOTRE-ID-GOOGLE-ANALYTICS"
```

### **2. Ajouter Votre Photo**

```bash
# Créer le dossier assets
mkdir -p docs/assets

# Ajouter votre photo
# docs/assets/profile.jpg
```

### **3. Personnaliser les Liens Sociaux**

```yaml
# Dans mkdocs.yml
social:
  - icon: fontawesome/brands/github
    link: https://github.com/VOTRE-USERNAME
  - icon: fontawesome/brands/linkedin
    link: https://linkedin.com/in/VOTRE-PROFIL
```

## 🚀 **Optimisations Post-Déploiement**

### **1. Google Analytics**

1. **Créer un compte Google Analytics**
2. **Obtenir l'ID de propriété** (format : G-XXXXXXXXXX)
3. **Remplacer dans mkdocs.yml**

### **2. Domaine Personnalisé (Optionnel)**

```bash
# Créer un fichier CNAME
echo "votre-domaine.com" > docs/CNAME

# Ou configurer dans .github/workflows/deploy.yml
# cname: votre-domaine.com
```

### **3. Optimisation SEO**

1. **Soumettre à Google Search Console**
2. **Créer un sitemap** (automatique avec MkDocs)
3. **Optimiser les métadonnées**

## 📱 **QR Codes**

### **Génération des QR Codes**

```bash
# Installer les dépendances
pip install qrcode[pil]

# Générer les QR codes
python generate_qr.py
```

### **Utilisation des QR Codes**

- **portfolio_qr.png** : QR code principal du portfolio
- **business_card_qr.png** : QR code pour carte de visite
- **qr_*.png** : QR codes pour projets spécifiques

## 🎉 **Félicitations !**

Votre portfolio est maintenant déployé et accessible au monde entier !

### **Prochaines Étapes**

1. **Partager** : Ajoutez le lien dans vos profils LinkedIn, GitHub, etc.
2. **Promouvoir** : Partagez sur les réseaux sociaux
3. **Maintenir** : Mettez à jour régulièrement le contenu
4. **Analyser** : Suivez les métriques avec Google Analytics

---

**🚀 Votre portfolio de data scientist est maintenant en ligne et prêt à vous ouvrir de nouvelles opportunités !** 🎊
