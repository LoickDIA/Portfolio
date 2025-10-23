---
title: "Guide SEO — Optimisation Portfolio Data Science"
description: "Guide SEO pour optimiser la visibilité du portfolio Data Science : mots-clés, structure, métadonnées, bonnes pratiques techniques."
---

# Guide SEO — Optimisation Portfolio Data Science

Ce guide vous accompagne dans l'optimisation SEO de votre portfolio pour maximiser sa visibilité et son impact professionnel.

## 🎯 Stratégie SEO

### Mots-clés cibles
- **Primaires** : Data Scientist, Machine Learning, Python, Streamlit
- **Secondaires** : Computer Vision, NLP, MLOps, Data Engineering
- **Longue traîne** : "Data Scientist Python", "Machine Learning Engineer", "Deep Learning Expert"

### Métadonnées optimisées
```yaml
# Configuration SEO dans mkdocs.yml
site_description: "Portfolio de Loïck Dernoncourt - Data Scientist expert en Machine Learning, Python et Streamlit. Projets en Computer Vision, NLP et MLOps."
site_author: "Loïck Dernoncourt"
site_url: "https://loick-dernoncourt.github.io/portfolio"

# Métadonnées par page
extra:
  social:
    - property: "og:title"
      content: "Portfolio Data Scientist - Loïck Dernoncourt"
    - property: "og:description"
      content: "Découvrez mes projets en Machine Learning, Python et Data Science"
    - property: "og:image"
      content: "https://loick-dernoncourt.github.io/portfolio/assets/og-image.png"
    - property: "og:type"
      content: "website"
    - property: "twitter:card"
      content: "summary_large_image"
    - property: "twitter:title"
      content: "Portfolio Data Scientist - Loïck Dernoncourt"
    - property: "twitter:description"
      content: "Expert en Machine Learning, Python et Data Science"
    - property: "twitter:image"
      content: "https://loick-dernoncourt.github.io/portfolio/assets/twitter-image.png"
```

## 📈 Optimisation du contenu

### Structure des titres
```markdown
# H1 : Mots-clés principaux (1 par page)
## H2 : Mots-clés secondaires
### H3 : Mots-clés longue traîne
#### H4 : Sous-sujets
```

### Densité de mots-clés
- **Titre H1** : 1-2% de densité
- **Contenu principal** : 2-3% de densité
- **Mots-clés LSI** : 5-10% de densité
- **Variations** : 3-5 variations par mot-clé

### Liens internes
```markdown
# Structure de liens internes
- Page d'accueil → Projets
- Projets → Compétences
- Compétences → Méthodologie
- Méthodologie → Lab
- Lab → Feedback
```

## 🚀 Optimisation technique

### Performance
```yaml
# Configuration de performance
plugins:
  - minify:
      minify_html: true
      minify_js: true
      minify_css: true
  - git-revision-date-localized:
      enable_creation_date: true
      fallback_to_build_date: true
```

### Mobile-First
```css
/* Optimisation mobile */
@media (max-width: 768px) {
  .md-content {
    font-size: 16px;
    line-height: 1.6;
  }
  
  .md-header {
    padding: 0.5rem;
  }
}
```

### Optimisation des images
```python
# Script d'optimisation des images
from PIL import Image
import os

def optimize_images():
    """Optimise toutes les images du portfolio"""
    
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                # Redimensionnement
                with Image.open(file_path) as img:
                    # Redimensionner si trop grande
                    if img.width > 1200:
                        img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                    
                    # Optimisation
                    img.save(file_path, optimize=True, quality=85)
                    
                print(f"✅ Image optimisée : {file_path}")

# Exécution
optimize_images()
```

## 📊 Analytics et monitoring

### Google Analytics 4
```html
<!-- Configuration GA4 -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

### Search Console
```xml
<!-- Sitemap pour Search Console -->
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://loick-dernoncourt.github.io/portfolio/</loc>
    <lastmod>2024-12-19</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://loick-dernoncourt.github.io/portfolio/projects/</loc>
    <lastmod>2024-12-19</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>
```

## 🎯 Stratégie de contenu

### Calendrier éditorial
- **Lundi** : Nouveau projet
- **Mercredi** : Article technique
- **Vendredi** : Mise à jour des métriques
- **Dimanche** : Réflexion méthodologique

### Fréquence de mise à jour
- **Contenu principal** : Mensuel
- **Projets** : Bimensuel
- **Métriques** : Hebdomadaire
- **Blog** : Bihebdomadaire

### Types de contenu
- **Tutoriels** : Guides pas-à-pas
- **Cas d'usage** : Études de cas réels
- **Comparaisons** : Outils et technologies
- **Tendances** : Veille technologique

## 🔗 Netlinking

### Liens internes
```markdown
# Structure de liens internes
- Page d'accueil → Projets (3-5 liens)
- Projets → Compétences (2-3 liens)
- Compétences → Méthodologie (1-2 liens)
- Méthodologie → Lab (1-2 liens)
```

### Liens externes
```markdown
# Liens vers des ressources de qualité
- Documentation officielle
- Articles de référence
- Outils et frameworks
- Communautés et forums
```

### Partenariats
- **Blogs techniques** : Échanges de liens
- **Communautés** : Mentions et partages
- **Conférences** : Présentations et networking
- **Collaborations** : Projets communs

## 📱 Optimisation mobile

### Responsive Design
```css
/* Breakpoints optimisés */
@media (max-width: 480px) {
  .md-content {
    padding: 1rem;
  }
  
  .md-header__title {
    font-size: 1.2rem;
  }
}

@media (max-width: 768px) {
  .md-nav {
    display: none;
  }
  
  .md-header__button {
    display: block;
  }
}
```

### Performance mobile
- **Temps de chargement** : < 3 secondes
- **Taille des images** : < 500KB
- **JavaScript** : Minifié et optimisé
- **CSS** : Critique et différé

## 🎯 Objectifs SEO

### Métriques à atteindre
- **Position moyenne** : Top 10 pour mots-clés cibles
- **Trafic organique** : +50% en 6 mois
- **Taux de clic** : > 5% sur SERP
- **Temps sur site** : > 3 minutes

### KPIs de succès
- **Visiteurs uniques** : 10,000/mois
- **Pages vues** : 25,000/mois
- **Taux de rebond** : < 40%
- **Pages par session** : > 3

### Évolution attendue
- **Mois 1-3** : Indexation et positionnement
- **Mois 4-6** : Amélioration des positions
- **Mois 7-9** : Stabilisation du trafic
- **Mois 10-12** : Croissance organique

## 🛠️ Outils SEO

### Outils d'analyse
- **Google Analytics** : Métriques détaillées
- **Search Console** : Performance de recherche
- **PageSpeed Insights** : Performance technique
- **Lighthouse** : Audit complet

### Outils de monitoring
- **SEMrush** : Analyse de la concurrence
- **Ahrefs** : Backlinks et mots-clés
- **Screaming Frog** : Audit technique
- **GTmetrix** : Performance web

## 🚀 Plan d'action SEO

### Phase 1 (Mois 1-2)
- [ ] **Audit technique** : Performance et structure
- [ ] **Optimisation on-page** : Titres, descriptions, contenu
- [ ] **Configuration analytics** : GA4 et Search Console
- [ ] **Sitemap** : Génération et soumission

### Phase 2 (Mois 3-4)
- [ ] **Contenu optimisé** : Articles et projets
- [ ] **Liens internes** : Structure et navigation
- [ ] **Images optimisées** : Compression et alt text
- [ ] **Mobile-first** : Responsive design

### Phase 3 (Mois 5-6)
- [ ] **Netlinking** : Liens externes de qualité
- [ ] **Contenu régulier** : Blog et mises à jour
- [ ] **Social signals** : Partages et mentions
- [ ] **Monitoring** : Suivi des performances

## 📞 Contact

**Besoin d'aide pour l'optimisation SEO de votre portfolio ?**

<div class="grid cards" markdown>

-   :material-email:{ .lg .middle } **Discutons de votre projet**

    ---

    [Dernoncourt.ck@gmail.com](mailto:Dernoncourt.ck@gmail.com)

-   :material-linkedin:{ .lg .middle } **Connectons-nous**

    ---

    [Profil professionnel](https://www.linkedin.com/in/loick-dernoncourt-241b8b123)

-   :material-github:{ .lg .middle } **Voir le code**

    ---

    [GitHub LoickDIA](https://github.com/LoickDIA)

</div>

**→ [Voir mes projets phares](portfolio-reel.md) | [Découvrir ma méthodologie](methodologie.md)**
