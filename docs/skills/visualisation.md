# 📊 Visualisation de données

Expertise en visualisation de données avec 4+ années d'expérience et 15+ projets réalisés.

## 🎯 Compétences principales

### Bibliothèques Python
- **Matplotlib** : Visualisations statiques et interactives
- **Seaborn** : Visualisations statistiques avancées
- **Plotly** : Graphiques interactifs et dashboards
- **Bokeh** : Visualisations web interactives
- **Altair** : Grammaire de graphiques

### Outils de BI
- **Tableau** : Dashboards et rapports
- **Power BI** : Visualisations Microsoft
- **Grafana** : Monitoring et métriques
- **D3.js** : Visualisations web personnalisées

### Types de visualisations
- **Statistiques** : Histogrammes, Box plots, Scatter plots
- **Temporelles** : Time series, Gantt charts
- **Géographiques** : Cartes, Heat maps
- **Hiérarchiques** : Treemaps, Sunburst
- **Réseaux** : Graph networks, Sankey diagrams

## 🛠️ Stack technique

### Bibliothèques Python
```python
# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Seaborn
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("husl")

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Pandas
import pandas as pd
import numpy as np
```

### Outils de développement
- **Jupyter Lab** : Développement interactif
- **Streamlit** : Applications web rapides
- **Dash** : Dashboards interactifs
- **Bokeh** : Visualisations web

## 📊 Projets réalisés

### Dashboard de ventes e-commerce
**Technologies** : Plotly, Dash, PostgreSQL  
**Résultat** : Dashboard interactif avec 10+ graphiques  
**Impact** : Amélioration de 25% de l'analyse des ventes

### Visualisation de données géographiques
**Technologies** : Folium, GeoPandas, OpenStreetMap  
**Résultat** : Cartes interactives avec 50K+ points  
**Impact** : Identification de 5 zones à fort potentiel

### Analyse de sentiment en temps réel
**Technologies** : Matplotlib, Seaborn, Real-time data  
**Résultat** : Graphiques temps réel avec 1K+ points/seconde  
**Impact** : Monitoring en temps réel des opinions

## 🔬 Méthodologie

### 1. Exploration des données
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv('data.csv')

# Statistiques descriptives
print(df.describe())

# Visualisation de la distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histogramme
axes[0, 0].hist(df['price'], bins=50, alpha=0.7)
axes[0, 0].set_title('Distribution des prix')
axes[0, 0].set_xlabel('Prix')
axes[0, 0].set_ylabel('Fréquence')

# Box plot
axes[0, 1].boxplot(df['price'])
axes[0, 1].set_title('Box plot des prix')
axes[0, 1].set_ylabel('Prix')

# Scatter plot
axes[1, 0].scatter(df['area'], df['price'], alpha=0.5)
axes[1, 0].set_title('Prix vs Surface')
axes[1, 0].set_xlabel('Surface')
axes[1, 0].set_ylabel('Prix')

# Corrélation
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Matrice de corrélation')

plt.tight_layout()
plt.show()
```

### 2. Visualisations avancées
```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Graphique interactif avec Plotly
fig = px.scatter(df, x='area', y='price', color='district',
                 size='rooms', hover_data=['price', 'area'],
                 title='Prix vs Surface par district')

fig.update_layout(
    title_font_size=20,
    xaxis_title="Surface (m²)",
    yaxis_title="Prix (€)",
    legend_title="District"
)

fig.show()

# Graphique en barres empilées
fig = px.bar(df, x='district', y='price', color='type',
             title='Prix par district et type de bien')

fig.update_layout(
    xaxis_title="District",
    yaxis_title="Prix moyen (€)",
    legend_title="Type de bien"
)

fig.show()
```

### 3. Dashboards interactifs
```python
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Layout du dashboard
app.layout = html.Div([
    html.H1("Dashboard de ventes", style={'textAlign': 'center'}),
    
    # Filtres
    html.Div([
        html.Label("Sélectionner le district:"),
        dcc.Dropdown(
            id='district-dropdown',
            options=[{'label': i, 'value': i} for i in df['district'].unique()],
            value=df['district'].unique()[0]
        )
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    # Graphiques
    dcc.Graph(id='price-chart'),
    dcc.Graph(id='area-chart')
])

# Callbacks pour l'interactivité
@app.callback(
    [Output('price-chart', 'figure'),
     Output('area-chart', 'figure')],
    [Input('district-dropdown', 'value')]
)
def update_charts(selected_district):
    filtered_df = df[df['district'] == selected_district]
    
    # Graphique des prix
    price_fig = px.histogram(filtered_df, x='price', nbins=30,
                           title=f'Distribution des prix - {selected_district}')
    
    # Graphique des surfaces
    area_fig = px.scatter(filtered_df, x='area', y='price',
                         title=f'Prix vs Surface - {selected_district}')
    
    return price_fig, area_fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

### 4. Visualisations géographiques
```python
import folium
import geopandas as gpd
from folium import plugins

# Création d'une carte interactive
m = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

# Ajout de marqueurs
for idx, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Prix: {row['price']}€<br>Surface: {row['area']}m²",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)

# Heatmap
heat_data = [[row['latitude'], row['longitude']] for idx, row in df.iterrows()]
plugins.HeatMap(heat_data).add_to(m)

# Sauvegarde de la carte
m.save('map.html')
```

## 📈 Types de visualisations

### Visualisations statistiques
```python
# Histogramme avec distribution normale
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['price'], bins=50, alpha=0.7, density=True, label='Données')
ax.axvline(df['price'].mean(), color='red', linestyle='--', label='Moyenne')
ax.axvline(df['price'].median(), color='green', linestyle='--', label='Médiane')
ax.set_xlabel('Prix (€)')
ax.set_ylabel('Densité')
ax.set_title('Distribution des prix')
ax.legend()
plt.show()

# Box plot par groupe
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='district', y='price', ax=ax)
ax.set_title('Distribution des prix par district')
ax.set_xlabel('District')
ax.set_ylabel('Prix (€)')
plt.xticks(rotation=45)
plt.show()
```

### Visualisations temporelles
```python
# Time series
fig, ax = plt.subplots(figsize=(15, 6))
df_time = df.groupby('date')['price'].mean().reset_index()
ax.plot(df_time['date'], df_time['price'], linewidth=2)
ax.set_title('Évolution du prix moyen dans le temps')
ax.set_xlabel('Date')
ax.set_ylabel('Prix moyen (€)')
plt.xticks(rotation=45)
plt.show()

# Graphique en aires empilées
fig = px.area(df, x='date', y='price', color='district',
              title='Évolution des prix par district')
fig.show()
```

### Visualisations géographiques
```python
# Carte de chaleur
fig = px.density_mapbox(df, lat='latitude', lon='longitude', z='price',
                        radius=10, center=dict(lat=48.8566, lon=2.3522),
                        zoom=10, mapbox_style="open-street-map")
fig.update_layout(title="Carte de chaleur des prix")
fig.show()
```

## 🎯 Bonnes pratiques

### Design
- **Couleurs** : Palette cohérente et accessible
- **Typographie** : Lisibilité et hiérarchie
- **Espacement** : Équilibre et clarté
- **Interactivité** : Engagement utilisateur

### Performance
- **Optimisation** : Réduction de la complexité
- **Caching** : Mise en cache des données
- **Lazy loading** : Chargement à la demande
- **Responsive** : Adaptation aux écrans

### Accessibilité
- **Contraste** : Respect des standards WCAG
- **Couleurs** : Support daltonien
- **Navigation** : Clavier et lecteurs d'écran
- **Texte** : Descriptions et légendes

## 🚀 Déploiement

### Streamlit
```python
import streamlit as st
import plotly.express as px
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Dashboard", layout="wide")

# Titre
st.title("Dashboard de ventes")

# Sidebar avec filtres
st.sidebar.header("Filtres")
district = st.sidebar.selectbox("District", df['district'].unique())
price_range = st.sidebar.slider("Fourchette de prix", 
                               int(df['price'].min()), 
                               int(df['price'].max()),
                               (int(df['price'].min()), int(df['price'].max())))

# Filtrage des données
filtered_df = df[(df['district'] == district) & 
                 (df['price'] >= price_range[0]) & 
                 (df['price'] <= price_range[1])]

# Métriques
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Nombre de biens", len(filtered_df))
with col2:
    st.metric("Prix moyen", f"{filtered_df['price'].mean():,.0f}€")
with col3:
    st.metric("Surface moyenne", f"{filtered_df['area'].mean():.0f}m²")
with col4:
    st.metric("Prix au m²", f"{filtered_df['price'].mean()/filtered_df['area'].mean():.0f}€/m²")

# Graphiques
fig = px.scatter(filtered_df, x='area', y='price', color='type',
                 title='Prix vs Surface par type de bien')
st.plotly_chart(fig, use_container_width=True)
```

### Dash
```python
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de ventes"),
    dcc.Graph(id='scatter-plot'),
    dcc.Slider(
        id='price-slider',
        min=df['price'].min(),
        max=df['price'].max(),
        value=df['price'].max(),
        marks={str(price): str(price) for price in df['price'].unique()},
        step=None
    )
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('price-slider', 'value')
)
def update_figure(selected_price):
    filtered_df = df[df['price'] <= selected_price]
    fig = px.scatter(filtered_df, x='area', y='price', color='type')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

## 📚 Ressources d'apprentissage

### Cours recommandés
- **Coursera** : Data Visualization with Python
- **Udacity** : Data Visualization Nanodegree
- **edX** : Data Visualization Fundamentals

### Livres essentiels
- **The Visual Display of Quantitative Information** (Edward Tufte)
- **Storytelling with Data** (Cole Nussbaumer Knaflic)
- **Data Visualization** (Kieran Healy)

### Pratique
- **Matplotlib** : Documentation officielle
- **Seaborn** : Documentation officielle
- **Plotly** : Documentation officielle
- **D3.js** : Documentation officielle

---

*Dernière mise à jour : {{ git_revision_date_localized }}*
