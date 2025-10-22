#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Déploiement Complet du Portfolio MkDocs"
echo "=========================================="

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "mkdocs.yml" ]; then
    echo "❌ Erreur : mkdocs.yml non trouvé. Êtes-vous dans le bon répertoire ?"
    exit 1
fi

# Vérifier que Git est initialisé
if [ ! -d ".git" ]; then
    echo "❌ Erreur : Repository Git non initialisé"
    exit 1
fi

# Demander l'URL du repository GitHub
echo "📝 Veuillez entrer l'URL de votre repository GitHub :"
echo "   Exemple : https://github.com/loick-dernoncourt/portfolio.git"
read -p "URL du repository : " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ Erreur : URL du repository requise"
    exit 1
fi

echo "🔧 Configuration du repository..."
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"

echo "📦 Ajout des fichiers..."
git add .

echo "💾 Commit des changements..."
git commit -m "Portfolio MkDocs avec projets réels - $(date '+%Y-%m-%d %H:%M:%S')" || echo "Aucun changement à committer"

echo "🚀 Poussée vers GitHub..."
git push -u origin master

echo "🌐 Déploiement sur GitHub Pages..."
mkdocs gh-deploy

echo "✅ Déploiement terminé !"
echo "=========================================="
echo "🎉 Votre portfolio est maintenant en ligne !"
echo "📱 URL : https://loick-dernoncourt.github.io/portfolio/"
echo "🔗 Ou : https://loick-dernoncourt.github.io/loick-dernoncourt.github.io/"
echo ""
echo "📋 Prochaines étapes :"
echo "1. Vérifier le site en ligne"
echo "2. Activer GitHub Pages dans les settings"
echo "3. Compléter les informations pour VALMED et SaaS"
echo "4. Partager sur LinkedIn et GitHub"
echo ""
echo "🎊 Félicitations ! Votre portfolio est prêt ! 🚀"
