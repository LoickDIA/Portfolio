#!/usr/bin/env python3
"""
Script de test pour valider le portfolio MkDocs
"""

import os
import sys
import subprocess
import yaml
import markdown
from pathlib import Path

def test_mkdocs_config():
    """Test de la configuration MkDocs"""
    print("🔧 Test de la configuration MkDocs...")
    
    try:
        with open('mkdocs.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Vérifications de base
        assert 'site_name' in config, "site_name manquant"
        assert 'theme' in config, "theme manquant"
        assert 'nav' in config, "navigation manquante"
        assert 'plugins' in config, "plugins manquants"
        
        print("✅ Configuration MkDocs valide")
        return True
        
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return False

def test_markdown_files():
    """Test des fichiers Markdown"""
    print("📝 Test des fichiers Markdown...")
    
    docs_dir = Path('docs')
    required_files = [
        'index.md',
        'about.md',
        'contact.md',
        'methodologie.md',
        'innovations.md',
        'projects/index.md',
        'projects/template.md',
        'projects/detection-objets-temps-reel.md',
        'projects/classification-textes-avancee.md',
        'projects/prediction-churn-avancee.md',
        'skills/index.md',
        'skills/machine-learning.md',
        'skills/deep-learning.md',
        'skills/data-engineering.md',
        'skills/visualisation.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = docs_dir / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    if missing_files:
        print(f"❌ Fichiers manquants: {missing_files}")
        return False
    
    print("✅ Tous les fichiers Markdown présents")
    return True

def test_markdown_syntax():
    """Test de la syntaxe Markdown"""
    print("📖 Test de la syntaxe Markdown...")
    
    docs_dir = Path('docs')
    md_files = list(docs_dir.rglob('*.md'))
    
    errors = []
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Test de conversion Markdown
            md = markdown.Markdown(extensions=['codehilite', 'tables'])
            html = md.convert(content)
            
        except Exception as e:
            errors.append(f"{md_file}: {e}")
    
    if errors:
        print(f"❌ Erreurs Markdown: {errors}")
        return False
    
    print("✅ Syntaxe Markdown valide")
    return True

def test_requirements():
    """Test des dépendances"""
    print("📦 Test des dépendances...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        # Vérification que les packages peuvent être importés
        for req in requirements:
            if req.strip() and not req.startswith('#'):
                package = req.split('>=')[0].split('==')[0]
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    print(f"⚠️  Package {package} non installé")
        
        print("✅ Dépendances vérifiées")
        return True
        
    except Exception as e:
        print(f"❌ Erreur requirements: {e}")
        return False

def test_build():
    """Test de construction du site"""
    print("🏗️  Test de construction du site...")
    
    try:
        result = subprocess.run(['mkdocs', 'build'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ Erreur de construction: {result.stderr}")
            return False
        
        # Vérification que le dossier site existe
        if not Path('site').exists():
            print("❌ Dossier site non créé")
            return False
        
        print("✅ Construction réussie")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout lors de la construction")
        return False
    except Exception as e:
        print(f"❌ Erreur construction: {e}")
        return False

def test_links():
    """Test des liens internes"""
    print("🔗 Test des liens internes...")
    
    docs_dir = Path('docs')
    md_files = list(docs_dir.rglob('*.md'))
    
    broken_links = []
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Recherche des liens markdown
        import re
        links = re.findall(r'\[.*?\]\((.*?)\)', content)
        
        for link in links:
            if link.startswith('http'):
                continue  # Liens externes
            
            # Liens internes
            if link.startswith('#'):
                continue  # Ancres
            
            # Vérification du fichier cible
            target_path = docs_dir / link
            if not target_path.exists():
                broken_links.append(f"{md_file}: {link}")
    
    if broken_links:
        print(f"❌ Liens cassés: {broken_links}")
        return False
    
    print("✅ Liens internes valides")
    return True

def main():
    """Fonction principale de test"""
    print("🚀 Test du portfolio MkDocs")
    print("=" * 50)
    
    tests = [
        test_mkdocs_config,
        test_markdown_files,
        test_markdown_syntax,
        test_requirements,
        test_links,
        test_build
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur dans {test.__name__}: {e}")
            results.append(False)
        print()
    
    # Résumé
    print("=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests réussis: {passed}/{total}")
    print(f"❌ Tests échoués: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 Tous les tests sont passés ! Le portfolio est prêt.")
        return 0
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
