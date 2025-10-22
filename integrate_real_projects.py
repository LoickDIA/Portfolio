#!/usr/bin/env python3
"""
Génère/Met à jour les pages projets MkDocs à partir de portfolio_data.json.
- Crée docs/portfolio-reel.md si absent
- Crée/maj docs/projects/<slug>.md pour chaque projet
Ne modifie PAS mkdocs.yml automatiquement (affiche la nav à coller).
"""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
PROJECTS_DIR = DOCS / "projects"
DATA = ROOT / "portfolio_data.json"

TEMPLATE_PROJECT = """# {name}

**Pitch (valeur métier).** {description}

## ⚙️ Stack
- **Langages** : {lang}
- **Frameworks** : {frameworks}
- **Outils** : {tools}

## 📊 Résultats (métriques)
{metrics_block}

## 🔗 Liens
- **Repo** : {repo_url}
- **Démo** : {demo_url}
"""

def ensure_dirs():
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

def fmt_list(xs):
    return ", ".join(xs) if xs else "ND"

def fmt_metrics(m):
    if not m:
        return "ND"
    lines = []
    for k, v in m.items():
        lines.append(f"- **{k}** : {v}")
    return "\n".join(lines)

def write_project_page(p):
    slug = p["slug"]
    path = PROJECTS_DIR / f"{slug}.md"
    content = TEMPLATE_PROJECT.format(
        name=p["name"],
        description=p["description"],
        lang=fmt_list(p.get("lang", [])),
        frameworks=fmt_list(p.get("frameworks", [])),
        tools=fmt_list(p.get("tools", [])),
        metrics_block=fmt_metrics(p.get("metrics", {})),
        repo_url=p.get("repo_url", "ND"),
        demo_url=p.get("demo_url", "ND"),
    )
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"[OK] {path}")

def write_overview(projects):
    overview = ["# Portfolio Réel", "", "## Projets", ""]
    for p in projects:
        overview.append(f"- [{p['name']}](projects/{p['slug']}.md) — {p['description']}")
    overview.append("\n---\n")
    overview.append("**Contacts**  \n- LinkedIn : <https://www.linkedin.com/in/loick-dernoncourt-241b8b123>  \n- GitHub : <https://github.com/LoickDIA>  \n- Email : <mailto:Dernoncourt.ck@gmail.com>")
    (DOCS / "portfolio-reel.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    print(f"[OK] {DOCS / 'portfolio-reel.md'}")

def print_nav(projects):
    print("\n---\nÀ ajouter dans `mkdocs.yml` (section nav):\n")
    print("  - Portfolio réel:")
    print("      - Vue d'ensemble: portfolio-reel.md")
    print("      - Projets réels:")
    for p in projects:
        print(f"          - {p['name']}: projects/{p['slug']}.md")

def main():
    ensure_dirs()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    projects = data["projects"]
    # pages
    for p in projects:
        write_project_page(p)
    write_overview(projects)
    print_nav(projects)

if __name__ == "__main__":
    main()