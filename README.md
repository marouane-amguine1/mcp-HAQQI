# ⚖️ Assistant Juridique Marocain — MCP + RAG + Ollama

Assistant intelligent en droit marocain, utilisant :

- **MCP** (Model Context Protocol) pour exposer des outils juridiques
- **ChromaDB + RAG** pour rechercher dans les lois marocaines
- **Ollama (Kimi)** pour générer des réponses expertes

---

---

## 🚀 Installation

### 1. Prérequis

```bash
# Python 3.11+
python --version

# Ollama installé et en cours d'exécution
ollama serve

# Télécharger le modèle Kimi (ou un autre)
ollama pull kimi
# Alternative si kimi n'est pas dispo :
# ollama pull mistral   (bon en français)
# ollama pull aya       (multilingue arabe/français)
# ollama pull qwen2.5   (bon en arabe)
```

### 2. Installation des dépendances

```bash
cd moroccan-legal-mcp
pip install -e .
# Ou manuellement :
pip install mcp chromadb sentence-transformers httpx pdfplumber
```

### 3. Indexer les lois (obligatoire avant utilisation)

**Option A : Données de démonstration (rapide, pour tester)**

```bash
python scripts/index_laws.py --source sample
```

**Option B : MizanQA (dataset académique de vraies lois marocaines)**

```bash
pip install datasets
python scripts/download_laws.py --source mizanqa
python scripts/index_laws.py --source json --file data/raw/mizanqa.json
```

**Option C : Vos propres PDFs de lois**

```bash
# Mettez vos PDFs dans data/raw/pdfs/
python scripts/index_laws.py --source pdf --dir ./data/raw/pdfs
```

**Option D : Bulletin Officiel (scraping)**

```bash
pip install requests beautifulsoup4
python scripts/download_laws.py --source bo --year 2024
python scripts/index_laws.py --source pdf --dir ./data/raw/pdfs
```

### 4. Tester le serveur

```bash
python -c "
import asyncio, sys
sys.path.insert(0, '.')
from scripts.index_laws import get_collection, search_laws
col = get_collection()
print(f'✅ {col.count()} chunks en base')
results = search_laws('mariage mineur')
for r in results:
    print(f'  • {r[\"law_name\"]} (score: {r[\"score\"]})')
"
```

### 5. Configurer Claude Desktop

Copiez le contenu de `claude_desktop_config.json` dans votre fichier de config Claude :

**macOS** : `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows** : `%APPDATA%\Claude\claude_desktop_config.json`

Remplacez `/CHEMIN/ABSOLU/vers/` par le chemin réel du projet, puis redémarrez Claude.

---

## 💬 Utilisation

Une fois configuré dans Claude Desktop, vous pouvez poser des questions comme :

> « Quelles sont les conditions légales pour le divorce au Maroc ? »
>
> « Est-ce qu'un employeur peut licencier sans préavis ? »
>
> « Quels sont mes droits si mon propriétaire ne rembourse pas ma caution ? »
>
> « ما هي حقوق المرأة في قانون الأسرة المغربي؟ »

---

## 📁 Structure du projet

```
moroccan-legal-mcp/
├── server.py                    # MCP Server principal
├── pyproject.toml               # Configuration Python
├── claude_desktop_config.json   # Config pour Claude Desktop
├── data/
│   ├── chroma/                  # Base ChromaDB (auto-créée)
│   └── raw/                     # Données brutes à indexer
│       ├── pdfs/                # Vos PDFs de lois
│       └── mizanqa.json         # Dataset MizanQA (si téléchargé)
└── scripts/
    ├── index_laws.py            # Indexation dans ChromaDB
    └── download_laws.py         # Téléchargement des sources
```

---

## 🔧 Configuration avancée

Variables d'environnement disponibles :

| Variable       | Défaut                   | Description                |
| -------------- | ------------------------ | -------------------------- |
| `OLLAMA_URL`   | `http://localhost:11434` | URL de l'instance Ollama   |
| `OLLAMA_MODEL` | `kimi`                   | Modèle Ollama à utiliser   |
| `CHROMA_PATH`  | `./data/chroma`          | Chemin de la base ChromaDB |

---

## 📚 Sources de données recommandées

| Source                                                                | Qualité    | Accès                 |
| --------------------------------------------------------------------- | ---------- | --------------------- |
| [MizanQA](https://huggingface.co/datasets/adlbh/MizanQA-v0)           | ⭐⭐⭐⭐   | Gratuit (HuggingFace) |
| [Bulletin Officiel](https://www.sgg.gov.ma/BulletinsOfficielsFr.aspx) | ⭐⭐⭐⭐⭐ | Gratuit (PDF)         |
| [data.gov.ma](https://data.gov.ma/data/fr/dataset/?tags=juridique)    | ⭐⭐⭐     | Gratuit (Open Data)   |

---

## ⚠️ Avertissement légal

Cet outil est un assistant informatif. Les conseils fournis **ne remplacent pas** l'avis d'un avocat qualifié. Pour toute situation juridique importante, consultez un professionnel du droit marocain.
