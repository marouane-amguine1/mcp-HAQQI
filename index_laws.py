"""
Script d'indexation des lois marocaines dans ChromaDB
Supporte : PDF, TXT, JSON (format MizanQA)

Usage :
  python scripts/index_laws.py --source pdf  --dir ./data/raw/pdfs
  python scripts/index_laws.py --source json --file ./data/raw/mizanqa.json
  python scripts/index_laws.py --source url  --url https://data.gov.ma/...
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ─── Configuration ─────────────────────────────────────────────────────────────
CHROMA_PATH  = os.getenv("CHROMA_PATH", "./data/chroma")
COLLECTION   = "lois_marocaines"
CHUNK_SIZE   = 800    # caractères par chunk
CHUNK_OVERLAP = 150   # chevauchement pour ne pas couper le contexte

DOMAIN_KEYWORDS = {
    "droit_civil":        ["civil", "obligations", "contrats", "dommage", "responsabilité"],
    "droit_commercial":   ["commercial", "société", "fonds de commerce", "registre"],
    "droit_penal":        ["pénal", "peine", "crime", "délit", "contravent"],
    "code_famille":       ["famille", "mariage", "divorce", "moudawwana", "tutelle", "héritage"],
    "droit_travail":      ["travail", "salarié", "employeur", "licenciement", "syndicat"],
    "droit_administratif":["administratif", "administration", "recours", "tribunal administratif"],
    "droit_immobilier":   ["immobilier", "foncier", "propriété", "hypothèque", "agence urbaine"],
    "procedure_civile":   ["procédure", "tribunal", "appel", "cassation", "huissier"],
}

# ─── Utilitaires ───────────────────────────────────────────────────────────────
def detect_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {d: sum(text_lower.count(kw) for kw in kws) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe le texte en chunks avec chevauchement."""
    chunks = []
    start  = 0
    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end]
        # Couper proprement à la fin d'une phrase si possible
        last_period = max(chunk.rfind(". "), chunk.rfind(".\n"), chunk.rfind("؟"))
        if last_period > chunk_size // 2:
            end   = start + last_period + 1
            chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 100]

def extract_article_number(text: str) -> str:
    """Extrait le numéro d'article le plus proche dans le texte."""
    patterns = [
        r"[Aa]rticle\s+(\d+[\-\w]*)",
        r"المادة\s+(\d+)",
        r"الفصل\s+(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return f"Art. {m.group(1)}"
    return ""

# ─── Collection ChromaDB ───────────────────────────────────────────────────────
def get_collection():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

# ─── Indexation depuis PDF ─────────────────────────────────────────────────────
def index_from_pdfs(pdf_dir: str):
    try:
        import pdfplumber
    except ImportError:
        print("❌ Installez pdfplumber : pip install pdfplumber")
        sys.exit(1)

    col      = get_collection()
    pdf_dir  = Path(pdf_dir)
    pdfs     = list(pdf_dir.glob("**/*.pdf"))
    print(f"📂 {len(pdfs)} PDFs trouvés dans {pdf_dir}")

    total = 0
    for pdf_path in pdfs:
        print(f"  📄 Traitement : {pdf_path.name}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except Exception as e:
            print(f"    ⚠️ Erreur lecture : {e}")
            continue

        if not full_text.strip():
            print(f"    ⚠️ PDF vide ou non extractible (scannérisé ?)")
            continue

        law_name = pdf_path.stem.replace("_", " ").replace("-", " ")
        domain   = detect_domain(full_text)
        chunks   = chunk_text(full_text)

        ids       = []
        documents = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{pdf_path.stem}_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source":   str(pdf_path),
                "law_name": law_name,
                "article":  extract_article_number(chunk),
                "domain":   domain,
                "chunk_id": i
            })

        # Ajout par batch de 50
        for j in range(0, len(ids), 50):
            col.upsert(
                ids=ids[j:j+50],
                documents=documents[j:j+50],
                metadatas=metadatas[j:j+50]
            )
        total += len(ids)
        print(f"    ✅ {len(ids)} chunks indexés (domaine: {domain})")

    print(f"\n✅ Terminé : {total} chunks ajoutés. Total en base : {col.count()}")

# ─── Indexation depuis JSON (format MizanQA ou personnalisé) ──────────────────
def index_from_json(json_file: str):
    col = get_collection()
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supporte deux formats : liste de lois ou format MizanQA
    laws = data if isinstance(data, list) else data.get("laws", data.get("data", []))

    ids = []; documents = []; metadatas = []
    for i, item in enumerate(laws):
        # Format MizanQA : {"question": ..., "answer": ..., "context": ..., "category": ...}
        if "context" in item:
            text   = item.get("context", "")
            domain = item.get("category", detect_domain(text)).lower().replace(" ", "_")
            law_name = item.get("law_name", f"MizanQA #{i}")
        # Format personnalisé : {"law_name": ..., "text": ..., "domain": ..., "article": ...}
        elif "text" in item:
            raw_text = item["text"]
            # Supporte text comme string ou liste de strings
            text     = " ".join(raw_text) if isinstance(raw_text, list) else raw_text
            domain   = item.get("domain", detect_domain(text))
            law_name = item.get("law_name", f"Loi #{i}")
        else:
            continue

        chunks = chunk_text(text)
        for j, chunk in enumerate(chunks):
            ids.append(f"law_{i}_chunk_{j}")
            documents.append(chunk)
            metadatas.append({
                "source":   json_file,
                "law_name": law_name,
                "article":  item.get("article", extract_article_number(chunk)),
                "domain":   domain,
                "chunk_id": j
            })

    for k in range(0, len(ids), 100):
        col.upsert(
            ids=ids[k:k+100],
            documents=documents[k:k+100],
            metadatas=metadatas[k:k+100]
        )

    print(f"✅ {len(ids)} chunks indexés depuis {json_file}. Total : {col.count()}")

# ─── Indexation d'exemple (données de test) ───────────────────────────────────
def index_sample_data():
    """Indexe quelques articles de démonstration pour tester."""
    col = get_collection()
    sample_laws = [
        {
            "id": "cod_oblig_art264",
            "law_name": "Code des Obligations et Contrats (DOC)",
            "article": "Art. 264",
            "domain": "droit_civil",
            "text": (
                "Article 264 : Celui qui cause, par sa faute, un dommage à autrui, "
                "est obligé de le réparer. Toute stipulation contraire est sans effet. "
                "La faute consiste, soit à omettre ce qu'on était tenu de faire, "
                "soit à faire ce dont on était tenu de s'abstenir, sans intention de causer "
                "un dommage. Dahir du 12 août 1913 formant le Code des Obligations et Contrats."
            ),
            "source": "Bulletin Officiel"
        },
        {
            "id": "moudawwana_art19",
            "law_name": "Code de la Famille (Moudawwana)",
            "article": "Art. 19",
            "domain": "code_famille",
            "text": (
                "Article 19 : La capacité matrimoniale s'acquiert, pour le garçon et la fille "
                "jouissant de leurs facultés mentales, à dix-huit ans révolus. "
                "Le juge de la famille peut autoriser le mariage du garçon et de la fille "
                "avant l'âge de la capacité matrimoniale, par décision motivée précisant "
                "l'intérêt et les raisons justifiant ce mariage. "
                "Loi n° 70-03 portant Code de la Famille, Bulletin Officiel n° 5358."
            ),
            "source": "Bulletin Officiel n° 5358"
        },
        {
            "id": "code_travail_art34",
            "law_name": "Code du Travail",
            "article": "Art. 34",
            "domain": "droit_travail",
            "text": (
                "Article 34 : Le contrat de travail à durée indéterminée peut cesser "
                "par la volonté du salarié, par la volonté de l'employeur ou par la volonté "
                "des deux parties. Le licenciement du salarié n'est valable que s'il est fondé "
                "sur une faute grave ou sur une nécessité de fonctionnement de l'entreprise. "
                "Loi n° 65-99 relative au Code du Travail."
            ),
            "source": "Loi 65-99"
        },
        {
            "id": "constitution_art19",
            "law_name": "Constitution du Royaume du Maroc",
            "article": "Art. 19",
            "domain": "droit_administratif",
            "text": (
                "Article 19 : L'homme et la femme jouissent, à égalité, des droits et libertés "
                "à caractère civil, politique, économique, social, culturel et environnemental, "
                "énoncés dans le présent titre et dans les autres dispositions de la Constitution, "
                "ainsi que dans les conventions et pactes internationaux dûment ratifiés par le Maroc. "
                "Constitution du Royaume du Maroc, adoptée par référendum le 1er juillet 2011."
            ),
            "source": "Constitution 2011"
        },
        {
            "id": "code_penal_art401",
            "law_name": "Code Pénal",
            "article": "Art. 401-403",
            "domain": "droit_penal",
            "text": (
                "Article 401 : L'homicide commis volontairement est qualifié meurtre. "
                "Il est puni de la réclusion perpétuelle. "
                "Article 402 : Tout meurtre commis avec préméditation ou guet-apens est qualifié "
                "assassinat et puni de mort. "
                "Article 403 : Constituent des circonstances aggravantes de l'assassinat : "
                "lorsqu'il a précédé, accompagné ou suivi un autre crime. "
                "Dahir n° 1-59-413 portant approbation du texte du Code Pénal."
            ),
            "source": "Dahir 1-59-413"
        },
    ]

    ids = [l["id"] for l in sample_laws]
    docs = [l["text"] for l in sample_laws]
    metas = [{k: v for k, v in l.items() if k not in ("id", "text")} for l in sample_laws]

    col.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"✅ {len(ids)} articles de démonstration indexés. Total : {col.count()}")

# ─── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexer les lois marocaines")
    parser.add_argument("--source", choices=["pdf", "json", "sample"], default="sample",
                        help="Source des données (default: sample)")
    parser.add_argument("--dir",  help="Dossier de PDFs (pour --source pdf)")
    parser.add_argument("--file", help="Fichier JSON (pour --source json)")
    args = parser.parse_args()

    if args.source == "pdf":
        if not args.dir:
            print("❌ --dir requis pour --source pdf"); sys.exit(1)
        index_from_pdfs(args.dir)
    elif args.source == "json":
        if not args.file:
            print("❌ --file requis pour --source json"); sys.exit(1)
        index_from_json(args.file)
    else:
        print("📋 Indexation des données de démonstration...")
        index_sample_data()
