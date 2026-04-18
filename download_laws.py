"""
Script de téléchargement des lois marocaines depuis des sources officielles.
Sources :
  1. Bulletin Officiel (sgg.gov.ma)
  2. MizanQA (Hugging Face)
  3. data.gov.ma (Open Data)

Usage :
  python scripts/download_laws.py --source mizanqa
  python scripts/download_laws.py --source bo --year 2024
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path("./data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Source 1 : MizanQA (Hugging Face) ────────────────────────────────────────
def download_mizanqa():
    """
    Télécharge MizanQA, un benchmark QA sur le droit marocain.
    Nécessite : pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Installez : pip install datasets")
        sys.exit(1)

    print("⬇️  Téléchargement de MizanQA depuis HuggingFace...")
    ds = load_dataset("adlbh/MizanQA-v0", trust_remote_code=True)

    out_path = OUTPUT_DIR / "mizanqa.json"
    data = []
    for split in ds:
        for item in ds[split]:
            data.append({
                "law_name": item.get("law_name", "MizanQA"),
                "article":  item.get("article", ""),
                "domain":   item.get("category", "general"),
                "text":     item.get("context", item.get("answer", "")),
                "source":   "MizanQA / HuggingFace"
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(data)} entrées sauvegardées dans {out_path}")
    print("➡️  Lancez maintenant : python scripts/index_laws.py --source json --file data/raw/mizanqa.json")

# ─── Source 2 : Bulletin Officiel ─────────────────────────────────────────────
def download_bulletin_officiel(year: int = 2024):
    """
    Télécharge les PDFs du Bulletin Officiel marocain.
    Nécessite : pip install requests beautifulsoup4
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("❌ Installez : pip install requests beautifulsoup4")
        sys.exit(1)

    pdf_dir = OUTPUT_DIR / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    base_url = "https://www.sgg.gov.ma/BulletinsOfficielsFr.aspx"
    headers  = {"User-Agent": "Mozilla/5.0 (research bot)"}

    print(f"🌐 Accès au Bulletin Officiel ({year})...")
    try:
        r = requests.get(base_url, headers=headers, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")

        # Chercher les liens PDF de l'année souhaitée
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower() and str(year) in (a.text + href):
                if not href.startswith("http"):
                    href = "https://www.sgg.gov.ma/" + href.lstrip("/")
                pdf_links.append(href)

        if not pdf_links:
            print(f"⚠️  Aucun PDF trouvé pour {year} — structure du site peut avoir changé.")
            print("   Téléchargez manuellement sur : https://www.sgg.gov.ma/BulletinsOfficielsFr.aspx")
            return

        print(f"📋 {len(pdf_links)} bulletins trouvés pour {year}")
        for url in pdf_links[:10]:  # Limiter à 10 pour le test
            filename = pdf_dir / url.split("/")[-1]
            if filename.exists():
                print(f"  ⏭️  Déjà téléchargé : {filename.name}")
                continue
            try:
                r = requests.get(url, headers=headers, timeout=60)
                filename.write_bytes(r.content)
                print(f"  ✅ {filename.name} ({len(r.content)//1024} KB)")
                time.sleep(1)  # délai poli
            except Exception as e:
                print(f"  ❌ Erreur : {e}")

        print(f"\n➡️  Indexez les PDFs : python scripts/index_laws.py --source pdf --dir {pdf_dir}")
    except Exception as e:
        print(f"❌ Erreur de connexion : {e}")

# ─── Source 3 : Open Data Maroc ───────────────────────────────────────────────
def download_open_data():
    """
    Télécharge le dataset des lois depuis data.gov.ma
    """
    try:
        import requests
    except ImportError:
        print("❌ Installez : pip install requests")
        sys.exit(1)

    api_url = "https://data.gov.ma/data/api/3/action/datastore_search"
    datasets = [
        ("Liste des lois et règlements",
         "https://data.gov.ma/data/fr/dataset/liste-des-lois-et-reglements"),
    ]

    print("ℹ️  Sources Open Data disponibles sur data.gov.ma :")
    for name, url in datasets:
        print(f"  • {name} : {url}")

    print("\n📋 Instructions manuelles :")
    print("  1. Visitez : https://data.gov.ma/data/fr/dataset/?tags=juridique")
    print("  2. Téléchargez les fichiers CSV/JSON")
    print("  3. Convertissez avec : python scripts/index_laws.py --source json --file <fichier>")

# ─── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharger les lois marocaines")
    parser.add_argument("--source", choices=["mizanqa", "bo", "opendata"],
                        required=True, help="Source à télécharger")
    parser.add_argument("--year", type=int, default=2024,
                        help="Année pour le Bulletin Officiel (default: 2024)")
    args = parser.parse_args()

    if args.source == "mizanqa":
        download_mizanqa()
    elif args.source == "bo":
        download_bulletin_officiel(args.year)
    elif args.source == "opendata":
        download_open_data()
