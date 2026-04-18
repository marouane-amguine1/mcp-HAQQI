"""
MCP Server - Assistant Juridique Marocain
Utilise Ollama (Kimi) + ChromaDB (RAG) pour répondre aux questions de droit marocain
Refactorisé avec langchain_ollama et langchain_core.prompts
"""

import asyncio
import json
import os
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ─── Configuration ────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "kimi")          # nom du modèle dans Ollama
CHROMA_PATH  = os.getenv("CHROMA_PATH", "./data/chroma")  # dossier ChromaDB local
COLLECTION   = "lois_marocaines"
TOP_K        = 5   # nombre de chunks RAG retournés

# ─── LangChain Setup ────────────────────────────────────────────────────────────
system_template = """Tu es un expert en droit marocain. Tu fournis des conseils juridiques
précis et accessibles, basés exclusivement sur la législation marocaine en vigueur.

Règles :
- Cite toujours les articles et lois exacts (numéro de dahir, code, décret…)
- Réponds en français sauf si l'utilisateur écrit en arabe
- Signale clairement quand une question dépasse le cadre juridique ou nécessite un avocat
- Ne donne jamais de conseil contraire au droit marocain
- Structure ta réponse : Résumé → Base légale → Explications → Recommandation

Domaines couverts : droit civil, droit commercial, droit pénal, code de la famille,
droit du travail, droit administratif, droit immobilier, procédure civile.
"""

human_template = """Contexte juridique extrait de la législation marocaine :
{context}

Question de l'utilisateur :
{question}

Réponds en te basant principalement sur le contexte fourni ci-dessus.
Si le contexte ne couvre pas la question, indique-le clairement."""

# Création du prompt template avec langchain_core
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

# Initialisation du modèle Ollama avec langchain_ollama
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.3,
    num_ctx=8192
)

# ─── Initialisation ChromaDB ───────────────────────────────────────────────────
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

# ─── RAG : recherche de chunks pertinents ─────────────────────────────────────
def search_laws(query: str, n_results: int = TOP_K, domain_filter: str | None = None) -> list[dict]:
    col = get_collection()
    where = {"domain": domain_filter} if domain_filter else None
    results = col.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", ""),
                "law_name": meta.get("law_name", ""),
                "article": meta.get("article", ""),
                "domain": meta.get("domain", ""),
                "score": round(1 - dist, 3)
            })
    return chunks

# ─── LangChain : génération de réponse ─────────────────────────────────────────
async def call_ollama(question: str, context: str) -> str:
    """Utilise LangChain pour générer une réponse avec Ollama."""
    # Création de la chaîne (prompt + LLM)
    chain = prompt | llm

    # Invocation avec les variables
    response = await chain.ainvoke({
        "context": context,
        "question": question
    })

    return response.content

# ─── MCP Server ───────────────────────────────────────────────────────────────
app = Server("moroccan-legal-assistant")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_law",
            description=(
                "Pose une question juridique en droit marocain. "
                "L'outil recherche les lois pertinentes et génère une réponse experte."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "La question juridique (français ou arabe)"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domaine juridique optionnel pour filtrer",
                        "enum": [
                            "droit_civil", "droit_commercial", "droit_penal",
                            "code_famille", "droit_travail", "droit_administratif",
                            "droit_immobilier", "procedure_civile"
                        ]
                    }
                },
                "required": ["question"]
            }
        ),
        types.Tool(
            name="get_article",
            description="Récupère le texte exact d'un article ou d'une loi marocaine.",
            inputSchema={
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "Ex: 'article 418 code des obligations', 'dahir 1-04-22'"
                    }
                },
                "required": ["reference"]
            }
        ),
        types.Tool(
            name="list_domains",
            description="Liste les domaines juridiques disponibles dans la base de données.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="check_db_status",
            description="Vérifie le nombre de documents indexés dans la base de données juridique.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:

    # ── Tool : query_law ──────────────────────────────────────────────────────
    if name == "query_law":
        question = arguments["question"]
        domain   = arguments.get("domain")

        # 1. Recherche RAG
        chunks = search_laws(question, domain_filter=domain)
        if not chunks:
            return [types.TextContent(
                type="text",
                text=(
                    "⚠️ Aucun texte juridique pertinent trouvé dans la base de données.\n"
                    "Veuillez d'abord indexer les lois marocaines avec le script `scripts/index_laws.py`."
                )
            )]

        # 2. Construction du contexte
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] {c['law_name']} — {c['article']}\n"
                f"Source: {c['source']} (pertinence: {c['score']})\n"
                f"{c['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 3. Génération Ollama
        try:
            answer = await call_ollama(question, context)
        except Exception as e:
            answer = (
                f"❌ Impossible de se connecter à Ollama: {e}\n"
                f"Vérifiez qu'Ollama tourne sur {OLLAMA_URL} avec le modèle '{OLLAMA_MODEL}'."
            )

        # 4. Sources citées
        sources_list = "\n".join(
            f"  • {c['law_name']} — {c['article']} (score: {c['score']})"
            for c in chunks
        )
        result = f"{answer}\n\n---\n**Sources consultées :**\n{sources_list}"
        return [types.TextContent(type="text", text=result)]

    # ── Tool : get_article ────────────────────────────────────────────────────
    elif name == "get_article":
        reference = arguments["reference"]
        chunks = search_laws(reference, n_results=3)
        if not chunks:
            return [types.TextContent(type="text", text=f"Article non trouvé : {reference}")]

        result = f"**Résultats pour : {reference}**\n\n"
        for c in chunks:
            result += (
                f"📄 **{c['law_name']}** — {c['article']}\n"
                f"Domaine : {c['domain']} | Source : {c['source']}\n\n"
                f"{c['text']}\n\n---\n\n"
            )
        return [types.TextContent(type="text", text=result)]

    # ── Tool : list_domains ───────────────────────────────────────────────────
    elif name == "list_domains":
        col = get_collection()
        try:
            all_meta = col.get(include=["metadatas"])["metadatas"]
            domains = sorted({m.get("domain", "inconnu") for m in all_meta if m})
            total   = col.count()
            result  = f"**Base juridique marocaine — {total} chunks indexés**\n\n"
            result += "Domaines disponibles :\n"
            result += "\n".join(f"  • {d}" for d in domains)
        except Exception as e:
            result = f"Erreur : {e}"
        return [types.TextContent(type="text", text=result)]

    # ── Tool : check_db_status ────────────────────────────────────────────────
    elif name == "check_db_status":
        try:
            col    = get_collection()
            count  = col.count()
            status = f"✅ Base de données opérationnelle\n📚 {count} chunks de lois indexés"
        except Exception as e:
            status = f"❌ Erreur ChromaDB : {e}"
        return [types.TextContent(type="text", text=status)]

    return [types.TextContent(type="text", text=f"Outil inconnu : {name}")]

# ─── Démarrage ────────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
