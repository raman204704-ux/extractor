import os
import uuid
import fitz  # PyMuPDF
from typing import List, Dict, Optional

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Embeddings & Reranker
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Vector Store & Sparse Retrieval
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "industry_rag"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "llama-3.3-70b-versatile"

# --- GLOBAL STATE ---
parent_store: Dict[str, Dict] = {}
bm25_corpus: List[Dict] = []
bm25_tokenized_corpus: List[List[str]] = []
bm25: Optional[BM25Okapi] = None
embedder = None
reranker = None
llm = None
qdrant = None


# --- INITIALIZATION ---
def init_system():
    global embedder, reranker, llm, qdrant, bm25

    print("Initializing Embedder...")
    embedder = HuggingFaceEmbeddings(model_name=DENSE_MODEL)

    print("Initializing Reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("Initializing Groq LLM...")
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=4000,
        api_key=GROQ_API_KEY
    )

    print("Initializing Qdrant (In-Memory)...")
    qdrant = QdrantClient(":memory:")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("System ready.\n")


# --- INGESTION LOGIC ---
def hierarchical_chunker(text: str, page_num: int, doc_id: str) -> List[Dict]:
    chunks = []
    parent_size = 1500
    child_size = 400
    child_overlap = 100

    for i in range(0, len(text), parent_size):
        parent_text = text[i:i + parent_size]
        parent_id = str(uuid.uuid4())

        parent_store[parent_id] = {
            "text": parent_text,
            "page": page_num,
            "doc_id": doc_id
        }

        for j in range(0, len(parent_text), child_size - child_overlap):
            child_text = parent_text[j:j + child_size]
            if len(child_text.strip()) < 50: continue

            chunks.append({
                "child_id": str(uuid.uuid4()),
                "parent_id": parent_id,
                "text": child_text,
                "page": page_num,
                "doc_id": doc_id
            })
    return chunks


def ingest_pdf(file_path: str):
    global bm25, bm25_corpus, bm25_tokenized_corpus

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"\nIngesting '{file_path}'...")
    doc = fitz.open(file_path)
    doc_id = str(uuid.uuid4())
    all_child_chunks = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if not text.strip(): continue
        child_chunks = hierarchical_chunker(text, page_num + 1, doc_id)
        all_child_chunks.extend(child_chunks)

    if not all_child_chunks:
        print("No text extracted from PDF.")
        return

    # 1. Index into Qdrant (Dense)
    print("Generating embeddings and uploading to Qdrant...")
    points = []
    texts_to_embed = [chunk["text"] for chunk in all_child_chunks]
    vectors = embedder.embed_documents(texts_to_embed)

    for chunk, vector in zip(all_child_chunks, vectors):
        points.append(PointStruct(
            id=chunk["child_id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                "parent_id": chunk["parent_id"],
                "page": chunk["page"],
                "doc_id": chunk["doc_id"]
            }
        ))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    # 2. Index into BM25 (Sparse)
    print("Indexing into BM25...")
    bm25_corpus.extend(all_child_chunks)
    bm25_tokenized_corpus.extend([chunk["text"].lower().split() for chunk in all_child_chunks])
    bm25 = BM25Okapi(bm25_tokenized_corpus)

    print(f"Success! Ingested {len(all_child_chunks)} child chunks from {len(doc)} pages.\n")


# --- RETRIEVAL LOGIC ---
def reciprocal_rank_fusion(lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    fused_scores = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            if item["child_id"] not in fused_scores:
                fused_scores[item["child_id"]] = 0
            fused_scores[item["child_id"]] += 1 / (k + rank)

    reranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    id_to_item = {item["child_id"]: item for lst in lists for item in lst}
    return [id_to_item[id] for id in reranked_ids]


def query_system(query: str):
    # 1. DENSE RETRIEVAL (Qdrant)
    query_vector = embedder.embed_query(query)
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=20
    )
    dense_hits = search_result.points
    dense_results = [{"child_id": hit.id, **hit.payload} for hit in dense_hits]
    # 2. SPARSE RETRIEVAL (BM25)
    if bm25:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = bm25_scores.argsort()[-20:][::-1]
        sparse_results = [bm25_corpus[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    else:
        sparse_results = []

    # 3. HYBRID FUSION (RRF)
    fused_results = reciprocal_rank_fusion([dense_results, sparse_results])
    top_candidates = fused_results[:15]

    # 4. CROSS-ENCODER RERANKER
    if top_candidates:
        pairs = [[query, doc["text"]] for doc in top_candidates]
        reranker_scores = reranker.predict(pairs)

        for doc, score in zip(top_candidates, reranker_scores):
            doc["rerank_score"] = score
        top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    final_context_chunks = top_candidates[:5]

    # 5. PARENT EXPANSION
    expanded_context = []
    seen_parents = set()

    for chunk in final_context_chunks:
        parent_id = chunk["parent_id"]
        if parent_id not in seen_parents:
            parent_data = parent_store[parent_id]
            expanded_context.append({
                "text": parent_data["text"],
                "page": parent_data["page"],
                "chunk_id": parent_id
            })
            seen_parents.add(parent_id)

    # 6. LLM GENERATION
    context_str = ""
    sources = []
    for i, ctx in enumerate(expanded_context):
        context_str += f"[Source {i + 1}, Page {ctx['page']}]: {ctx['text']}\n\n"
        sources.append({"page": ctx["page"], "text": ctx["text"][:150] + "..."})

    prompt = ChatPromptTemplate.from_template("""
    You are an expert industry analyst. Use the following context to answer the user's question.
    You MUST cite your sources using the format [Source X, Page Y].
    If the context does not contain the answer, say "Insufficient information in provided context."

    Context:
    {context}

    Question: {question}
    """)

    chain = prompt | llm | StrOutputParser()
    print("\nThinking...")
    answer = chain.invoke({"context": context_str, "question": query})

    print("\n" + "=" * 50)
    print("ANSWER:")
    print(answer)
    print("=" * 50)

import os
import uuid
import fitz  # PyMuPDF
from typing import List, Dict, Optional

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Embeddings & Reranker
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Vector Store & Sparse Retrieval
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "industry_rag"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "llama-3.3-70b-versatile"

# --- GLOBAL STATE ---
parent_store: Dict[str, Dict] = {}
bm25_corpus: List[Dict] = []
bm25_tokenized_corpus: List[List[str]] = []
bm25: Optional[BM25Okapi] = None
embedder = None
reranker = None
llm = None
qdrant = None


# --- INITIALIZATION ---
def init_system():
    global embedder, reranker, llm, qdrant, bm25

    print("Initializing Embedder...")
    embedder = HuggingFaceEmbeddings(model_name=DENSE_MODEL)

    print("Initializing Reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("Initializing Groq LLM...")
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=4000,
        api_key=GROQ_API_KEY
    )

    print("Initializing Qdrant (In-Memory)...")
    qdrant = QdrantClient(":memory:")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("System ready.\n")


# --- INGESTION LOGIC ---
def hierarchical_chunker(text: str, page_num: int, doc_id: str) -> List[Dict]:
    chunks = []
    parent_size = 1500
    child_size = 400
    child_overlap = 100

    for i in range(0, len(text), parent_size):
        parent_text = text[i:i + parent_size]
        parent_id = str(uuid.uuid4())

        parent_store[parent_id] = {
            "text": parent_text,
            "page": page_num,
            "doc_id": doc_id
        }

        for j in range(0, len(parent_text), child_size - child_overlap):
            child_text = parent_text[j:j + child_size]
            if len(child_text.strip()) < 50: continue

            chunks.append({
                "child_id": str(uuid.uuid4()),
                "parent_id": parent_id,
                "text": child_text,
                "page": page_num,
                "doc_id": doc_id
            })
    return chunks


def ingest_pdf(file_path: str):
    global bm25, bm25_corpus, bm25_tokenized_corpus

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"\nIngesting '{file_path}'...")
    doc = fitz.open(file_path)
    doc_id = str(uuid.uuid4())
    all_child_chunks = []

    # Create a text file to save the raw extracted text so you can view it
    output_txt_path = "extracted_text_preview.txt"
    with open(output_txt_path, "w", encoding="utf-8") as text_file:

        for page_num, page in enumerate(doc):
            text = page.get_text("text",sort=True)
            if not text.strip(): continue

            # Write the extracted text to the file
            text_file.write(f"\n{'=' * 30} PAGE {page_num + 1} {'=' * 30}\n")
            text_file.write(text)
            text_file.write("\n")

            # Print a quick preview to the console
            print(f"--- Extracted Page {page_num + 1} (Preview) ---")
            print(text[:200].replace('\n', ' ') + "...\n")

            child_chunks = hierarchical_chunker(text, page_num + 1, doc_id)
            all_child_chunks.extend(child_chunks)

    print(
        f"[INFO] Full extracted text saved to: '{output_txt_path}'. Open this file to see exactly what the system read.\n")

    if not all_child_chunks:
        print("No text extracted from PDF.")
        return

    # 1. Index into Qdrant (Dense)
    print("Generating embeddings and uploading to Qdrant...")
    points = []
    texts_to_embed = [chunk["text"] for chunk in all_child_chunks]
    vectors = embedder.embed_documents(texts_to_embed)

    for chunk, vector in zip(all_child_chunks, vectors):
        points.append(PointStruct(
            id=chunk["child_id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                "parent_id": chunk["parent_id"],
                "page": chunk["page"],
                "doc_id": chunk["doc_id"]
            }
        ))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    # 2. Index into BM25 (Sparse)
    print("Indexing into BM25...")
    bm25_corpus.extend(all_child_chunks)
    bm25_tokenized_corpus.extend([chunk["text"].lower().split() for chunk in all_child_chunks])
    bm25 = BM25Okapi(bm25_tokenized_corpus)

    print(f"Success! Ingested {len(all_child_chunks)} child chunks from {len(doc)} pages.\n")


# --- RETRIEVAL LOGIC ---
def reciprocal_rank_fusion(lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    fused_scores = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            if item["child_id"] not in fused_scores:
                fused_scores[item["child_id"]] = 0
            fused_scores[item["child_id"]] += 1 / (k + rank)

    reranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    id_to_item = {item["child_id"]: item for lst in lists for item in lst}
    return [id_to_item[id] for id in reranked_ids]


def query_system(query: str):
    # 1. DENSE RETRIEVAL (Qdrant)
    query_vector = embedder.embed_query(query)
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=20
    )
    dense_hits = search_result.points
    dense_results = [{"child_id": hit.id, **hit.payload} for hit in dense_hits]

    # 2. SPARSE RETRIEVAL (BM25)
    if bm25:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = bm25_scores.argsort()[-20:][::-1]
        sparse_results = [bm25_corpus[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    else:
        sparse_results = []

    # 3. HYBRID FUSION (RRF)
    fused_results = reciprocal_rank_fusion([dense_results, sparse_results])
    top_candidates = fused_results[:15]

    # 4. CROSS-ENCODER RERANKER
    if top_candidates:
        pairs = [[query, doc["text"]] for doc in top_candidates]
        reranker_scores = reranker.predict(pairs)

        for doc, score in zip(top_candidates, reranker_scores):
            doc["rerank_score"] = score
        top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    final_context_chunks = top_candidates[:5]

    # 5. PARENT EXPANSION
    expanded_context = []
    seen_parents = set()

    for chunk in final_context_chunks:
        parent_id = chunk["parent_id"]
        if parent_id not in seen_parents:
            parent_data = parent_store[parent_id]
            expanded_context.append({
                "text": parent_data["text"],
                "page": parent_data["page"],
                "chunk_id": parent_id
            })
            seen_parents.add(parent_id)

    # 6. LLM GENERATION
    context_str = ""
    sources = []
    for i, ctx in enumerate(expanded_context):
        context_str += f"[Source {i + 1}, Page {ctx['page']}]: {ctx['text']}\n\n"
        sources.append({"page": ctx["page"], "text": ctx["text"][:150] + "..."})

    prompt = ChatPromptTemplate.from_template("""
    You are an expert industry analyst. Use the following context to answer the user's question.
    You MUST cite your sources using the format [Source X, Page Y].
    If the context does not contain the answer, say "Insufficient information in provided context."

    Context:
    {context}

    Question: {question}
    """)

    chain = prompt | llm | StrOutputParser()
    print("\nThinking...")
    answer = chain.invoke({"context": context_str, "question": query})

    print("\n" + "=" * 50)
    print("ANSWER:")
    print(answer)
    print("=" * 50)
    print("Sources used:")
    for i, src in enumerate(sources):
        print(f"[Source {i + 1}, Page {src['page']}]: {src['text']}")
    print("=" * 50 + "\n")


# --- MAIN CLI LOOP ---
if __name__ == "__main__":
    init_system()

    pdf_path = input("Enter the path to your PDF file (e.g., hospital.pdf): ")
    ingest_pdf(pdf_path)

    print("You can now ask questions about the document. Type 'exit' to quit.")
    while True:
        user_query = input("> ")
        if user_query.lower() in ['exit', 'quit']:
            break
        if not user_query.strip():
            continue
        query_system(user_query)
    print("Sources used:")
    for i, src in enumerate(sources):
        print(f"[Source {i + 1}, Page {src['page']}]: {src['text']}")
    print("=" * 50 + "\n")


# --- MAIN CLI LOOP ---
if __name__ == "__main__":
    init_system()

    pdf_path = input("Enter the path to your PDF file (e.g., hospital.pdf): ")
    ingest_pdf(pdf_path)

    print("You can now ask questions about the document. Type 'exit' to quit.")
    while True:
        user_query = input("> ")
        if user_query.lower() in ['exit', 'quit']:
            break
        if not user_query.strip():
            continue
        query_system(user_query)