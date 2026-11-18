import os
import re
import uuid
import fitz  # PyMuPDF
from typing import List, Dict, Optional

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings & Reranker
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Vector Store & Sparse Retrieval
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
# NOTE: never hardcode API keys in source. Set this in your shell/.env instead:
#   export GROQ_API_KEY="your-key-here"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Export it as an environment variable before running "
        "(and if you previously had a key hardcoded in this file, rotate it in the Groq console)."
    )

COLLECTION_NAME = "industry_rag"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "llama-3.3-70b-versatile"

# BGE models are asymmetric: retrieval quality drops noticeably if this instruction
# is not prepended to queries. It should NOT be added to the indexed passages.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Reranker relevance floor. Chunks scoring below this are treated as noise rather
# than being force-fed to the LLM as "context". Tune by inspecting rerank_score
# distributions for your corpus (bge-reranker-base scores are unbounded logits,
# not 0-1 probabilities) -- start here and adjust based on observed behavior.
MIN_RERANK_SCORE = -2.0
MAX_CONTEXT_CHUNKS = 6

# --- LAYOUT & CHUNKING CONFIG ---
IGNORE_PATTERNS = [
    "RFP Response No.",
    "TAB",
    "Local Staff",
    "Local\nStaff",
    "Local",
    "Staff",
    "PROJECT EXECUTIVE OVERSIGHT",
    "Wilder Institute/Calgary Zoo",
    "Clark Builders ©",
    "DocuSign Envelope ID",
    "All unauthorized copying",
    "Creating a Team Logo",
    "DO NOT MODIFY"
]

# Known sub-headings that should stay inline with the profile, formatted as markdown
SUBHEADINGS = {
    "EDUCATION", "CERTIFICATIONS", "SELECT EXPERIENCE",
    "ADDITIONAL EXPERIENCE", "HONORABLE MENTIONS",
    "STAFF & KEY PERSONNEL", "PRECONSTRUCTION CONSTRUCTION",
    "RENOVATIONTOWER & ENABLING", "RELEVANT PROJECT EXPERIENCE"
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)

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
    # recreate_collection is deprecated in newer qdrant-client; this is the
    # forward-compatible equivalent and avoids a silent no-op if the client
    # version changes behavior.
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("System ready.\n")


# --- LAYOUT-AWARE CHUNKING LOGIC ---
def is_heading(text: str) -> bool:
    """Checks if a line of text is an uppercase heading."""
    text = text.strip()
    if not text:
        return False
    text = text.replace("\n", " ")
    # Guard against long all-caps paragraphs (e.g. bolded disclaimers) being
    # misread as headings -- real headings are short.
    if len(text) < 3 or len(text) > 60:
        return False
    if len(text.split()) > 6:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z\s\-&']+", text))


def is_profile_name(text: str) -> bool:
    """
    Checks if a heading is specifically a person's name (2 to 4 words).
    Excludes known sub-headings like 'SELECT EXPERIENCE' to prevent splitting profiles.
    Requires every word to look like a name token (letters only, length >= 2) to
    reduce false positives from short all-caps labels such as "OUR TEAM" or "KEY DATES".
    """
    text = text.strip().replace("\n", " ")
    if not is_heading(text):
        return False
    if text in SUBHEADINGS:
        return False
    words = text.split()
    if not (2 <= len(words) <= 4):
        return False
    return all(w.isalpha() and len(w) >= 2 for w in words)


def extract_page_blocks(page):
    """
    Extracts blocks while maintaining strict column reading order.
    Reads Left Column top-to-bottom, then Right Column top-to-bottom.
    Filters out duplicate blocks and unwanted headers/footers.
    """
    blocks = page.get_text("blocks", sort=True)
    page_width = page.rect.width
    midline = page_width / 2

    classified = []
    seen_texts = set()  # Track seen texts to prevent duplicate text boxes

    for b in blocks:
        x0, y0, x1, y1, text, _, _ = b
        text = text.strip()
        if not text:
            continue
        if any(p in text for p in IGNORE_PATTERNS):
            continue

        # Deduplication check
        if text in seen_texts:
            continue
        seen_texts.add(text)

        # Strict Column Detection
        if x1 <= midline + 10:
            classified.append({"type": "left", "y": y0, "text": text})
        elif x0 >= midline - 10:
            classified.append({"type": "right", "y": y0, "text": text})
        else:
            classified.append({"type": "full", "y": y0, "text": text})

    classified.sort(key=lambda b: b["y"])

    extracted = []
    left_buffer = []
    right_buffer = []

    def flush_buffers():
        nonlocal left_buffer, right_buffer
        left_buffer.sort(key=lambda b: b["y"])
        right_buffer.sort(key=lambda b: b["y"])

        for item in left_buffer:
            extracted.append(item["text"])
        for item in right_buffer:
            extracted.append(item["text"])

        left_buffer = []
        right_buffer = []

    for item in classified:
        if item["type"] == "full":
            flush_buffers()
            extracted.append(item["text"])
        elif item["type"] == "right":
            right_buffer.append(item)
        else:
            left_buffer.append(item)

    flush_buffers()
    return extracted


def build_sections(blocks):
    """
    Builds logical sections. ONLY profile names (2-4 words) start a new parent
    section. Subheadings like 'EDUCATION' are converted to markdown '## EDUCATION'
    and stay inline with their paragraphs.
    """
    sections = []
    current_heading = "UNKNOWN"
    current_content = []

    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue

        block_text_clean = " ".join(lines)

        if len(block_text_clean) < 3:
            continue

        is_sub = False
        for sub in SUBHEADINGS:
            if block_text_clean.startswith(sub):
                is_sub = True
                current_content.append(f"## {sub}")
                remaining = block_text_clean[len(sub):].strip()
                if remaining:
                    current_content.append(remaining)
                break

        if is_sub:
            continue

        words = block_text_clean.split()
        name_candidate = " ".join(words[:2])
        if len(words) >= 3 and is_profile_name(" ".join(words[:3])):
            name_candidate = " ".join(words[:3])

        if is_profile_name(name_candidate):
            if name_candidate == current_heading:
                remaining_text = block_text_clean[len(name_candidate):].strip()
                if remaining_text:
                    current_content.append(remaining_text)
                continue

            if current_content:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_content)
                })
            current_heading = name_candidate
            current_content = []

            remaining_text = block_text_clean[len(name_candidate):].strip()
            if remaining_text:
                current_content.append(remaining_text)
        else:
            current_content.append(block)

    if current_content:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_content)
        })

    return sections


def layout_chunker(page, page_number, doc_id):
    blocks = extract_page_blocks(page)
    sections = build_sections(blocks)
    chunks = []

    for section in sections:
        if len(section["content"].strip()) < 30:
            continue

        parent_id = str(uuid.uuid4())

        parent_store[parent_id] = {
            "heading": section["heading"],
            "text": section["content"],
            "page": page_number,
            "doc_id": doc_id
        }

        children = splitter.split_text(section["content"])

        for child in children:
            if len(child.strip()) < 50:
                continue
            chunks.append({
                "child_id": str(uuid.uuid4()),
                "parent_id": parent_id,
                "heading": section["heading"],
                "text": child,
                "page": page_number,
                "doc_id": doc_id
            })

    return chunks


# --- INGESTION LOGIC ---
def ingest_pdf(file_path: str):
    global bm25, bm25_corpus, bm25_tokenized_corpus

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"\nIngesting '{file_path}'...")
    doc = fitz.open(file_path)
    doc_id = str(uuid.uuid4())
    all_child_chunks = []

    output_txt_path = "extracted_text_preview.txt"
    with open(output_txt_path, "w", encoding="utf-8") as text_file:

        for page_num, page in enumerate(doc):
            child_chunks = layout_chunker(page, page_num + 1, doc_id)
            all_child_chunks.extend(child_chunks)

            text_file.write(f"\n{'=' * 30} PAGE {page_num + 1} {'=' * 30}\n")
            for chunk in child_chunks:
                text_file.write(f"[{chunk['heading']}] {chunk['text']}\n---\n")

    print(
        f"[INFO] Full extracted text saved to: '{output_txt_path}'. Open this file to see exactly what the system read.\n")

    if not all_child_chunks:
        print("No text extracted from PDF.")
        return

    # 1. Index into Qdrant (Dense) -- NOTE: no query instruction prefix here.
    # BGE instructions are query-side only; adding them to passages hurts retrieval.
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
                "heading": chunk["heading"],
                "page": chunk["page"],
                "doc_id": chunk["doc_id"]
            }
        ))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    # 2. Index into BM25 (Sparse)
    print("Indexing into BM25...")
    bm25_corpus.extend(all_child_chunks)
    bm25_tokenized_corpus.extend([tokenize(chunk["text"]) for chunk in all_child_chunks])
    bm25 = BM25Okapi(bm25_tokenized_corpus)

    print(f"Success! Ingested {len(all_child_chunks)} child chunks from {len(doc)} pages.\n")


def tokenize(text: str) -> List[str]:
    """
    Word-boundary tokenizer for BM25. The original `.lower().split()` left
    punctuation glued to tokens (e.g. "experience," / "years." / "(PMP)"),
    which silently breaks exact-term matches against a clean query token.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


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

    # Deduplicate by normalized text, not just by id -- the same passage can
    # occasionally get indexed as separate dense/sparse entries with different
    # ids if ingestion is re-run, and duplicate context wastes reranker/LLM
    # budget without adding information.
    results = []
    seen_text = set()
    for id_ in reranked_ids:
        item = id_to_item[id_]
        norm = " ".join(item["text"].split()).lower()
        if norm in seen_text:
            continue
        seen_text.add(norm)
        results.append(item)
    return results


def query_system(query: str):
    # 1. DENSE RETRIEVAL (Qdrant) -- query gets the BGE retrieval instruction,
    # the indexed passages do not.
    query_vector = embedder.embed_query(BGE_QUERY_INSTRUCTION + query)
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=20
    )
    dense_hits = search_result.points
    dense_results = [{"child_id": hit.id, **hit.payload} for hit in dense_hits]

    # 2. SPARSE RETRIEVAL (BM25)
    if bm25:
        tokenized_query = tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = bm25_scores.argsort()[-20:][::-1]
        sparse_results = [bm25_corpus[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    else:
        sparse_results = []

    # 3. HYBRID FUSION (RRF)
    fused_results = reciprocal_rank_fusion([dense_results, sparse_results])
    top_candidates = fused_results[:15]

    if not top_candidates:
        print("\n" + "=" * 50)
        print("ANSWER:\nInsufficient information in provided context.")
        print("=" * 50 + "\n")
        return

    # 4. CROSS-ENCODER RERANKER
    pairs = [[query, doc["text"]] for doc in top_candidates]
    reranker_scores = reranker.predict(pairs)

    for doc, score in zip(top_candidates, reranker_scores):
        doc["rerank_score"] = float(score)
    top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 5. RELEVANCE FLOOR + DYNAMIC TOP-K
    # Take chunks above the relevance floor, up to MAX_CONTEXT_CHUNKS, instead
    # of always forcing exactly 5 -- this stops low-relevance filler from
    # padding out the context (and misleading the LLM) when a query genuinely
    # isn't well covered by the document.
    final_context_chunks = [
        c for c in top_candidates if c["rerank_score"] >= MIN_RERANK_SCORE
    ][:MAX_CONTEXT_CHUNKS]

    if not final_context_chunks:
        print("\n" + "=" * 50)
        print("ANSWER:\nInsufficient information in provided context.")
        print("=" * 50)
        print("(Best available match scored below the relevance threshold: "
              f"{top_candidates[0]['rerank_score']:.2f} < {MIN_RERANK_SCORE})")
        print("=" * 50 + "\n")
        return

    # 6. PARENT EXPANSION
    expanded_context = []
    seen_parents = set()

    for chunk in final_context_chunks:
        parent_id = chunk["parent_id"]
        if parent_id not in seen_parents:
            parent_data = parent_store[parent_id]
            expanded_context.append({
                "text": parent_data["text"],
                "page": parent_data["page"],
                "heading": parent_data.get("heading", ""),
                "chunk_id": parent_id
            })
            seen_parents.add(parent_id)

    # 7. LLM GENERATION
    context_str = ""
    sources = []
    for i, ctx in enumerate(expanded_context):
        heading = ctx.get('heading', '')
        header = f"[Source {i + 1}, Page {ctx['page']}"
        if heading and heading != "UNKNOWN":
            header += f", Section: {heading}"
        header += "]"
        context_str += f"{header}: {ctx['text']}\n\n"
        sources.append({"page": ctx["page"], "heading": heading, "text": ctx["text"][:150] + "..."})

    prompt = ChatPromptTemplate.from_template("""
    You are an expert industry analyst. Use ONLY the following context to answer the user's question.
    You MUST cite your sources using the format [Source X, Page Y].
    Do not use outside knowledge or make assumptions beyond what is stated in the context.
    If the context does not contain the answer, say exactly: "Insufficient information in provided context."

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
        heading_str = f" ({src['heading']})" if src.get('heading') and src['heading'] != "UNKNOWN" else ""
        print(f"[Source {i + 1}, Page {src['page']}{heading_str}]: {src['text']}")
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