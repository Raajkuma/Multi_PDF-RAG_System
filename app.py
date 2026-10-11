import os
import re
import json
import hashlib
import itertools
import unicodedata
from collections import Counter

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import pymupdf4llm

try:
    import pymupdf
except ImportError:  # older PyMuPDF versions
    import fitz as pymupdf

# ----------------------------------------------------------------------------
# Environment (.env) + LangSmith tracing
# ----------------------------------------------------------------------------
load_dotenv()

# Gemini: accept either GOOGLE_API_KEY or GEMINI_API_KEY
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# LangSmith: accept new (LANGSMITH_*) or old (LANGCHAIN_*) variable names
_ls_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_ENABLED = bool(_ls_key)
if LANGSMITH_ENABLED:
    os.environ["LANGSMITH_API_KEY"] = _ls_key
    os.environ["LANGCHAIN_API_KEY"] = _ls_key
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "document-qa-rag")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])

GEMINI_ENABLED = bool(os.getenv("GOOGLE_API_KEY"))

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
INITIAL_DATA_FILE = "data.txt"
INDEX_DIR = "faiss_index"
REGISTRY_FILE = os.path.join(INDEX_DIR, "registry.json")  # {file_hash: file_name}
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SUPPORTED_EXT = (".pdf", ".txt", ".md")

# Chunking
CHUNK_SIZE = 1200        # max characters per chunk (text only, section path is added on top)
CHUNK_OVERLAP = 300     # characters of overlap, always whole sentences/paragraphs
MIN_WORDS = 5           # chunks with fewer words are dropped (page numbers, stray captions)
MIN_TAIL_CHARS = 200    # a tiny last piece of a section is merged into the previous chunk

# LLM and retrieval settings (fixed, not shown in the UI)
GEMINI_MODEL ="gemini-3.1-flash-lite" #"gemini-3.8-flash"
TEMPERATURE = 0.0
TOP_K = 5  # Final top documents passed to prompt after reranking

NO_ANSWER = "No any data available regarding the question"

PROMPT_TEMPLATE = """You are a strict question-answering assistant for a library of documents.
Answer the question using ONLY the numbered context sources below.

Rules:
1. Use only information that is explicitly stated in the context.
2. Do NOT use outside knowledge and do NOT guess or infer beyond the context.
3. If the answer is not explicitly found in the context, reply with exactly this sentence and nothing else (no citations):
{no_answer}
4. After every statement taken from the context, cite its source number in square brackets, like [1] or [2][3]. Cite only numbers that appear in the context.
5. If several documents are relevant, combine them and cite each one. Keep the answer clear and concise.

Context:
{context}

Question: {question}

Answer:"""


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner="Loading reranker model...")
def get_reranker():
    return CrossEncoder(RERANK_MODEL)


@st.cache_resource
def get_llm(model_name: str, temperature: float):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


# ----------------------------------------------------------------------------
# Registry helpers (tracks which files are already ingested)
# ----------------------------------------------------------------------------
def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def bytes_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ----------------------------------------------------------------------------
# Citation helpers
# ----------------------------------------------------------------------------
def page_label(meta: dict) -> str:
    p, pe = meta.get("page_number"), meta.get("page_end")
    if not p:
        return ""
    return f"{p}-{pe}" if pe and pe != p else str(p)


def cite_label(meta: dict) -> str:
    src = meta.get("source", "unknown")
    pg = page_label(meta)
    return f"{src}, p.{pg}" if pg else src


CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def resolve_citations(answer: str, docs: list):
    """
    Keeps clean numeric citations ([1], [2][3]) in the answer text instead of 
    expanding them into filenames and page numbers, while tracking used sources.
    """
    used = []

    def repl(match):
        nums = []
        for part in re.split(r"[,;]", match.group(1)):
            n = int(part.strip())
            if 1 <= n <= len(docs):
                if n not in used:
                    used.append(n)
                if n not in nums:
                    nums.append(n)
        return "".join([f"[{n}]" for n in nums]) if nums else ""

    resolved = CITE_RE.sub(repl, answer)
    # Clean up duplicate consecutive citations if any
    resolved = re.sub(r"(\[\d+\])\s*\1", r"\1", resolved)
    return resolved, used


# ----------------------------------------------------------------------------
# PDF extraction (layout-aware: headings, lists, tables, reading order)
# ----------------------------------------------------------------------------
def extract_pdf_pages(data: bytes) -> list:
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        if getattr(doc, "needs_pass", False):
            raise ValueError("PDF is password protected")
        try:
            pages = pymupdf4llm.to_markdown(
                doc, page_chunks=True, header=False, footer=False, show_progress=False
            )
        except TypeError:
            pages = pymupdf4llm.to_markdown(doc, page_chunks=True)
    finally:
        doc.close()
    return [p["text"] if isinstance(p, dict) else str(p) for p in pages]


# ----------------------------------------------------------------------------
# Content-aware cleaning
# ----------------------------------------------------------------------------
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_RE = re.compile(r"^(?:[-*•▪◦]|\d{1,3}[.)])\s+")
NOISE_LINE_RE = re.compile(r"^(?:==>.*<==|[-*_=]{3,})$")
PAGE_NUM_RE = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.I)
SKIP_SECTION_RE = re.compile(
    r"^\s*(?:\d+[.)]?\s*|[ivxlc]+[.)]\s+)?"
    r"(?:references?|bibliography|works cited|literature cited|citations|"
    r"acknowledge?ments?|table of contents|contents)\s*:?\s*$",
    re.I,
)
REF_HINTS = re.compile(
    r"doi\.org|\bdoi:|arxiv|https?://|\bet al\.|\bpp\.|\bvol\.|proceedings|journal of|\bisbn\b", re.I
)


def clean_inline(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")
    s = re.sub(r"\*\*|__", "", s)
    s = re.sub(r"(?<![\w*])_(?=\S)(.+?)(?<=\S)_(?![\w*])", r"\1", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def remove_page_furniture(pages: list) -> list:
    n = len(pages)
    split = [p.splitlines() for p in pages]

    def key(line):
        s = clean_inline(re.sub(r"^[#>\s]+", "", line)).lower()
        return re.sub(r"\d+", "#", s)

    def edges(lines):
        idx = [i for i, l in enumerate(lines) if l.strip()]
        return set(idx[:3] + idx[-3:])

    counts = Counter()
    for lines in split:
        counts.update({key(lines[i]) for i in edges(lines)})
    threshold = max(3, int(0.4 * n))
    repeated = {k for k, c in counts.items() if k and c >= threshold} if n >= 4 else set()

    kept_headings = set()
    cleaned = []
    for lines in split:
        drop = set()
        for i in sorted(edges(lines)):
            k = key(lines[i])
            bare = re.sub(r"^[\W_]+|[\W_]+$", "", clean_inline(lines[i]))
            if PAGE_NUM_RE.match(bare):
                drop.add(i)
            elif k in repeated:
                if lines[i].lstrip().startswith("#") and k not in kept_headings:
                    kept_headings.add(k)
                else:
                    drop.add(i)
        cleaned.append("\n".join(l for i, l in enumerate(lines) if i not in drop))
    return cleaned


def looks_like_references(text: str) -> bool:
    hits = len(REF_HINTS.findall(text))
    return hits >= 3 and hits / max(len(text.split()), 1) * 100 >= 3


def looks_like_toc(text: str) -> bool:
    return len(re.findall(r"(?:\.\s?){4,}\s*\d+", text)) >= 3


def is_junk(text: str) -> bool:
    words = re.findall(r"[^\W\d_]{2,}", text)
    return len(words) < MIN_WORDS or looks_like_references(text) or looks_like_toc(text)


# ----------------------------------------------------------------------------
# Document-aware parsing
# ----------------------------------------------------------------------------
def join_run(lines: list, unwrap: bool):
    if lines[0].startswith("|"):
        kind = "table"
    elif LIST_RE.match(lines[0]):
        kind = "list"
    else:
        kind = "paragraph"
    if not unwrap:
        return "\n".join(lines), kind

    out = []
    for s in lines:
        if out and not LIST_RE.match(s) and not s.startswith("|") and not out[-1].startswith("|"):
            prev = out[-1]
            if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha() and s[:1].islower():
                out[-1] = prev[:-1] + s
            else:
                out[-1] = prev + " " + s
        else:
            out.append(s)
    return "\n".join(out), kind


def parse_blocks(pages: list, unwrap: bool) -> list:
    blocks, stack = [], []
    run, code, in_code = [], [], False
    page = None

    def trail():
        return tuple(t for _, t in stack)

    def flush_run():
        nonlocal run
        if run:
            text, kind = join_run(run, unwrap)
            if text.strip():
                blocks.append({"text": text, "kind": kind, "path": trail(), "ps": page, "pe": page})
        run = []

    for page, md in pages:
        for raw in md.splitlines():
            stripped = raw.strip()
            if stripped.startswith("```"):
                if in_code:
                    if code:
                        blocks.append(
                            {"text": "\n".join(code), "kind": "code", "path": trail(), "ps": page, "pe": page}
                        )
                    code, in_code = [], False
                else:
                    flush_run()
                    in_code = True
                continue
            if in_code:
                code.append(raw.rstrip())
                continue

            s = clean_inline(raw)
            if not s or NOISE_LINE_RE.match(s):
                flush_run()
                continue

            m = HEADING_RE.match(stripped)
            if m and len(m.group(2)) <= 150:
                flush_run()
                level, title = len(m.group(1)), clean_inline(m.group(2)).strip("# ").strip()
                if title and not PAGE_NUM_RE.match(title):
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                continue

            run.append(s)
        flush_run()
        if in_code and code:
            blocks.append({"text": "\n".join(code), "kind": "code", "path": trail(), "ps": page, "pe": page})
            code, in_code = [], False
    return blocks


def merge_cross_page(blocks: list) -> list:
    out = []
    for b in blocks:
        if out:
            a = out[-1]
            if (
                a["kind"] == "paragraph"
                and b["kind"] == "paragraph"
                and a["path"] == b["path"]
                and a["pe"] is not None
                and b["ps"] == a["pe"] + 1
                and not re.search(r"[.!?:;\"”')\]]$", a["text"])
                and b["text"][:1].islower()
            ):
                a["text"] = (
                    a["text"][:-1] + b["text"]
                    if a["text"].endswith("-") and a["text"][-2:-1].isalpha()
                    else a["text"] + " " + b["text"]
                )
                a["pe"] = b["pe"]
                continue
        out.append(dict(b))
    return out


# ----------------------------------------------------------------------------
# Content-aware chunking
# ----------------------------------------------------------------------------
def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text)
    return [p for p in parts if p.strip()]


def hard_split(text: str, size: int) -> list:
    pieces, cur = [], ""
    for word in text.split():
        while len(word) > size:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(word[:size])
            word = word[size:]
        if cur and len(cur) + 1 + len(word) > size:
            pieces.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def fit(part: str, size: int) -> list:
    if len(part) <= size:
        return [part]
    sentences = split_sentences(part)
    if len(sentences) > 1:
        return [p for s in sentences for p in fit(s, size)]
    return hard_split(part, size)


def block_to_units(block: dict, size: int) -> list:
    text, kind = block["text"], block["kind"]
    if len(text) <= size:
        return [{"text": text, "joiner": "\n\n", "ps": block["ps"], "pe": block["pe"]}]

    prose = kind == "paragraph" and "\n" not in text
    parts = [text] if prose else text.split("\n")
    part_joiner = " " if prose else "\n"
    units = []
    for part in parts:
        for j, piece in enumerate(fit(part, size)):
            joiner = "\n\n" if not units else (part_joiner if j == 0 else " ")
            units.append({"text": piece, "joiner": joiner, "ps": block["ps"], "pe": block["pe"]})
    return units


def split_table(block: dict, size: int) -> list:
    lines = block["text"].split("\n")
    header = lines[:2] if len(lines) > 2 and re.fullmatch(r"\|?[\s:|-]+\|?", lines[1]) else []
    rows = lines[len(header):]
    pieces, cur = [], []
    for row in rows:
        if cur and len("\n".join(header + cur + [row])) > size:
            pieces.append({"text": "\n".join(header + cur), "ps": block["ps"], "pe": block["pe"]})
            cur = []
        cur.append(row)
    if cur:
        pieces.append({"text": "\n".join(header + cur), "ps": block["ps"], "pe": block["pe"]})
    return pieces


def _make_chunk(units: list) -> dict:
    text = units[0]["text"]
    for u in units[1:]:
        text += u["joiner"] + u["text"]
    pages = [p for u in units for p in (u["ps"], u["pe"]) if p]
    return {"text": text, "ps": min(pages) if pages else None, "pe": max(pages) if pages else None}


def chunk_section(blocks: list, size: int, overlap: int) -> list:
    chunks, cur, n_carry = [], [], 0

    def joined_len(units, extra):
        return sum(len(u["text"]) for u in units) + sum(len(u["joiner"]) for u in units[1:]) + (
            len(extra["joiner"]) + len(extra["text"]) if units else len(extra["text"])
        )

    def flush():
        nonlocal cur, n_carry
        chunks.append(_make_chunk(cur))
        tail, total = [], 0
        for u in reversed(cur[:-1]):
            if total + len(u["text"]) > overlap:
                break
            tail.insert(0, u)
            total += len(u["text"])
        cur, n_carry = tail, len(tail)

    for block in blocks:
        if block["kind"] == "table" and len(block["text"]) > size:
            if len(cur) > n_carry:
                chunks.append(_make_chunk(cur))
            cur, n_carry = [], 0
            chunks.extend(split_table(block, size))
            continue
        for u in block_to_units(block, size):
            while cur and joined_len(cur, u) > size:
                if len(cur) <= n_carry:
                    cur, n_carry = [], 0
                else:
                    flush()
            cur.append(u)

    new_units = cur[n_carry:]
    if new_units:
        new_len = sum(len(u["text"]) for u in new_units)
        if chunks and new_len < MIN_TAIL_CHARS:
            prev = chunks[-1]
            for u in new_units:
                prev["text"] += u["joiner"] + u["text"]
                if u["pe"]:
                    prev["pe"] = max(prev["pe"] or u["pe"], u["pe"])
        else:
            chunks.append(_make_chunk(cur))
    return chunks


def build_chunks(file_name: str, pages: list, unwrap: bool) -> list:
    blocks = merge_cross_page(parse_blocks(pages, unwrap))
    docs = []
    for path, group in itertools.groupby(blocks, key=lambda b: b["path"]):
        if any(SKIP_SECTION_RE.match(t) for t in path):
            continue
        crumb = " > ".join(path)[:150]
        for c in chunk_section(list(group), CHUNK_SIZE, CHUNK_OVERLAP):
            if is_junk(c["text"]):
                continue
            content = f"Section: {crumb}\n\n{c['text']}" if crumb else c["text"]
            meta = {"source": file_name, "section": crumb}
            if c["ps"]:
                meta["page_number"] = c["ps"]
                meta["page_end"] = c["pe"] or c["ps"]
            docs.append(Document(page_content=content, metadata=meta))
    return docs


def load_and_chunk(file_name: str, data: bytes):
    if file_name.lower().endswith(".pdf"):
        raw_pages = remove_page_furniture(extract_pdf_pages(data))
        pages = list(enumerate(raw_pages, start=1))
        return build_chunks(file_name, pages, unwrap=True), len(pages)
    text = data.decode("utf-8-sig", errors="replace")
    return build_chunks(file_name, [(None, text)], unwrap=False), 0


# ----------------------------------------------------------------------------
# Vector store
# ----------------------------------------------------------------------------
def load_vectorstore():
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return None


def chunk_key(doc) -> str:
    raw = f"{doc.metadata.get('source', '')}|{doc.metadata.get('page_number', '')}|{doc.page_content.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def existing_keys(vectorstore) -> set:
    if vectorstore is None:
        return set()
    return {chunk_key(d) for d in vectorstore.docstore._dict.values()}


def index_summary(vectorstore) -> dict:
    if vectorstore is None:
        return {}
    counts = Counter(d.metadata.get("source", "unknown") for d in vectorstore.docstore._dict.values())
    return dict(sorted(counts.items()))


def has_legacy_chunks(vectorstore) -> bool:
    if vectorstore is None:
        return False
    return any("section" not in d.metadata for d in vectorstore.docstore._dict.values())


def remove_duplicates(vectorstore) -> int:
    if vectorstore is None:
        return 0
    seen, to_delete = set(), []
    for doc_id, d in vectorstore.docstore._dict.items():
        k = chunk_key(d)
        if k in seen:
            to_delete.append(doc_id)
        else:
            seen.add(k)
    if to_delete:
        vectorstore.delete(to_delete)
        vectorstore.save_local(INDEX_DIR)
    return len(to_delete)


def ingest_files(files, vectorstore, progress_cb=None):
    registry = load_registry()
    known = existing_keys(vectorstore)
    report = []
    total = len(files)

    for idx, (name, data) in enumerate(files):
        if progress_cb:
            progress_cb(idx, total, name)

        h = bytes_hash(data)
        if h in registry:
            report.append((name, "skipped (already ingested)"))
            continue

        try:
            chunks, n_pages = load_and_chunk(name, data)
            if not chunks:
                report.append((name, "skipped (no usable text; scanned PDF?)"))
                continue

            fresh = []
            for c in chunks:
                k = chunk_key(c)
                if k not in known:
                    known.add(k)
                    fresh.append(c)

            if fresh:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(fresh, get_embeddings())
                else:
                    vectorstore.add_documents(fresh)
                vectorstore.save_local(INDEX_DIR)

            registry[h] = name
            save_registry(registry)

            if not fresh:
                report.append((name, "skipped (content already in index)"))
            else:
                page_note = f", {n_pages} pages" if n_pages else ""
                dup = len(chunks) - len(fresh)
                dup_note = f", {dup} duplicate chunks skipped" if dup else ""
                report.append((name, f"ingested ({len(fresh)} chunks{page_note}{dup_note})"))
        except Exception as e:
            report.append((name, f"failed: {e}"))

    if progress_cb:
        progress_cb(total, total, "")
    return vectorstore, report


def read_folder(folder: str, recursive: bool):
    files = []
    if recursive:
        walker = os.walk(folder)
    else:
        walker = [(folder, [], [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])]
    for root, _, names in walker:
        for n in sorted(names):
            if n.lower().endswith(SUPPORTED_EXT):
                full = os.path.join(root, n)
                with open(full, "rb") as f:
                    files.append((os.path.relpath(full, folder), f.read()))
    return files


def reset_index():
    if os.path.isdir(INDEX_DIR):
        for f in os.listdir(INDEX_DIR):
            os.remove(os.path.join(INDEX_DIR, f))


# ----------------------------------------------------------------------------
# Hybrid Search & Reranking Pipeline
# ----------------------------------------------------------------------------
def hybrid_retrieve_and_rerank(question: str, vectorstore, k: int = TOP_K):
    if vectorstore is None:
        return []

    all_docs = list(vectorstore.docstore._dict.values())
    if not all_docs:
        return []

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 15

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    combined_docs = ensemble_retriever.invoke(question)

    seen_content = set()
    unique_docs = []
    for d in combined_docs:
        h = hashlib.md5(d.page_content.encode("utf-8")).hexdigest()
        if h not in seen_content:
            seen_content.add(h)
            unique_docs.append(d)

    if not unique_docs:
        return []

    reranker = get_reranker()
    pairs = [(question, doc.page_content) for doc in unique_docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:k]]
    return top_docs


# ----------------------------------------------------------------------------
# RAG Generation
# ----------------------------------------------------------------------------
def format_docs(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        blocks.append(f"[{i}] (Source: {cite_label(d.metadata)})\n{d.page_content}")
    return "\n\n".join(blocks)


def generate_response(question: str, vectorstore, llm, k: int):
    if vectorstore is None:
        return {"answer": NO_ANSWER, "sources": [], "docs": [], "cited": False}

    docs = hybrid_retrieve_and_rerank(question, vectorstore, k=k)

    if not docs:
        return {"answer": NO_ANSWER, "sources": [], "docs": [], "cited": False}

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).partial(no_answer=NO_ANSWER)
    chain = prompt | llm | StrOutputParser()

    context = format_docs(docs)
    raw = chain.invoke({"context": context, "question": question}).strip()

    if not docs or NO_ANSWER.lower() in raw.lower():
        return {"answer": NO_ANSWER, "sources": [], "docs": docs, "cited": False}

    answer, used = resolve_citations(raw, docs)
    cited = bool(used)
    if not used:
        used = list(range(1, len(docs) + 1))

    sources = []
    for n in used:
        meta = docs[n - 1].metadata
        sources.append(
            {
                "n": n,
                "file": meta.get("source", "unknown"),
                "page": page_label(meta),
                "label": cite_label(meta),
                "text": docs[n - 1].page_content,
            }
        )
    return {"answer": answer, "sources": sources, "docs": docs, "cited": cited}


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Hybrid Search & Rerank RAG", page_icon="📚", layout="wide")
st.title("📚 Multi-document Q&A with Hybrid Search & Reranking")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initial_checked" not in st.session_state:
    st.session_state.initial_checked = False

if not st.session_state.initial_checked:
    st.session_state.initial_checked = True
    if st.session_state.vectorstore is None and os.path.exists(INITIAL_DATA_FILE):
        with open(INITIAL_DATA_FILE, "rb") as f:
            initial_bytes = f.read()
        with st.spinner(f"First run: building index from {INITIAL_DATA_FILE}..."):
            st.session_state.vectorstore, _ = ingest_files([(INITIAL_DATA_FILE, initial_bytes)], None)


def run_ingestion(files):
    bar = st.progress(0.0, text="Starting...")

    def cb(i, total, name):
        done = i / total if total else 1.0
        text = f"Processing {name} ({i + 1}/{total})" if name else "Done"
        bar.progress(min(done, 1.0), text=text)

    st.session_state.vectorstore, report = ingest_files(files, st.session_state.vectorstore, cb)
    bar.empty()
    for name, status in report:
        if status.startswith("ingested"):
            st.success(f"{name}: {status}")
        elif status.startswith("failed"):
            st.error(f"{name}: {status}")
        else:
            st.info(f"{name}: {status}")


# ---- Sidebar ----
with st.sidebar:
    st.header("📤 Add documents")
    uploaded = st.file_uploader(
        "Select PDF / TXT / MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Process documents", type="primary", disabled=not uploaded):
        run_ingestion([(u.name, u.getvalue()) for u in uploaded])

    with st.expander("Or ingest a folder from this machine"):
        folder_path = st.text_input("Folder path", placeholder="/home/me/reports")
        recursive = st.checkbox("Include subfolders", value=True)
        if st.button("Ingest folder", disabled=not folder_path.strip()):
            if not os.path.isdir(folder_path.strip()):
                st.error("That folder doesn't exist.")
            else:
                folder_files = read_folder(folder_path.strip(), recursive)
                if not folder_files:
                    st.warning("No PDF/TXT/MD files found in that folder.")
                else:
                    run_ingestion(folder_files)

    st.divider()
    st.header("🗂️ Indexed documents")
    summary = index_summary(st.session_state.vectorstore)
    if summary:
        for name, n_chunks in summary.items():
            st.write(f"• {name} ({n_chunks} chunks)")
        st.caption(f"{len(summary)} files, {sum(summary.values())} chunks indexed")
        if has_legacy_chunks(st.session_state.vectorstore):
            st.warning("Some chunks have no section info. Click **Reset index** and re-upload.")
    else:
        st.caption("Nothing indexed yet.")

    if st.button("Remove duplicate chunks", disabled=not summary):
        removed = remove_duplicates(st.session_state.vectorstore)
        if removed:
            st.success(f"Removed {removed} duplicate chunks.")
        else:
            st.info("No duplicates found.")

    if st.button("Reset index"):
        reset_index()
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def render_sources(msg: dict):
    sources, docs = msg.get("sources", []), msg.get("docs", [])

    if sources:
        title = "Sources" if msg.get("cited") else "Sources (retrieved)"
        st.markdown(f"**📎 {title}**")
        seen = set()
        for s in sources:
            if s["label"] in seen:
                continue
            seen.add(s["label"])
            word = "pages" if "-" in s["page"] else "page"
            page = f", {word} {s['page']}" if s["page"] else ""
            st.markdown(f"- 📄 **{s['file']}**{page}")

    if docs:
        cited_numbers = {s["n"] for s in sources} if msg.get("cited") else set()
        with st.expander(f"🔍 Retrieved & Reranked chunks ({len(docs)})"):
            for i, d in enumerate(docs, start=1):
                mark = " ✅ used in answer" if i in cited_numbers else ""
                st.markdown(f"**[{i}]** `{cite_label(d.metadata)}`{mark}")
                st.text(d.page_content)
                if i < len(docs):
                    st.divider()


# ---- Chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg)

# ---- Chat input ----
if st.session_state.vectorstore is None:
    st.info("No documents indexed yet. Add files from the sidebar to get started.")

if not GEMINI_ENABLED:
    st.error("GOOGLE_API_KEY (or GEMINI_API_KEY) not found in your .env file.")

question = st.chat_input("Ask a question about your documents...", disabled=not GEMINI_ENABLED)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Running hybrid search, reranking, and generating answer..."):
                llm = get_llm(GEMINI_MODEL, TEMPERATURE)
                out = generate_response(question, st.session_state.vectorstore, llm, TOP_K)
            st.markdown(out["answer"])
            msg = {"role": "assistant", "content": out["answer"], **{k: out[k] for k in ("sources", "docs", "cited")}}
            render_sources(msg)
            st.session_state.messages.append(msg)
        except Exception as e:
            st.error(f"Could not generate a response: {e}")