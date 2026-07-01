"""
api.py  –  FastAPI backend  (OFFLINE-SAFE + ASCII-FREE + ACCURATE IMAGE RETRIEVAL)
Run:  uvicorn api:app --reload --port 8000
"""

# ── OFFLINE flags — must come BEFORE all library imports ─────────────────────
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_HUB_OFFLINE"]         = "1"

import re, json, traceback, asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings; warnings.filterwarnings("ignore")

import fitz, torch, faiss
import numpy as np
from PIL import Image
from pathlib import Path

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import CrossEncoder
import ollama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from table_reader import load_table_from_pdf, answer_marks_query, is_marks_query

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "data"
IMAGE_PATH     = "extracted_images"
REPORTS_PATH   = "reports"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CLIP_MODEL_ID  = "openai/clip-vit-base-patch32"
OLLAMA_MODEL   = "llama3.2:3b"
FAISS_TOP_K    = 20
RERANK_TOP_N   = 4
IMAGE_TOP_K    = 3

for d in (IMAGE_PATH, DATA_PATH, REPORTS_PATH):
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="Control Systems + Marks AI Expert")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/images",  StaticFiles(directory=IMAGE_PATH),  name="images")
app.mount("/reports", StaticFiles(directory=REPORTS_PATH), name="reports")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query:      str
    session_id: str = "default"

class Source(BaseModel):
    page:    int
    file:    str
    snippet: str
    score:   float = 0.0

class AskResponse(BaseModel):
    answer:      str
    answer_type: str
    images:      list[str]
    sources:     list[Source]
    report_path: str = ""

class QuestionRequest(BaseModel):
    topic:      str
    marks:      int
    count:      int = 5
    session_id: str = "default"

class QuestionItem(BaseModel):
    number:   int
    question: str
    answer:   str

class QuestionResponse(BaseModel):
    topic:     str
    marks:     int
    questions: list[QuestionItem]

class StatusResponse(BaseModel):
    status:          str
    pdf_count:       int
    image_count:     int
    ollama_ready:    bool
    students_loaded: int


# ── Global state ──────────────────────────────────────────────────────────────
_bot      = None
_sessions: dict[str, list] = {}
_df       = None


# ═════════════════════════════════════════════════════════════════════════════
# ASCII ART STRIPPER
# ═════════════════════════════════════════════════════════════════════════════
def strip_ascii_art(text: str) -> str:
    """Remove ASCII block diagrams and code-fenced art from LLM output."""
    # Remove all fenced code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", "", text, flags=re.MULTILINE)

    lines = text.split("\n")
    clean = []
    ART   = set("|+/<>^v─━═┼┬┴┤├┐┘└┌")

    for line in lines:
        s = line.strip()
        if not s:
            clean.append("")
            continue
        art_count = sum(1 for c in s if c in ART)
        # Also count long runs of dashes as art (e.g. "------")
        dash_run  = len(re.findall(r"-{4,}", s)) > 0
        ratio     = art_count / max(len(s), 1)
        is_bullet = s[0] in ("-", "•", "*", "#") and ratio < 0.5
        if (ratio > 0.25 or dash_run) and not is_bullet:
            continue
        clean.append(line)

    result = re.sub(r"\n{3,}", "\n\n", "\n".join(clean))
    return result.strip()


# ═════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION
# ═════════════════════════════════════════════════════════════════════════════
DIAGRAM_WORDS = {
    "diagram", "draw", "show", "image", "figure", "block", "illustrate",
    "sketch", "picture", "display", "visualize", "graph", "plot",
    "chart", "schematic", "depict",
}
SOURCE_WORDS = {
    "source", "sources", "reference", "references", "page", "pages",
    "where", "citation", "cite", "from which", "which pdf",
}
MARKS_PATTERN = re.compile(r"\b(2|5|10)\s*marks?\b", re.IGNORECASE)


def detect_intent(query: str) -> dict:
    q = query.lower()
    m = MARKS_PATTERN.search(query)
    return {
        "wants_diagram": any(w in q for w in DIAGRAM_WORDS),
        "wants_sources": any(w in q for w in SOURCE_WORDS),
        "marks":         int(m.group(1)) if m else None,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═════════════════════════════════════════════════════════════════════════════
_NO_ASCII = (
    "\n\n[STRICT RULES — NO EXCEPTIONS]\n"
    "- NEVER draw ASCII art, box diagrams, pipe-character tables, or any text graphics.\n"
    "- NEVER use ``` code blocks of any kind.\n"
    "- If you want to show a diagram, instead write 2 plain English sentences describing it.\n"
    "- Use ONLY the provided Context for technical content.\n"
    "- Violating these rules will cause a system error."
)

SYSTEM_NORMAL = (
    "You are a knowledgeable and friendly AI Assistant specialising in "
    "Control Systems Engineering.\n"
    "- For casual conversation, respond warmly.\n"
    "- For technical questions, use ONLY the provided Context.\n"
    "- Keep answers clear and concise."
    + _NO_ASCII
)


def system_structured(marks: int) -> str:
    if marks == 2:
        depth = "very concise: 1-2 bullet points, one short conclusion"
    elif marks == 5:
        depth = "moderate: 3-4 bullet points, 2-sentence conclusion"
    else:
        depth = "detailed: 5-6 bullet points with elaboration, 3-sentence conclusion"
    return (
        f"You are an AI Assistant specialising in Control Systems Engineering.\n"
        f"The user wants a {marks}-mark exam answer. Required depth: {depth}.\n\n"
        f"YOU MUST OUTPUT EXACTLY THIS FORMAT AND NOTHING ELSE:\n\n"
        f"**Definition:**\n"
        f"[One clear sentence defining the concept]\n\n"
        f"**Key Points:**\n"
        f"- [Point 1]\n"
        f"- [Point 2]\n"
        f"(add more based on required depth)\n\n"
        f"**Conclusion:**\n"
        f"[Closing sentence(s)]"
        + _NO_ASCII
    )


def system_structured_with_diagram(marks: int) -> str:
    return system_structured(marks) + (
        "\n\n[DIAGRAM NOTE] The user also asked for a diagram. "
        "Relevant PDF images are shown separately in the UI. "
        "Do NOT describe or recreate the diagram — just follow the format above."
    )


def system_with_sources(marks: int | None) -> str:
    """Use when 'with sources' is requested, with or without marks."""
    base = system_structured(marks) if marks else SYSTEM_NORMAL
    return base + (
        "\n\n[SOURCES NOTE] At the end of your answer, do NOT list sources yourself. "
        "Sources are displayed automatically from the retrieved context."
    )


SYSTEM_DIAGRAM_ONLY = (
    "You are an AI Assistant specialising in Control Systems Engineering.\n"
    "The user asked for a diagram. Relevant images from the PDFs are shown in the UI.\n"
    "Write 2-3 plain prose sentences describing the concept — not the diagram itself.\n"
    "Use ONLY the provided Context."
    + _NO_ASCII
)


def question_gen_prompt(topic: str, marks: int, count: int) -> str:
    if marks == 2:
        depth = "1-2 key points, brief conclusion"
    elif marks == 5:
        depth = "3-4 key points, 2-sentence conclusion"
    else:
        depth = "5-6 key points with elaboration, 3-sentence conclusion"
    return (
        f"You are an expert Control Systems exam paper setter.\n"
        f"Generate exactly {count} exam questions on: \"{topic}\" ({marks} marks each).\n"
        f"Required answer depth: {depth}.\n\n"
        f"Return ONLY valid JSON — no markdown, no code fences, no extra text:\n"
        f'{{"questions":['
        f'{{"number":1,"question":"...","answer":{{"definition":"...","key_points":["..."],"conclusion":"..."}}}}'
        f"]}}\n\n"
        f"RULES:\n"
        f"- Use ONLY provided Context.\n"
        f"- Vary difficulty across questions.\n"
        f"- No ASCII art, no code blocks in answers.\n"
        f"- key_points must be a JSON array of strings."
    )


# ═════════════════════════════════════════════════════════════════════════════
# IMAGE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
def extract_images_from_pdfs() -> list:
    paths = []
    for fname in os.listdir(DATA_PATH):
        if not fname.lower().endswith(".pdf") or "student_marks" in fname.lower():
            continue
        doc = fitz.open(os.path.join(DATA_PATH, fname))
        for pnum, page in enumerate(doc):
            for iidx, info in enumerate(page.get_images(full=True)):
                base  = doc.extract_image(info[0])
                sname = f"{fname}_p{pnum}_i{iidx}.png"
                spath = os.path.join(IMAGE_PATH, sname)
                with open(spath, "wb") as fh:
                    fh.write(base["image"])
                paths.append(spath)
    print(f"  ✓  {len(paths)} image(s) extracted.")
    return paths


# ═════════════════════════════════════════════════════════════════════════════
# INDEX BUILDER
# ═════════════════════════════════════════════════════════════════════════════
def build_indexes():
    # ── Text ─────────────────────────────────────────────────────────────────
    all_docs = []
    for fname in os.listdir(DATA_PATH):
        if fname.lower().endswith(".pdf") and "student_marks" not in fname.lower():
            loader = PyPDFLoader(os.path.join(DATA_PATH, fname))
            all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks   = splitter.split_documents(all_docs)
    print(f"  ✓  {len(chunks)} text chunks indexed.")

    t_emb = HuggingFaceEmbeddings(
        model_name    = EMBED_MODEL,
        model_kwargs  = {"local_files_only": True},
        encode_kwargs = {"normalize_embeddings": True},
    )
    text_vs = LangFAISS.from_documents(chunks, t_emb) if chunks else None
    print("  ✓  Text FAISS index ready.")

    # ── Reranker ──────────────────────────────────────────────────────────────
    print("  … Loading reranker …")
    reranker = CrossEncoder(RERANKER_MODEL, local_files_only=True)
    print("  ✓  Reranker ready.")

    # ── CLIP ──────────────────────────────────────────────────────────────────
    print("  … Loading CLIP …")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
    clip_model.eval()
    print("  ✓  CLIP ready.")

    # ── Image FAISS (inner-product = cosine on normalised vecs) ───────────────
    img_list = sorted([
        os.path.join(IMAGE_PATH, f)
        for f in os.listdir(IMAGE_PATH)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    valid_paths, valid_embs = [], []
    for p in img_list:
        try:
            pil = Image.open(p).convert("RGB")
            if pil.width < 80 or pil.height < 80:   # skip tiny icons
                continue
            inp = clip_processor(images=pil, return_tensors="pt")
            with torch.no_grad():
                feat = clip_model.get_image_features(**inp)
            vec = feat.pooler_output.detach().cpu().numpy().squeeze()
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            valid_paths.append(p)
            valid_embs.append(vec)
        except Exception:
            continue

    if valid_embs:
        arr       = np.array(valid_embs, dtype="float32")
        img_index = faiss.IndexFlatIP(arr.shape[1])
        img_index.add(arr)
        print(f"  ✓  {len(valid_paths)} diagram images indexed.")
    else:
        img_index = faiss.IndexFlatIP(512)
        print("  ⚠  No images found.")

    return text_vs, reranker, img_index, valid_paths, clip_model, clip_processor


# ═════════════════════════════════════════════════════════════════════════════
# EXPERT CLASS
# ═════════════════════════════════════════════════════════════════════════════
class ControlSystemExpert:
    MAX_HISTORY = 6

    def __init__(self, text_vs, reranker, img_idx, img_list, clip_model, clip_processor):
        self.text_vs        = text_vs
        self.reranker       = reranker
        self.img_idx        = img_idx
        self.img_list       = img_list
        self.clip_model     = clip_model
        self.clip_processor = clip_processor

    def _retrieve(self, query: str, with_sources: bool = False):
        if self.text_vs is None:
            return "", []
        raw_docs = self.text_vs.similarity_search(query, k=FAISS_TOP_K)
        pairs    = [(query, d.page_content) for d in raw_docs]
        scores   = self.reranker.predict(pairs)
        ranked   = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)
        top_docs = ranked[:RERANK_TOP_N]
        context  = "\n\n".join(d.page_content for _, d in top_docs)

        if not with_sources:
            return context, []

        sources, seen = [], set()
        for score, doc in top_docs:
            meta    = doc.metadata or {}
            page    = int(meta.get("page", 0)) + 1
            file    = Path(meta.get("source", "unknown.pdf")).name
            snippet = doc.page_content[:140].replace("\n", " ").strip()
            key     = (page, file)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "page": page, "file": file,
                    "snippet": snippet, "score": round(float(score), 3),
                })
        return context, sources

    def _retrieve_images(self, query: str) -> list:
        if not self.img_list:
            return []
        inp = self.clip_processor(
            text=[query], return_tensors="pt", truncation=True, max_length=77
        )
        with torch.no_grad():
            feat = self.clip_model.get_text_features(**inp)
        vec = feat.pooler_output.detach().cpu().numpy().astype("float32")
        vec = vec / (np.linalg.norm(vec) + 1e-9)

        k             = min(IMAGE_TOP_K * 4, len(self.img_list))
        scores_i, ids = self.img_idx.search(vec, k)
        clip_hits     = [
            (float(scores_i[0][i]), self.img_list[ids[0][i]])
            for i in range(k) if ids[0][i] >= 0
        ]

        # Keyword boost: filename words overlapping with query words
        q_words = set(re.findall(r"\w+", query.lower()))
        boosted = []
        for score, path in clip_hits:
            fname_words = set(re.findall(r"\w+", Path(path).stem.lower()))
            overlap     = len(q_words & fname_words)
            boosted.append((score + overlap * 0.15, path))

        boosted.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in boosted[:IMAGE_TOP_K]]

    def ask(self, query: str, history: list) -> dict:
        intent        = detect_intent(query)
        wants_diagram = intent["wants_diagram"]
        wants_sources = intent["wants_sources"]
        marks         = intent["marks"]

        context, sources = self._retrieve(query, with_sources=wants_sources)
        img_paths        = self._retrieve_images(query) if wants_diagram else []

        # Select system prompt — all combinations handled
        if marks and wants_diagram and wants_sources:
            sys_p, answer_type = system_structured_with_diagram(marks), "structured"
        elif marks and wants_diagram:
            sys_p, answer_type = system_structured_with_diagram(marks), "structured"
        elif marks and wants_sources:
            sys_p, answer_type = system_with_sources(marks), "structured"
        elif marks:
            sys_p, answer_type = system_structured(marks), "structured"
        elif wants_diagram and wants_sources:
            sys_p, answer_type = SYSTEM_DIAGRAM_ONLY, "normal"
        elif wants_diagram:
            sys_p, answer_type = SYSTEM_DIAGRAM_ONLY, "normal"
        elif wants_sources:
            sys_p, answer_type = system_with_sources(None), "normal"
        else:
            sys_p, answer_type = SYSTEM_NORMAL, "normal"

        user_content = (
            f"[Context — reranked]\n{context}\n\n[Question]\n{query}"
            if context.strip() else query
        )
        messages = (
            [{"role": "system", "content": sys_p}]
            + history[-(self.MAX_HISTORY * 2):]
            + [{"role": "user", "content": user_content}]
        )

        resp   = ollama.chat(
            model=OLLAMA_MODEL, messages=messages,
            options={"temperature": 0.4, "top_p": 0.9, "repeat_penalty": 1.15},
        )
        answer = strip_ascii_art(resp["message"]["content"].strip())

        return {
            "answer":      answer,
            "answer_type": answer_type,
            "images":      [f"/images/{Path(p).name}" for p in img_paths],
            "sources":     sources,
            "report_path": "",
        }

    def generate_questions(self, topic: str, marks: int, count: int) -> list:
        context, _ = self._retrieve(topic)
        sys_msg    = question_gen_prompt(topic, marks, count)
        user_msg   = (
            f"[Context — reranked]\n{context}\n\n"
            f"Generate {count} × {marks}-mark questions on: {topic}"
            if context.strip()
            else f"Generate {count} × {marks}-mark questions on: {topic}"
        )
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": user_msg},
            ],
            options={"temperature": 0.5, "top_p": 0.9},
        )
        raw = resp["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
        raw = re.sub(r"```$",          "", raw, flags=re.MULTILINE).strip()
        m   = re.search(r"\{[\s\S]*\}", raw)
        if m: raw = m.group(0)
        try:
            return json.loads(raw).get("questions", [])
        except json.JSONDecodeError:
            return [{
                "number": 1, "question": f"Topic: {topic}",
                "answer": {"definition": raw[:400], "key_points": [], "conclusion": "Retry."},
            }]


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup_event():
    global _bot, _df
    print("🚀  Loading models …")

    try:
        _df = load_table_from_pdf()
        print(f"  ✓  {len(_df)} students loaded." if _df is not None
              else "  ⚠  Student marks PDF not found.")
    except Exception:
        _df = None
        print("  ⚠  Could not load student marks.")

    try:
        extract_images_from_pdfs()
    except Exception as e:
        print(f"  ⚠  Image extraction error: {e}")

    loop = asyncio.get_event_loop()
    try:
        tv, rr, ii, il, cm, cp = await loop.run_in_executor(
            ThreadPoolExecutor(max_workers=1), build_indexes
        )
        _bot = ControlSystemExpert(tv, rr, ii, il, cm, cp)
        print("✓  Bot ready.")
    except Exception as e:
        print(f"✗  Index build failed: {e}")
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/status", response_model=StatusResponse)
async def status():
    pdfs = len([f for f in os.listdir(DATA_PATH)  if f.endswith(".pdf")])
    imgs = len([f for f in os.listdir(IMAGE_PATH) if f.endswith(".png")])
    ok   = False
    try:
        ok = OLLAMA_MODEL in [m.model for m in ollama.list().models]
    except Exception:
        pass
    return StatusResponse(
        status=("ready" if _bot else "loading"),
        pdf_count=pdfs, image_count=imgs,
        ollama_ready=ok,
        students_loaded=(len(_df) if _df is not None else 0),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not _bot:
        raise HTTPException(503, "Bot still loading — please wait and retry.")

    history = _sessions.setdefault(req.session_id, [])

    exam_marks = bool(re.search(r'\b(2|5|10)\s*marks?\b', req.query, re.IGNORECASE))
    if _df is not None and is_marks_query(req.query) and not exam_marks:
        try:
            answer_text, report_file = answer_marks_query(req.query, _df)
            answer_text = answer_text or "Sorry, I could not find that in the student data."
            report_url  = ""
            if report_file and os.path.exists(report_file):
                report_url = f"/reports/{Path(report_file).name}"
            history += [
                {"role": "user",      "content": req.query},
                {"role": "assistant", "content": answer_text},
            ]
            return AskResponse(
                answer=answer_text, answer_type="marks",
                images=[], sources=[], report_path=report_url,
            )
        except Exception as e:
            traceback.print_exc(); raise HTTPException(500, str(e))

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _bot.ask(req.query, history)
        )
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, str(e))

    history += [
        {"role": "user",      "content": req.query},
        {"role": "assistant", "content": result["answer"]},
    ]
    return AskResponse(
        answer=result["answer"], answer_type=result["answer_type"],
        images=result["images"],
        sources=[Source(**s) for s in result["sources"]],
        report_path="",
    )


@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions(req: QuestionRequest):
    if not _bot:                  raise HTTPException(503, "Still loading.")
    if req.marks not in (2,5,10): raise HTTPException(400, "marks must be 2, 5, or 10.")
    if not 1 <= req.count <= 10:  raise HTTPException(400, "count must be 1-10.")
    try:
        loop   = asyncio.get_event_loop()
        raw_qs = await loop.run_in_executor(
            None, lambda: _bot.generate_questions(req.topic, req.marks, req.count)
        )
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, str(e))

    items = []
    for i, q in enumerate(raw_qs):
        ans = q.get("answer", {})
        if isinstance(ans, dict):
            kp_str    = "\n".join(f"- {p}" for p in ans.get("key_points", []))
            formatted = (
                f"**Definition:**\n{ans.get('definition','')}\n\n"
                f"**Key Points:**\n{kp_str}\n\n"
                f"**Conclusion:**\n{ans.get('conclusion','')}"
            ).strip()
        else:
            formatted = str(ans)
        items.append(QuestionItem(
            number=q.get("number", i+1),
            question=q.get("question", ""),
            answer=formatted,
        ))
    return QuestionResponse(topic=req.topic, marks=req.marks, questions=items)


@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.get("/")
async def root():
    return {"message": "Control Systems + Marks AI Expert — running."}