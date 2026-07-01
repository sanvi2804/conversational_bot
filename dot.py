"""
dot.py  –  CLI version  (OFFLINE-SAFE + ASCII-FREE + ACCURATE IMAGE RETRIEVAL)
Routes:
  - Marks / student queries  → table_reader.py  (no LLM)
  - Control Systems queries  → RAG + Reranker + Ollama
"""

# ── OFFLINE flags — must come BEFORE all library imports ─────────────────────
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_HUB_OFFLINE"]         = "1"

import json, re, subprocess, platform
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

from table_reader import load_table_from_pdf, answer_marks_query, is_marks_query

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "data"
IMAGE_PATH     = "extracted_images"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CLIP_MODEL_ID  = "openai/clip-vit-base-patch32"
OLLAMA_MODEL   = "llama3.2:3b"
FAISS_TOP_K    = 20
RERANK_TOP_N   = 4
IMAGE_TOP_K    = 3

os.makedirs(IMAGE_PATH, exist_ok=True)
os.makedirs(DATA_PATH,  exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# ASCII ART STRIPPER
# ═════════════════════════════════════════════════════════════════════════════
def strip_ascii_art(text: str) -> str:
    """Remove ASCII block diagrams and code-fenced art from LLM output."""
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
        f"Definition:\n"
        f"[One clear sentence defining the concept]\n\n"
        f"Key Points:\n"
        f"- [Point 1]\n"
        f"- [Point 2]\n"
        f"(add more based on required depth)\n\n"
        f"Conclusion:\n"
        f"[Closing sentence(s)]"
        + _NO_ASCII
    )


def system_structured_with_diagram(marks: int) -> str:
    return system_structured(marks) + (
        "\n\n[DIAGRAM NOTE] The user asked for a diagram. "
        "PDF images are opened separately. Do NOT describe or recreate the diagram."
    )


def system_with_sources(marks: int | None) -> str:
    base = system_structured(marks) if marks else SYSTEM_NORMAL
    return base + (
        "\n\n[SOURCES NOTE] Sources are displayed automatically. "
        "Do NOT list sources yourself at the end."
    )


SYSTEM_DIAGRAM_ONLY = (
    "You are an AI Assistant specialising in Control Systems Engineering.\n"
    "The user asked for a diagram. Relevant images are being opened separately.\n"
    "Write 2-3 plain prose sentences describing the concept.\n"
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
        f"- No ASCII art, no code blocks.\n"
        f"- key_points must be a JSON array of strings."
    )


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def show_image(img_path: str) -> None:
    try:
        img = Image.open(img_path)
        w, h = img.size
        print(f"\n  🖼   Opening [{w}×{h}] — {img_path}")
        img.show(title=os.path.basename(img_path))
    except Exception as e:
        print(f"  ⚠  PIL failed ({e}) — trying OS viewer …")
        try:
            s = platform.system()
            if s == "Darwin":    subprocess.Popen(["open",     img_path])
            elif s == "Windows": subprocess.Popen(["start",    img_path], shell=True)
            else:                subprocess.Popen(["xdg-open", img_path])
        except Exception as e2:
            print(f"  ✗  Cannot open: {e2}")


def print_sources(sources: list) -> None:
    if not sources:
        return
    print("\n  📄  Sources (reranked by relevance):")
    for s in sources:
        score_str = f"  score={s.get('score', 0):.3f}" if "score" in s else ""
        print(f"     ├─ {s['file']}  →  Page {s['page']}{score_str}")
        print(f"     │   \"{s['snippet']}…\"")
    print()


def print_questions(questions: list, marks: int, topic: str) -> None:
    sep = "─" * 62
    print(f"\n{'═'*62}")
    print(f"  📝  {len(questions)} × {marks}-mark questions on: {topic}")
    print(f"{'═'*62}")
    for q in questions:
        print(f"\n{sep}")
        print(f"  Q{q['number']}. [{marks} marks]\n  {q['question']}")
        print(sep)
        ans = q.get("answer", {})
        if isinstance(ans, dict):
            print(f"\n  Definition:\n  {ans.get('definition', '')}")
            kps = ans.get("key_points", [])
            if kps:
                print("\n  Key Points:")
                for p in kps:
                    print(f"    • {p}")
            print(f"\n  Conclusion:\n  {ans.get('conclusion', '')}")
        else:
            print(f"\n  {ans}")
    print(f"\n{'═'*62}\n")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — EXTRACT IMAGES
# ═════════════════════════════════════════════════════════════════════════════
def extract_images() -> list:
    print("\n[Step 1] Extracting images from PDFs …")
    paths = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("  ⚠  No PDFs found."); return paths
    for fname in pdf_files:
        if "student_marks" in fname.lower():
            continue
        doc = fitz.open(os.path.join(DATA_PATH, fname))
        for pnum, page in enumerate(doc):
            for iidx, info in enumerate(page.get_images(full=True)):
                base  = doc.extract_image(info[0])
                sname = f"{fname}_p{pnum}_i{iidx}.png"
                spath = os.path.join(IMAGE_PATH, sname)
                with open(spath, "wb") as f:
                    f.write(base["image"])
                paths.append(spath)
    print(f"  ✓  {len(paths)} image(s) extracted.")
    return paths


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD INDEXES
# ═════════════════════════════════════════════════════════════════════════════
def setup_indexes():
    print("\n[Step 2] Building indexes …")

    all_docs = []
    for fname in os.listdir(DATA_PATH):
        if fname.lower().endswith(".pdf") and "student_marks" not in fname.lower():
            loader = PyPDFLoader(os.path.join(DATA_PATH, fname))
            all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks   = splitter.split_documents(all_docs)
    print(f"  ✓  {len(chunks)} chunks from {len(all_docs)} pages.")

    t_emb = HuggingFaceEmbeddings(
        model_name    = EMBED_MODEL,
        model_kwargs  = {"local_files_only": True},
        encode_kwargs = {"normalize_embeddings": True},
    )
    text_vs = LangFAISS.from_documents(chunks, t_emb) if chunks else None
    print("  ✓  Text FAISS index ready.")

    print(f"  … Loading reranker ({RERANKER_MODEL}) …")
    reranker = CrossEncoder(RERANKER_MODEL, local_files_only=True)
    print("  ✓  Reranker ready.")

    print("  … Loading CLIP …")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
    clip_model.eval()
    print("  ✓  CLIP ready.")

    img_list = sorted([
        os.path.join(IMAGE_PATH, f)
        for f in os.listdir(IMAGE_PATH)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    valid_paths, valid_embs = [], []
    for p in img_list:
        try:
            pil = Image.open(p).convert("RGB")
            if pil.width < 80 or pil.height < 80:
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
        print(f"  ✓  Image index ready ({len(valid_paths)} diagrams).")
    else:
        img_index = faiss.IndexFlatIP(512)
        print("  ⚠  No images found.")

    return text_vs, reranker, img_index, valid_paths, clip_model, clip_processor


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — CHECK OLLAMA
# ═════════════════════════════════════════════════════════════════════════════
def check_ollama():
    print(f"\n[Step 3] Checking Ollama ({OLLAMA_MODEL}) …")
    try:
        available = [m.model for m in ollama.list().models]
        if OLLAMA_MODEL not in available:
            print(f"  ⚠  '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
        else:
            print(f"  ✓  '{OLLAMA_MODEL}' ready.")
    except Exception as e:
        print(f"  ✗  Ollama unreachable: {e}\n     Run: ollama serve")
        raise SystemExit(1)


# ═════════════════════════════════════════════════════════════════════════════
# BOT CLASS
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
        self.chat_history   = []

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

        q_words = set(re.findall(r"\w+", query.lower()))
        boosted = []
        for score, path in clip_hits:
            fname_words = set(re.findall(r"\w+", Path(path).stem.lower()))
            overlap     = len(q_words & fname_words)
            boosted.append((score + overlap * 0.15, path))

        boosted.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in boosted[:IMAGE_TOP_K]]

    def ask(self, query: str) -> tuple:
        intent        = detect_intent(query)
        wants_diagram = intent["wants_diagram"]
        wants_sources = intent["wants_sources"]
        marks         = intent["marks"]

        context, sources = self._retrieve(query, with_sources=wants_sources)
        images           = self._retrieve_images(query) if wants_diagram else []

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
            + self.chat_history[-(self.MAX_HISTORY * 2):]
            + [{"role": "user", "content": user_content}]
        )

        resp   = ollama.chat(
            model=OLLAMA_MODEL, messages=messages,
            options={"temperature": 0.4, "top_p": 0.9, "repeat_penalty": 1.15},
        )
        answer = strip_ascii_art(resp["message"]["content"].strip())

        self.chat_history.append({"role": "user",      "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer, images, sources, answer_type

    def generate_questions(self, topic: str, marks: int, count: int) -> list:
        context, _ = self._retrieve(topic)
        sys_msg    = question_gen_prompt(topic, marks, count)
        user_msg   = (
            f"[Context — reranked]\n{context}\n\n"
            f"Generate {count} × {marks}-mark questions on: {topic}"
            if context.strip()
            else f"Generate {count} × {marks}-mark questions on: {topic}"
        )
        print(f"\n  ⏳  Generating {count} × {marks}-mark questions on '{topic}' …")
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

    def clear_history(self):
        self.chat_history.clear()


# ═════════════════════════════════════════════════════════════════════════════
# QUESTION GENERATOR MENU
# ═════════════════════════════════════════════════════════════════════════════
def run_question_generator(bot: ControlSystemExpert) -> None:
    print("\n" + "─"*62)
    print("  📝  QUESTION GENERATOR")
    print("─"*62)
    topic = input("  Topic: ").strip()
    if not topic:
        print("  ⚠  No topic entered.\n"); return
    print("  Mark type:  [1] 2 marks   [2] 5 marks   [3] 10 marks")
    marks = {"1": 2, "2": 5, "3": 10}.get(input("  Choice (1/2/3): ").strip(), 5)
    print("  Count:      [1] 2   [2] 5   [3] 10")
    count = {"1": 2, "2": 5, "3": 10}.get(input("  Choice (1/2/3): ").strip(), 5)
    questions = bot.generate_questions(topic, marks, count)
    print_questions(questions, marks, topic)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    check_ollama()

    print("\n[Step 0] Loading student marks table …")
    try:
        df = load_table_from_pdf()
        if df is not None:
            print(f"  ✓  {len(df)} students loaded from PDF.")
        else:
            print("  ⚠  Student marks PDF not found — marks queries will not work.")
    except Exception:
        df = None
        print("  ⚠  Could not load student marks PDF.")

    extract_images()
    text_vs, reranker, img_idx, img_list, clip_model, clip_processor = setup_indexes()
    bot = ControlSystemExpert(text_vs, reranker, img_idx, img_list, clip_model, clip_processor)

    banner = """
╔══════════════════════════════════════════════════════════════╗
║  AI EXPERT  (Control Systems + Student Marks)               ║
╠══════════════════════════════════════════════════════════════╣
║  CONTROL SYSTEMS QUERIES                                     ║
║    explain X                         → normal answer         ║
║    explain X with diagram            → text + image          ║
║    explain X for 5 marks             → structured answer     ║
║    explain X for 5 marks with source → answer + PDF pages    ║
║    explain X for 5 marks with diagram → structured + image   ║
║                                                              ║
║  STUDENT MARKS QUERIES                                       ║
║    marks of <name>  /  top 5 in physics                      ║
║    analyse performance of <name>                             ║
║    generate report for <name>                                ║
║                                                              ║
║  COMMANDS: questions · images · clear · exit                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    imgs     = [f for f in os.listdir(IMAGE_PATH) if f.endswith(".png")]
    students = len(df) if df is not None else 0
    print(f"  📂  {len(imgs)} images  |  👥  {students} students loaded\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in {"exit", "quit"}:
            print("Goodbye! 🎛️"); break

        if cmd == "clear":
            bot.clear_history()
            print("  ✓  History cleared.\n"); continue

        if cmd == "questions":
            run_question_generator(bot); continue

        if cmd == "images":
            imgs = sorted(os.listdir(IMAGE_PATH))
            print(f"\n  📂  {len(imgs)} image(s):")
            for n in imgs: print(f"       {n}")
            print(); continue

        # ── Route: marks query ────────────────────────────────────────────────
        exam_marks = bool(re.search(r'\b(2|5|10)\s*marks?\b', user_input, re.IGNORECASE))
        if df is not None and is_marks_query(user_input) and not exam_marks:
            try:
                answer_text, report_file = answer_marks_query(user_input, df)
                if answer_text:
                    print(f"\nBot:\n{answer_text}\n")
                if report_file and os.path.exists(report_file):
                    print(f"  📄  Report saved: {report_file}")
                    try:
                        s = platform.system()
                        if s == "Darwin":    subprocess.Popen(["open",     report_file])
                        elif s == "Windows": subprocess.Popen(["start",    report_file], shell=True)
                        else:                subprocess.Popen(["xdg-open", report_file])
                    except Exception:
                        pass
            except Exception as e:
                print(f"  ✗  Marks query error: {e}")
            continue

        # ── Route: control systems ────────────────────────────────────────────
        answer, images, sources, _ = bot.ask(user_input)
        print(f"\nBot:\n{answer}\n")
        print_sources(sources)

        if images:
            print(f"  📊  {len(images)} diagram(s) — opening …")
            for p in images: show_image(p)
            print()
        elif detect_intent(user_input)["wants_diagram"] and not bot.img_list:
            print(f"  ⚠  No images extracted. Add PDFs to '{DATA_PATH}/' and restart.\n")


if __name__ == "__main__":
    main()