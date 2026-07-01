"""
table_reader.py
─────────────────────────────────────────────────────
Loads student marks from PDF, answers queries, generates
Performance Intelligence and Personalised Recommendations.

NOTE: This module is 100% pure Python (pandas + math + reportlab).
      No LLM is used here at all — works with any model setup.

Returns: answer_marks_query(query, df) → (str, str|None)
         str      = text answer
         str|None = PDF report path (only for report queries)
─────────────────────────────────────────────────────
"""

import pdfplumber
import pandas as pd
import numpy as np
import re
import os
import math

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER

# ─────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────
TABLE_PDF_PATH  = "data/student marks copy.pdf"
REPORT_OUT_PATH = "reports"
os.makedirs(REPORT_OUT_PATH, exist_ok=True)

ALL_SUBJECTS = ["biology", "physics", "chemistry"]
MAX_MARKS    = {"biology": 60, "physics": 60, "chemistry": 60, "total": 180}

SUBJECT_MAP = {
    "biology":   "biology",  "bio":  "biology",
    "physics":   "physics",  "phy":  "physics",
    "chemistry": "chemistry","chem": "chemistry",
    "maths": "total", "math": "total", "total": "total",
}

STUDY_TIPS = {
    "biology": [
        "Focus on NCERT diagrams — KCET asks directly from them.",
        "Revise classification, genetics, and ecology chapters thoroughly.",
        "Make short notes on plant physiology and human physiology.",
        "Practice labelling diagrams daily — easy scoring area.",
        "Revise Biomolecules and Cell Biology — high-weightage topics.",
    ],
    "physics": [
        "Practice numerical problems daily — at least 10 per day.",
        "Focus on Electrostatics, Current Electricity, and Optics.",
        "Memorise all formulae in a formula sheet and revise every morning.",
        "Understand derivations — KCET asks conceptual numericals.",
        "Practice previous year papers to improve speed and accuracy.",
    ],
    "chemistry": [
        "Revise organic reaction mechanisms thoroughly.",
        "Make a table of all named reactions with reagents and products.",
        "Focus on p-Block and d-Block elements — high scoring.",
        "Practice Mole Concept and Equilibrium numericals every day.",
        "Revise NCERT inorganic chemistry tables completely.",
    ],
}

MARKS_KEYWORDS = [
    "marks", "score", "scored", "highest", "lowest",
    "second highest", "2nd highest", "top", "rank",
    "average", "mean", "avg", "student", "pass", "fail",
    "biology", "bio", "physics", "phy", "chemistry", "chem", "total",
    "percent", "percentage", "topper",
    "who got", "who scored", "who is",
    "maths", "math", "best rank", "highest rank",
    "highest percent", "highest percentage",
    "best percent", "best percentage",
    "admission", "adm",
    "report", "generate report", "performance", "analysis",
    "consistent", "consistency", "strength", "weakness",
    "predict", "suggestion", "improve", "trend",
    "deviation", "variance", "intelligence", "insight",
    "analyse", "analyze", "recommend", "recommendation",
    "mentor", "target", "goal", "reach", "achieve",
    "how to", "how can", "what should", "advice",
]

# ── Report cache: adm_no → filepath ──────────────────────────────────────────
# Avoids regenerating the same PDF every time it is requested.
_report_cache: dict = {}


# ─────────────────────────────────────────────────────
# LOAD TABLE
# ─────────────────────────────────────────────────────
def load_table_from_pdf():
    try:
        all_students = []
        with pdfplumber.open(TABLE_PDF_PATH) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    for row in table:
                        if not row or not any(row):
                            continue
                        first = str(row[0]).strip() if row[0] else ""
                        if not re.match(r'^\d{6,7}$', first):
                            continue

                        def safe(v):
                            try:
                                return float(v) if v not in (None, '', 'None') else None
                            except Exception:
                                return None

                        r = list(row) + [None] * 20
                        s = {
                            "adm_no":            str(r[0]).strip(),
                            "name":              str(r[1]).strip() if r[1] else "",
                            "section":           str(r[2]).strip() if r[2] else "",
                            "total":             safe(r[3]),
                            "total_rank":        safe(r[4]),
                            "total_percent":     safe(r[5]),
                            "biology":           safe(r[6]),
                            "biology_rank":      safe(r[7]),
                            "biology_percent":   safe(r[8]),
                            "physics":           safe(r[9]),
                            "physics_rank":      safe(r[10]),
                            "physics_percent":   safe(r[11]),
                            "chemistry":         safe(r[12]),
                            "chemistry_rank":    safe(r[13]),
                            "chemistry_percent": safe(r[14]),
                        }
                        if not s["name"] or s["name"] == "None":
                            continue
                        all_students.append(s)

        if not all_students:
            print("WARNING: No student data found in PDF.")
            return None

        df = pd.DataFrame(all_students).reset_index(drop=True)
        print(f"[Table] Loaded {len(df)} students.")
        return df

    except Exception as e:
        print(f"Table read error: {e}")
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────
# QUERY DETECTION HELPERS
# ─────────────────────────────────────────────────────
def is_marks_query(query):
    q = query.lower()
    return any(k in q for k in MARKS_KEYWORDS)

def is_report_query(query):
    q = query.lower()
    return any(k in q for k in [
        "report", "generate report", "full report",
        "performance report", "export", "pdf report", "pdf",
    ])

def is_intelligence_query(query):
    q = query.lower()
    return any(k in q for k in [
        "performance", "analysis", "consistent", "consistency",
        "strength", "weakness", "predict", "deviation",
        "analyze", "analyse", "intelligence", "insight", "trend",
    ])

def is_recommendation_query(query):
    q = query.lower()
    return any(k in q for k in [
        "recommend", "recommendation", "mentor", "improve",
        "suggestion", "suggest", "advice", "target", "goal",
        "reach", "achieve", "how to", "how can",
        "what should", "help me", "guide",
    ])

def detect_all_subjects(query):
    q     = query.lower()
    found = []
    for kw, col in SUBJECT_MAP.items():
        if kw in q and col not in found:
            found.append(col)
    return found

def detect_subject(query):
    f = detect_all_subjects(query)
    return f[0] if f else None

def find_student_by_name(query, df):
    q       = query.lower()
    q_words = [w for w in re.split(r'\W+', q) if len(w) >= 3]

    for _, row in df.iterrows():
        if str(row["name"]).lower() in q:
            return row
    for _, row in df.iterrows():
        name_words = [w for w in str(row["name"]).lower().split() if len(w) >= 3]
        if name_words and all(w in q for w in name_words):
            return row
    for _, row in df.iterrows():
        name_words = [w for w in str(row["name"]).lower().split() if len(w) >= 4]
        if any(nw in q_words for nw in name_words):
            return row
    return None

def find_student_by_adm(query, df):
    for num in re.findall(r'\b\d{6,7}\b', query):
        matched = df[df["adm_no"] == num]
        if len(matched) > 0:
            return matched.iloc[0].to_dict()
    return None

def fmt_rank(val):
    return int(val) if pd.notna(val) else "N/A"

def fmt_pct(val):
    return val if pd.notna(val) else "N/A"

def detect_wants(q):
    wants_marks   = any(k in q for k in ["marks", "score", "scored"])
    wants_rank    = "rank" in q
    wants_percent = any(k in q for k in ["percent", "percentage", "%"])
    wants_average = any(k in q for k in ["average", "mean", "avg"])
    if not any([wants_marks, wants_rank, wants_percent, wants_average]):
        wants_marks = True
    return wants_marks, wants_rank, wants_percent, wants_average

def student_subject_block(name, row, subject, wants_marks, wants_rank, wants_percent):
    label = "Total" if subject == "total" else subject.title()
    lines = [f"{name} — {label}:"]
    if wants_marks:   lines.append(f"  Marks:      {row[subject]}")
    if wants_rank:    lines.append(f"  Rank:       {fmt_rank(row[f'{subject}_rank'])}")
    if wants_percent: lines.append(f"  Percentage: {fmt_pct(row[f'{subject}_percent'])}%")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# PERFORMANCE INTELLIGENCE
# ─────────────────────────────────────────────────────
def get_performance_intelligence(row, df):
    name   = row["name"]
    scores = {s: row[s] for s in ALL_SUBJECTS if pd.notna(row.get(s))}
    if not scores:
        return f"Not enough score data to analyse {name}."

    vals      = list(scores.values())
    n         = len(vals)
    avg_score = round(sum(vals) / n, 2)
    variance  = sum((v - avg_score) ** 2 for v in vals) / n
    std_score = round(math.sqrt(variance), 2)

    if std_score <= 5:
        consistency, cons_emoji, cons_note = (
            "Very Consistent", "✅", "performs evenly across all subjects")
    elif std_score <= 10:
        consistency, cons_emoji, cons_note = (
            "Moderately Consistent", "⚠️",
            "noticeable gap between subjects — balance needed")
    else:
        consistency, cons_emoji, cons_note = (
            "Inconsistent", "❌",
            f"large spread ({std_score} pts std dev) — urgent rebalancing needed")

    strongest = max(scores, key=scores.get)
    weakest   = min(scores, key=scores.get)

    above_avg_lines, below_avg_lines = [], []
    for s in ALL_SUBJECTS:
        if pd.notna(row.get(s)):
            cls_avg = df[s].mean()
            diff    = round(row[s] - cls_avg, 1)
            if diff >= 0:
                above_avg_lines.append(
                    f"    ✓ {s.title()}: {row[s]} marks  (+{diff} above class avg {round(cls_avg,1)})")
            else:
                below_avg_lines.append(
                    f"    ✗ {s.title()}: {row[s]} marks  ({abs(diff)} below class avg {round(cls_avg,1)})")

    total_students = len(df)
    rank = fmt_rank(row.get("total_rank"))
    if rank != "N/A":
        rank_pct  = round((1 - rank / total_students) * 100, 1)
        rank_note = f"better than {rank_pct}% of the class"
    else:
        rank_note = "rank not available"

    total_pct = row.get("total_percent") or 0
    if total_pct >= 80:   trend = "🚀 Excellent — in the top tier of the class"
    elif total_pct >= 65: trend = "📈 Good — above average; push into top tier"
    elif total_pct >= 50: trend = "📊 Average — consistent effort needed"
    elif total_pct >= 35: trend = "📉 Below average — needs structured intervention"
    else:                 trend = "🔴 Critical — immediate intensive study plan required"

    if total_pct >= 70:
        prediction = "On track for a top rank in the final exam."
    elif total_pct >= 50:
        prediction = ("Satisfactory foundation. "
                      "Focused effort on weaker subjects can improve rank significantly.")
    elif total_pct >= 35:
        prediction = "At risk in weaker subjects. A structured daily plan is essential now."
    else:
        prediction = ("High risk across subjects. "
                      "Immediate intervention with teacher guidance is strongly recommended.")

    topper_total = df["total"].max()
    gap_to_top   = round(topper_total - (row.get("total") or 0), 1)
    gap_note     = (f"  Gap to class topper: {gap_to_top} marks "
                    f"({round(gap_to_top/180*100,1)}% of total)")

    sep   = "─" * 48
    lines = [
        f"📊 Performance Intelligence — {name}",
        sep,
        f"  Avg score (Bio+Phy+Chem) : {avg_score} / {(MAX_MARKS['biology']+MAX_MARKS['physics']+MAX_MARKS['chemistry'])//3}",
        f"  Std deviation            : {std_score}  → {cons_emoji} {consistency}",
        f"  ({cons_note})",
        f"  Strongest subject        : {strongest.title()}  ({scores[strongest]} marks)",
        f"  Weakest subject          : {weakest.title()}  ({scores[weakest]} marks)",
        f"  Overall rank             : {rank} / {total_students}  ({rank_note})",
        gap_note,
        "",
        "  Subject vs class average:",
    ]
    lines += above_avg_lines or ["    None above average"]
    lines += below_avg_lines or ["    None below average"]
    lines += [
        "",
        f"  Trend   : {trend}",
        f"  Verdict : {prediction}",
        sep,
        "  💡 One-line summary:",
        (f"  {name} is {consistency.lower()} "
         f"({'performing well' if total_pct >= 50 else 'struggling'}) — "
         f"{strongest.title()} is the strongest subject and "
         f"{weakest.title()} needs the most attention."),
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# PERSONALISED RECOMMENDATIONS
# ─────────────────────────────────────────────────────
def get_personalized_recommendations(row, df, target_pct=None):
    name      = row["name"]
    total_pct = row.get("total_percent") or 0
    scores    = {s: row[s] for s in ALL_SUBJECTS if pd.notna(row.get(s))}
    if not scores:
        return f"Not enough data to generate recommendations for {name}."

    if target_pct is None:
        if total_pct >= 90:   target_pct = 95.0
        elif total_pct >= 75: target_pct = 90.0
        elif total_pct >= 60: target_pct = 75.0
        elif total_pct >= 45: target_pct = 60.0
        else:                 target_pct = 50.0

    toppers = {s: {"top": df[s].max(),
                   "gap": round(df[s].max() - (row.get(s) or 0), 1)}
               for s in ALL_SUBJECTS}

    total_max     = sum(MAX_MARKS[s] for s in ALL_SUBJECTS)
    target_total  = round(target_pct / 100 * total_max, 1)
    current_total = sum(scores.get(s, 0) for s in ALL_SUBJECTS)
    total_gap     = round(target_total - current_total, 1)

    vals    = [scores.get(s, 0) for s in ALL_SUBJECTS]
    avg_v   = sum(vals) / len(vals) if vals else 1
    weights = {s: max(0.01, avg_v - scores.get(s, 0)) for s in ALL_SUBJECTS}
    wt_sum  = sum(weights.values()) or 1

    subject_targets = {}
    for s in ALL_SUBJECTS:
        current = scores.get(s, 0)
        gain    = round(max(0, total_gap * weights[s] / wt_sum), 1)
        needed  = min(MAX_MARKS[s], round(current + gain, 1))
        subject_targets[s] = {"current": current, "target": needed,
                               "gain_needed": round(needed - current, 1)}

    priority = sorted(ALL_SUBJECTS, key=lambda s: scores.get(s, 0))
    sep      = "─" * 48
    lines    = [
        f"🎯 Personalised Recommendations — {name}",
        sep,
        f"  Current total  : {current_total} / {total_max}  ({total_pct}%)",
        f"  Target         : {target_pct}%  ({target_total} marks needed)",
        f"  Marks to gain  : {max(0, total_gap)} marks across all subjects",
        sep,
        "  📚 Subject-wise Action Plan:",
        "",
    ]

    for rank_idx, s in enumerate(priority, 1):
        cur   = subject_targets[s]["current"]
        tgt   = subject_targets[s]["target"]
        gain  = subject_targets[s]["gain_needed"]
        gap   = toppers[s]["gap"]
        label = s.title()
        pct_s = round(cur / MAX_MARKS[s] * 100, 1)

        pl = (
            "🔴 PRIORITY 1 (Weakest — focus most here)" if rank_idx == 1 else
            "🟡 PRIORITY 2 (Needs improvement)"         if rank_idx == 2 else
            "🟢 PRIORITY 3 (Maintain strength)"
        )
        lines += [
            f"  {label}   {pl}",
            f"    Current score   : {cur} / {MAX_MARKS[s]}  ({pct_s}%)",
            f"    Target score    : {tgt} / {MAX_MARKS[s]}  ({round(tgt/MAX_MARKS[s]*100,1)}%)",
            (f"    Need to gain    : +{gain} marks" if gain > 0
             else "    Status          : Already at or above target ✅"),
            f"    Class average   : {round(df[s].mean(), 1)}",
            f"    Gap to topper   : {gap} marks",
            "    Study tips:",
        ]
        for tip in STUDY_TIPS.get(s, [])[:2]:
            lines.append(f"      • {tip}")
        lines.append("")

    weakest_subj = priority[0]
    weakest_gain = subject_targets[weakest_subj]["gain_needed"]
    lines += [
        sep,
        "  🧑‍🏫 Mentor Summary:",
        (f"  To reach {target_pct}%, {name} needs to gain {max(0, total_gap)} marks overall. "
         f"Highest priority: {weakest_subj.title()} — improve by {weakest_gain} marks. "
         f"Study numericals and practice previous year papers daily."),
        sep,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# PDF REPORT GENERATOR  (with file cache)
# ─────────────────────────────────────────────────────
def generate_pdf_report(row, df):
    adm_no    = row["adm_no"]
    name      = row["name"]
    section   = row["section"]

    # ── Cache check: return existing file instantly ───────────────
    if adm_no in _report_cache and os.path.exists(_report_cache[adm_no]):
        print(f"[REPORT] Cache hit: {_report_cache[adm_no]}")
        return _report_cache[adm_no]

    safe_name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    filename  = os.path.join(REPORT_OUT_PATH, f"{adm_no}_{safe_name}_report.pdf")

    doc    = SimpleDocTemplate(filename, pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=0.75*inch,   bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    DARK_BLUE  = colors.HexColor("#1a1a2e")
    CYAN       = colors.HexColor("#00d4ff")
    GREEN      = colors.HexColor("#10a37f")
    LIGHT_GREY = colors.HexColor("#f5f5f5")
    RED        = colors.HexColor("#e74c3c")
    ORANGE     = colors.HexColor("#f39c12")

    def ps(pname, **kwargs):
        base = kwargs.pop("parent", styles["Normal"])
        return ParagraphStyle(pname, parent=base, **kwargs)

    title_style  = ps("T",  parent=styles["Title"],   fontSize=22, textColor=DARK_BLUE,
                       alignment=TA_CENTER, spaceAfter=6)
    sub_style    = ps("S",  fontSize=11, textColor=colors.grey,
                       alignment=TA_CENTER, spaceAfter=4)
    h2_style     = ps("H2", parent=styles["Heading2"], fontSize=13, textColor=DARK_BLUE,
                       spaceBefore=14, spaceAfter=6)
    body_style   = ps("B",  fontSize=10, leading=16)
    bullet_style = ps("BU", fontSize=10, leading=16, leftIndent=20)
    footer_style = ps("F",  fontSize=8,  textColor=colors.grey, alignment=TA_CENTER)

    story          = []
    total_students = len(df)

    # ── Header ────────────────────────────────────────
    story.append(Paragraph("Student Performance Report", title_style))
    story.append(Paragraph("KCET Grand Test — AI Performance Analysis", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=CYAN))
    story.append(Spacer(1, 10))

    info_tbl = Table(
        [["Name", name, "Adm No.", adm_no],
         ["Section", section, "Test", "KCET Grand Test"]],
        colWidths=[1.1*inch, 2.6*inch, 1.1*inch, 2.1*inch]
    )
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), DARK_BLUE),
        ("BACKGROUND",  (2, 0), (2, -1), DARK_BLUE),
        ("TEXTCOLOR",   (0, 0), (0, -1), colors.white),
        ("TEXTCOLOR",   (2, 0), (2, -1), colors.white),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("PADDING",     (0, 0), (-1, -1), 7),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [LIGHT_GREY, colors.white]),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 16))

    # ── Section 1: Performance Summary ───────────────
    story.append(Paragraph("1. Performance Summary", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 6))

    scores   = {s: row.get(s) for s in ALL_SUBJECTS}
    mk_data  = [["Subject", "Marks", "Max", "Rank", "%", "Class Avg", "Status"]]
    for s in ALL_SUBJECTS:
        mks     = row.get(s)
        cls_avg = round(df[s].mean(), 1)
        status  = "Above Avg" if (pd.notna(mks) and mks >= cls_avg) else "Below Avg"
        mk_data.append([
            s.title(),
            str(mks) if pd.notna(mks) else "N/A",
            str(MAX_MARKS[s]),
            str(fmt_rank(row.get(f"{s}_rank"))),
            f"{row.get(f'{s}_percent')}%" if pd.notna(row.get(f"{s}_percent")) else "N/A",
            str(cls_avg),
            status,
        ])

    t_mks    = row.get("total")
    t_cls    = round(df["total"].mean(), 1)
    t_status = "Above Avg" if (pd.notna(t_mks) and t_mks >= t_cls) else "Below Avg"
    mk_data.append([
        "TOTAL",
        str(t_mks) if pd.notna(t_mks) else "N/A",
        "180",
        str(fmt_rank(row.get("total_rank"))),
        f"{row.get('total_percent')}%" if pd.notna(row.get("total_percent")) else "N/A",
        str(t_cls),
        t_status,
    ])

    mk_tbl = Table(mk_data, colWidths=[1.05*inch, 0.8*inch, 0.6*inch,
                                        0.7*inch, 0.85*inch, 0.9*inch, 1.0*inch])
    mk_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),  (-1, 0),  DARK_BLUE),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",      (0, 1),  (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0),  (-1, -1), 9),
        ("PADDING",       (0, 0),  (-1, -1), 6),
        ("GRID",          (0, 0),  (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -2), [LIGHT_GREY, colors.white]),
        ("BACKGROUND",    (0, -1), (-1, -1), colors.HexColor("#e8f4f8")),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    for i, rd in enumerate(mk_data[1:], 1):
        c = GREEN if rd[-1] == "Above Avg" else RED
        mk_tbl.setStyle(TableStyle([("TEXTCOLOR", (6, i), (6, i), c)]))
    story.append(mk_tbl)
    story.append(Spacer(1, 16))

    # ── Section 2: Performance Intelligence ──────────
    sc_vals   = [row[s] for s in ALL_SUBJECTS if pd.notna(row.get(s))]
    avg_sc    = round(sum(sc_vals)/len(sc_vals), 2) if sc_vals else 0
    n         = len(sc_vals)
    var       = sum((v-avg_sc)**2 for v in sc_vals)/n if n else 0
    std_sc    = round(math.sqrt(var), 2)
    sc_dict   = {s: row[s] for s in ALL_SUBJECTS if pd.notna(row.get(s))}
    strongest = max(sc_dict, key=sc_dict.get) if sc_dict else "N/A"
    weakest   = min(sc_dict, key=sc_dict.get) if sc_dict else "N/A"
    cons      = ("Very Consistent" if std_sc <= 5 else
                 "Moderately Consistent" if std_sc <= 10 else "Inconsistent")

    story.append(Paragraph("2. Performance Intelligence", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 6))

    intel_data = [
        ["Metric", "Value"],
        ["Average Score (Bio+Phy+Chem)", str(avg_sc)],
        ["Standard Deviation (Consistency)", str(std_sc)],
        ["Consistency Level", cons],
        ["Strongest Subject", strongest.title() if strongest != "N/A" else "N/A"],
        ["Weakest Subject",   weakest.title()   if weakest   != "N/A" else "N/A"],
        ["Class Rank", f"{fmt_rank(row.get('total_rank'))} / {total_students}"],
        ["Gap to Class Topper",
         f"{round(df['total'].max() - (row.get('total') or 0), 1)} marks"],
    ]
    intel_tbl = Table(intel_data, colWidths=[3.1*inch, 3.8*inch])
    intel_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),  (-1, 0),  CYAN),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  DARK_BLUE),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",      (0, 1),  (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",      (1, 1),  (1, -1),  "Helvetica"),
        ("FONTSIZE",      (0, 0),  (-1, -1), 10),
        ("PADDING",       (0, 0),  (-1, -1), 7),
        ("GRID",          (0, 0),  (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -1), [LIGHT_GREY, colors.white]),
    ]))
    story.append(intel_tbl)
    story.append(Spacer(1, 16))

    # ── Section 3: Strengths & Weaknesses ────────────
    story.append(Paragraph("3. Strengths and Areas for Improvement", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 6))

    strengths_list, weaknesses_list = [], []
    for s in ALL_SUBJECTS:
        if pd.notna(row.get(s)):
            cls_avg = df[s].mean()
            diff    = round(row[s] - cls_avg, 1)
            if diff >= 0:
                strengths_list.append(
                    f"{s.title()}: {row[s]} marks (+{diff} above class avg {round(cls_avg,1)})")
            else:
                weaknesses_list.append(
                    f"{s.title()}: {row[s]} marks ({abs(diff)} below class avg {round(cls_avg,1)})")

    if strengths_list:
        story.append(Paragraph("<b>Strengths:</b>", body_style))
        for s in strengths_list:
            story.append(Paragraph(f"• {s}", bullet_style))
        story.append(Spacer(1, 6))
    if weaknesses_list:
        story.append(Paragraph("<b>Areas Needing Improvement:</b>", body_style))
        for w in weaknesses_list:
            story.append(Paragraph(f"• {w}", bullet_style))
    story.append(Spacer(1, 10))

    # ── Section 4: Personalised Recommendations ───────
    story.append(Paragraph("4. Personalised Recommendations", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 6))

    total_pct_v = row.get("total_percent") or 0
    if total_pct_v >= 90:   tgt = 95.0
    elif total_pct_v >= 75: tgt = 90.0
    elif total_pct_v >= 60: tgt = 75.0
    elif total_pct_v >= 45: tgt = 60.0
    else:                   tgt = 50.0

    total_max    = sum(MAX_MARKS[s] for s in ALL_SUBJECTS)
    target_total = round(tgt / 100 * total_max, 1)
    curr_total   = sum(sc_dict.get(s, 0) for s in ALL_SUBJECTS)
    total_gap    = round(target_total - curr_total, 1)

    rec_data = [["Subject", "Current", "Target", "Need", "Gap to Topper"]]
    priority = sorted(ALL_SUBJECTS, key=lambda s: sc_dict.get(s, 0))
    for s in ALL_SUBJECTS:
        cur  = sc_dict.get(s, 0)
        tg   = min(MAX_MARKS[s], round(cur + max(0, total_gap/len(ALL_SUBJECTS)), 1))
        gain = round(max(0, tg - cur), 1)
        gap  = round(df[s].max() - cur, 1)
        rec_data.append([s.title(), str(cur), str(tg), f"+{gain}", str(gap)])

    rec_tbl = Table(rec_data,
                    colWidths=[1.3*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.3*inch])
    rec_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),  (-1, 0),  GREEN),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",      (0, 1),  (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0),  (-1, -1), 9),
        ("PADDING",       (0, 0),  (-1, -1), 6),
        ("GRID",          (0, 0),  (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -1), [LIGHT_GREY, colors.white]),
    ]))
    story.append(rec_tbl)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<b>Target:</b> Reach {tgt}% — need +{max(0, total_gap)} marks total.",
        body_style))
    story.append(Spacer(1, 6))

    for s in priority:
        tips = STUDY_TIPS.get(s, [])
        if tips:
            story.append(Paragraph(f"<b>{s.title()} Tips:</b>", body_style))
            for t in tips[:2]:
                story.append(Paragraph(f"• {t}", bullet_style))
            story.append(Spacer(1, 4))
    story.append(Spacer(1, 10))

    # ── Section 5: Prediction ─────────────────────────
    story.append(Paragraph("5. Prediction", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 6))

    if total_pct_v >= 70:
        pred_text, pred_color = ("Excellent performance. On track for a top rank.", GREEN)
    elif total_pct_v >= 50:
        pred_text, pred_color = (
            "Satisfactory. Focused effort on weaker subjects will improve rank.", ORANGE)
    else:
        pred_text, pred_color = (
            "At risk. Immediate structured intervention is essential.", RED)

    story.append(Paragraph(pred_text,
                            ParagraphStyle("PR", parent=styles["Normal"],
                                           fontSize=10, leading=16,
                                           textColor=pred_color, leftIndent=10)))
    story.append(Spacer(1, 16))

    # ── Section 6: Study Suggestions ──────────────────
    story.append(Paragraph("6. Study Suggestions", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 6))

    suggestions = []
    if weakest and weakest != "N/A":
        wk_gap = round(df[weakest].max() - (sc_dict.get(weakest) or 0), 1)
        suggestions.append(
            f"Prioritise {weakest.title()} — improve by {wk_gap} marks to reach topper level.")
    if std_sc > 10:
        suggestions.append("Balance study time across subjects — current spread is too high.")
    if total_pct_v < 50:
        suggestions.append("Aim for 50% first. Attempt all questions — never leave blanks.")
    suggestions += [
        "Attempt one full KCET mock test every week under timed conditions.",
        "Review every wrong answer immediately after the test.",
        "Revise NCERT line by line — 60% of KCET is directly from NCERT.",
    ]
    for i, s in enumerate(suggestions, 1):
        story.append(Paragraph(f"{i}. {s}", bullet_style))
        story.append(Spacer(1, 3))

    # ── Footer ────────────────────────────────────────
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        "Generated by AI Conversational Bot — Confidential", footer_style))

    doc.build(story)
    print(f"[REPORT] Saved: {filename}")

    # ── Store in cache ────────────────────────────────
    _report_cache[adm_no] = filename
    return filename


# ─────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────
def answer_marks_query(query, df):
    if df is None:
        return "Student marks data is not available.", None

    q = query.lower()
    student_row = find_student_by_adm(q, df) or find_student_by_name(query, df)

    if student_row is not None:
        print(f"[MARKS] Student found: {student_row['name']}")
    else:
        print(f"[MARKS] No student matched for: '{query}'")

    if is_report_query(query):
        if student_row is not None:
            filepath = generate_pdf_report(student_row, df)
            return (f"Report generated for {student_row['name']}.\n"
                    f"Click the Download PDF Report button below."), filepath
        names = ", ".join(df["name"].str.title().tolist())
        return (f"Please specify a student name.\n"
                f"Try: 'generate report for <name>' or 'report for <adm_no>'\n"
                f"Available students: {names}"), None

    if is_recommendation_query(query):
        if student_row is not None:
            target    = None
            pct_match = re.search(r'(\d{2,3})\s*%', query)
            if pct_match:
                target = float(pct_match.group(1))
            return get_personalized_recommendations(student_row, df, target), None
        names = ", ".join(df["name"].str.title().tolist())
        return (f"Please specify a student name.\n"
                f"Example: 'recommend for <name>'\nStudents: {names}"), None

    if is_intelligence_query(query):
        if student_row is not None:
            return get_performance_intelligence(student_row, df), None
        names = ", ".join(df["name"].str.title().tolist())
        return (f"Please specify a student name.\n"
                f"Example: 'analyse performance of <name>'\nStudents: {names}"), None

    if student_row is not None:
        return _handle_student_query(q, student_row), None

    result = _handle_class_query(q, df)
    return result, None


# ─────────────────────────────────────────────────────
# STUDENT QUERY HANDLER
# ─────────────────────────────────────────────────────
def _handle_student_query(q, row):
    name     = row["name"]
    subjects = detect_all_subjects(q)
    wants_marks, wants_rank, wants_percent, wants_average = detect_wants(q)

    if wants_average:
        scores = {s: row[s] for s in ALL_SUBJECTS if pd.notna(row.get(s))}
        if not scores:
            return f"No subject scores found for {name}."
        avg   = round(sum(scores.values()) / len(scores), 2)
        lines = [f"Average marks for {name}: {avg}"]
        for s, v in scores.items():
            lines.append(f"  {s.title()}: {v}")
        return "\n".join(lines)

    if subjects:
        return "\n\n".join(
            student_subject_block(name, row, s, wants_marks, wants_rank, wants_percent)
            for s in subjects
        )

    lines = [f"Full profile for {name}:"]
    for s in ["total"] + ALL_SUBJECTS:
        label = "Total" if s == "total" else s.title()
        lines.append(f"  {label}: {row[s]} marks | "
                     f"Rank {fmt_rank(row.get(f'{s}_rank'))} | "
                     f"{fmt_pct(row.get(f'{s}_percent'))}%")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# CLASS QUERY HANDLER
# ─────────────────────────────────────────────────────
def _handle_class_query(q, df):
    subjects = detect_all_subjects(q)
    cols     = subjects if subjects else ["total"]
    wants_marks, wants_rank, wants_percent, wants_average = detect_wants(q)

    sort_type = (
        "rank"    if any(k in q for k in ["highest rank", "best rank", "top rank", "by rank"]) else
        "percent" if any(k in q for k in ["highest percent", "highest percentage",
                                           "best percent", "by percent", "by %"]) else
        "marks"
    )

    if "top" in q:
        nums  = re.findall(r'\d+', q)
        n     = int(nums[0]) if nums else 5
        blocks = []
        for col in cols:
            label = "Total" if col == "total" else col.title()
            if sort_type == "rank":
                top   = df.nsmallest(n, f"{col}_rank")
                lines = [f"Top {n} by Rank — {label}:"]
                for i, (_, r) in enumerate(top.iterrows(), 1):
                    lines.append(f"  {i}. {r['name']} | Rank {fmt_rank(r[f'{col}_rank'])} "
                                  f"| {r[col]} marks | {fmt_pct(r[f'{col}_percent'])}%")
            elif sort_type == "percent":
                top   = df.nlargest(n, f"{col}_percent")
                lines = [f"Top {n} by Percentage — {label}:"]
                for i, (_, r) in enumerate(top.iterrows(), 1):
                    lines.append(f"  {i}. {r['name']} | {fmt_pct(r[f'{col}_percent'])}% "
                                  f"| {r[col]} marks")
            else:
                top   = df.nlargest(n, col)
                lines = [f"Top {n} by Marks — {label}:"]
                for i, (_, r) in enumerate(top.iterrows(), 1):
                    lines.append(f"  {i}. {r['name']} | {r[col]} marks "
                                  f"| Rank {fmt_rank(r[f'{col}_rank'])} "
                                  f"| {fmt_pct(r[f'{col}_percent'])}%")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    if "second highest" in q or "2nd highest" in q:
        blocks = []
        for col in cols:
            label = "Total" if col == "total" else col.title()
            uniq  = df[col].dropna().sort_values(ascending=False).unique()
            if len(uniq) >= 2:
                sec  = uniq[1]
                mchd = df[df[col] == sec]
                blocks.append(f"2nd highest in {label}: {', '.join(mchd['name'].values)} "
                               f"with {sec} marks.")
            else:
                blocks.append(f"Not enough data for 2nd highest in {label}.")
        return "\n".join(blocks)

    if any(k in q for k in ["highest rank", "best rank", "top rank"]):
        blocks = []
        for col in cols:
            label   = "Total" if col == "total" else col.title()
            best    = df[f"{col}_rank"].min()
            matched = df[df[f"{col}_rank"] == best]
            blocks.append(f"Highest Rank in {label}: {', '.join(matched['name'].values)} "
                          f"| Rank {int(best)}")
        return "\n".join(blocks)

    if any(k in q for k in ["lowest rank", "worst rank", "last rank"]):
        blocks = []
        for col in cols:
            label   = "Total" if col == "total" else col.title()
            worst   = df[f"{col}_rank"].max()
            matched = df[df[f"{col}_rank"] == worst]
            blocks.append(f"Lowest Rank in {label}: {', '.join(matched['name'].values)} "
                          f"| Rank {int(worst)}")
        return "\n".join(blocks)

    if any(k in q for k in ["highest percent", "highest percentage", "highest %",
                              "best percent", "best percentage", "maximum percent"]):
        blocks = []
        for col in cols:
            label   = "Total" if col == "total" else col.title()
            mx      = df[f"{col}_percent"].max()
            matched = df[df[f"{col}_percent"] == mx]
            blocks.append(f"Highest % in {label}: {', '.join(matched['name'].values)} | {mx}%")
        return "\n".join(blocks)

    if any(k in q for k in ["lowest percent", "lowest percentage", "minimum percent"]):
        blocks = []
        for col in cols:
            label   = "Total" if col == "total" else col.title()
            mn      = df[f"{col}_percent"].min()
            matched = df[df[f"{col}_percent"] == mn]
            blocks.append(f"Lowest % in {label}: {', '.join(matched['name'].values)} | {mn}%")
        return "\n".join(blocks)

    if wants_average:
        if subjects:
            return "\n".join(
                f"Class average in {('Total' if c=='total' else c.title())}: "
                f"{round(df[c].dropna().mean(), 2)} marks."
                for c in subjects
            )
        lines = ["Class averages:"]
        for s in ALL_SUBJECTS + ["total"]:
            label = "Total" if s == "total" else s.title()
            lines.append(f"  {label}: {round(df[s].dropna().mean(), 2)} marks")
        return "\n".join(lines)

    if "rank" in q:
        nums = re.findall(r'\d+', q)
        if nums:
            rn     = float(nums[0])
            blocks = []
            for col in cols:
                label   = "Total" if col == "total" else col.title()
                matched = df[df[f"{col}_rank"] == rn]
                if not matched.empty:
                    blocks.append(
                        f"Rank {int(rn)} in {label}: "
                        f"{', '.join(matched['name'].values)} | "
                        f"{', '.join(str(v) for v in matched[col].values)} marks | "
                        f"{', '.join(str(v) for v in matched[f'{col}_percent'].values)}%")
                else:
                    blocks.append(f"No student at Rank {int(rn)} in {label}.")
            return "\n".join(blocks)
        return "Please specify a rank number (e.g. 'who is rank 3 in physics')."

    if any(k in q for k in ["highest", "topper", "top scorer", "who scored", "maximum"]):
        blocks = []
        for col in cols:
            label   = "Total" if col == "total" else col.title()
            mx      = df[col].max()
            matched = df[df[col] == mx]
            names   = ", ".join(matched["name"].values)
            parts   = [f"Highest — {label}:", f"  Student: {names}"]
            if wants_marks:   parts.append(f"  Marks:      {mx}")
            if wants_rank:    parts.append(f"  Rank:       {fmt_rank(matched[f'{col}_rank'].values[0])}")
            if wants_percent: parts.append(f"  Percentage: {fmt_pct(matched[f'{col}_percent'].values[0])}%")
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)

    if any(k in q for k in ["lowest", "least", "minimum"]):
        blocks = []
        for col in cols:
            label   = "Total" if col == "total" else col.title()
            mn      = df[col].min()
            matched = df[df[col] == mn]
            blocks.append(f"Lowest in {label}: {', '.join(matched['name'].values)} "
                          f"with {mn} marks.")
        return "\n".join(blocks)

    if "pass" in q or "fail" in q:
        col     = cols[0] if cols else "total"
        pct_col = f"{col}_percent"
        label   = "Total" if col == "total" else col.title()
        if "fail" in q:
            lst  = df[df[pct_col] < 50]["name"].tolist()
            body = "\n".join(f"  {n}" for n in lst) or "  None"
            return f"Students who failed in {label} (< 50%):\n{body}"
        lst  = df[df[pct_col] >= 50]["name"].tolist()
        body = "\n".join(f"  {n}" for n in lst) or "  None"
        return f"Students who passed in {label} (>= 50%):\n{body}"

    single = subjects[0] if len(subjects) == 1 else None
    if single and not any(k in q for k in [
        "highest", "lowest", "average", "top", "rank", "percent",
        "pass", "fail", "all", "show", "list", "who", "second",
        "2nd", "maximum", "minimum",
    ]):
        return (f"What would you like to know about {single.title()} marks?\n"
                f"  'highest marks in {single}'\n"
                f"  'highest rank in {single}'\n"
                f"  'average marks in {single}'\n"
                f"  'top 5 in {single}'")

    if any(k in q for k in ["all students", "show all", "list all", "marks"]):
        col   = cols[0] if cols else "total"
        label = "Total" if col == "total" else col.title()
        lines = [f"All students — {label} (high to low):"]
        for _, r in df.sort_values(col, ascending=False).iterrows():
            lines.append(f"  {r['name']}: {r[col]} marks | "
                         f"Rank {fmt_rank(r[f'{col}_rank'])} | "
                         f"{fmt_pct(r[f'{col}_percent'])}%")
        return "\n".join(lines)

    return None