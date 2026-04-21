"""
Build WM9B7_reflection_final.pdf from WM9B7_reflection_draft.md.

Uses pdflatex if available, otherwise falls back to reportlab.
Always writes the .tex file for reference.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).parent.parent
PLOTS  = ROOT / "notebooks" / "plots"
MD     = ROOT / "notebooks" / "WM9B7_reflection_draft.md"
TEX    = ROOT / "notebooks" / "WM9B7_reflection_final.tex"
PDF    = ROOT / "notebooks" / "WM9B7_reflection_final.pdf"
OUTDIR = ROOT / "notebooks"

md_text = MD.read_text(encoding="utf-8")


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def escape(text: str) -> str:
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def process_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*",
                  lambda m: f"\\textbf{{{m.group(1)}}}", text)
    text = re.sub(r"\*(.+?)\*",
                  lambda m: f"\\textit{{{m.group(1)}}}", text)
    text = re.sub(r"`(.+?)`",
                  lambda m: f"\\texttt{{{m.group(1)}}}", text)
    return text


def md_to_latex(md: str) -> str:
    lines       = md.split("\n")
    out         = []
    in_table    = False
    in_itemize  = False

    for line in lines:
        if (line.startswith("**Student:**") or
                line.startswith("**Module:**") or
                line.startswith("**Word count:**")):
            continue

        if line.startswith("# ") and not line.startswith("## "):
            if in_itemize:
                out.append(r"\end{itemize}"); in_itemize = False
            out.append(f"\\section*{{{escape(line[2:].strip())}}}")
            continue
        if line.startswith("## ") and not line.startswith("### "):
            if in_itemize:
                out.append(r"\end{itemize}"); in_itemize = False
            out.append(f"\\subsection*{{{escape(line[3:].strip())}}}")
            continue
        if line.startswith("### "):
            if in_itemize:
                out.append(r"\end{itemize}"); in_itemize = False
            out.append(f"\\subsubsection*{{{escape(line[4:].strip())}}}")
            continue

        if line.strip() in ("---", "***", "___"):
            out.append(r"\noindent\rule{\linewidth}{0.4pt}")
            continue

        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                in_table = True
                cols = [c.strip() for c in line.split("|") if c.strip()]
                col_spec = "l" + ("l" * (len(cols) - 1))
                out.append(r"\begin{center}")
                out.append(r"\begin{tabular}{" + col_spec + r"}\hline")
                cells = " & ".join(
                    f"\\textbf{{{escape(c)}}}" for c in cols
                )
                out.append(cells + r" \\ \hline")
            else:
                if re.match(r"^\|[\s\-:|]+\|$", line.strip()):
                    continue
                cells_raw = [c.strip() for c in line.split("|") if c.strip()]
                cells = " & ".join(escape(c) for c in cells_raw)
                out.append(cells + r" \\")
            continue
        else:
            if in_table:
                out.append(r"\hline\end{tabular}")
                out.append(r"\end{center}")
                in_table = False

        if line.startswith("> "):
            if in_itemize:
                out.append(r"\end{itemize}"); in_itemize = False
            content = process_inline(escape(line[2:].strip()))
            out.append(r"\begin{quote}\textit{" + content + r"}\end{quote}")
            continue

        if line.startswith("- ") or line.startswith("* "):
            if not in_itemize:
                out.append(r"\begin{itemize}")
                in_itemize = True
            content = process_inline(escape(line[2:].strip()))
            out.append(f"  \\item {content}")
            continue
        else:
            if in_itemize and line.strip() != "":
                out.append(r"\end{itemize}")
                in_itemize = False

        if line.strip() == "":
            if in_table:
                out.append(r"\hline\end{tabular}")
                out.append(r"\end{center}")
                in_table = False
            out.append("")
            continue

        out.append(process_inline(escape(line)))

    if in_itemize:
        out.append(r"\end{itemize}")
    if in_table:
        out.append(r"\hline\end{tabular}")
        out.append(r"\end{center}")

    return "\n".join(out)


def figure_block(img_name: str, caption: str, label: str) -> str:
    p = PLOTS / img_name
    if p.exists():
        latex_path = str(p).replace("\\", "/")
        return (
            "\n\\begin{figure}[h]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.95\\textwidth]{{{latex_path}}}\n"
            f"\\caption{{{escape(caption)}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{figure}\n"
        )
    return (
        "\n\\begin{center}\\textit{"
        f"[{escape(caption)} --- figure not found at {img_name}]"
        "}\\end{center}\n"
    )


body_latex = md_to_latex(md_text)

fig1 = figure_block(
    "04_confusion_matrices_comparison.png",
    "Figure 1: Confusion matrices for TF-IDF + Logistic Regression (left) "
    "and DistilBERT (right) on AG News test samples. "
    "DistilBERT shows notably fewer Business/World confusions.",
    "fig:confusion",
)
body_latex = body_latex.replace(
    r"\subsubsection*{2.4 Justification for Deep Learning in This Project}",
    fig1 +
    r"\subsubsection*{2.4 Justification for Deep Learning in This Project}",
)

fig2 = figure_block(
    "05_attention_visualisation.png",
    "Figure 2: Token-level attention weights from the final DistilBERT layer, "
    "averaged over all heads from the [CLS] token. Elevated weights on "
    "semantically significant tokens demonstrate contextual sensitivity "
    "absent from bag-of-words representations.",
    "fig:attention",
)
body_latex = body_latex.replace(
    r"\subsubsection*{2.3 Empirical Comparison}",
    fig2 +
    r"\subsubsection*{2.3 Empirical Comparison}",
)

fig3 = figure_block(
    "01_class_distribution.png",
    "Figure 3: AG News class distribution (train and test splits). "
    "The balanced 25\\% per class eliminates class imbalance as a "
    "confounding variable in the ML vs DL comparison.",
    "fig:classdist",
)
body_latex = body_latex.replace(
    r"\subsection*{4.2 Retrieval-Augmented Generation}",
    fig3 +
    r"\subsection*{4.2 Retrieval-Augmented Generation}",
)

tex_doc = (
    r"\documentclass[11pt,a4paper]{article}" + "\n"
    r"\usepackage[a4paper, margin=2.54cm]{geometry}" + "\n"
    r"\usepackage[T1]{fontenc}" + "\n"
    r"\usepackage[utf8]{inputenc}" + "\n"
    r"\usepackage{microtype}" + "\n"
    r"\usepackage{graphicx}" + "\n"
    r"\usepackage{booktabs}" + "\n"
    r"\usepackage{array}" + "\n"
    r"\usepackage{tabularx}" + "\n"
    r"\usepackage{longtable}" + "\n"
    r"\usepackage{setspace}" + "\n"
    r"\usepackage{parskip}" + "\n"
    r"\usepackage{hyperref}" + "\n"
    r"\usepackage{url}" + "\n"
    r"\usepackage{titlesec}" + "\n"
    r"\usepackage{enumitem}" + "\n"
    r"\usepackage{fancyhdr}" + "\n"
    r"\usepackage{hanging}" + "\n\n"
    r"\setstretch{1.15}" + "\n\n"
    r"\titleformat{\section}[block]" + "\n"
    r"  {\normalfont\large\bfseries}{\thesection}{1em}{}" + "\n"
    r"\titleformat{\subsection}[block]" + "\n"
    r"  {\normalfont\normalsize\bfseries}{\thesubsection}{1em}{}" + "\n"
    r"\titleformat{\subsubsection}[block]" + "\n"
    r"  {\normalfont\normalsize\bfseries}{\thesubsubsection}{1em}{}" + "\n\n"
    r"\setlength{\parskip}{6pt}" + "\n"
    r"\setlength{\parindent}{0pt}" + "\n\n"
    r"\pagestyle{fancy}" + "\n"
    r"\fancyhf{}" + "\n"
    r"\rhead{\small WMG9B7 Individual Assessment}" + "\n"
    r"\lhead{\small Sourabha K Kallapur}" + "\n"
    r"\rfoot{\small \thepage}" + "\n"
    r"\renewcommand{\headrulewidth}{0.4pt}" + "\n\n"
    r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black}"
    + "\n\n"
    r"\begin{document}" + "\n\n"
    r"\begin{center}" + "\n"
    r"{\Large\bfseries FinSight: A Critical Reflection on Deep Learning\\" + "\n"
    r"for Financial News Risk Intelligence}\\[8pt]" + "\n"
    r"{\normalsize Student: Sourabha K Kallapur}\\" + "\n"
    r"{\normalsize Module: WMG9B7 --- Artificial Intelligence and Deep Learning}\\" + "\n"
    r"{\normalsize Word count: 2703 (excluding references)}\\[6pt]" + "\n"
    r"\noindent\rule{\linewidth}{0.8pt}" + "\n"
    r"\end{center}" + "\n\n"
    + body_latex + "\n\n"
    r"\end{document}" + "\n"
)

TEX.write_text(tex_doc, encoding="utf-8")
print(f"LaTeX file written: {TEX}")

# ── Try pdflatex ──────────────────────────────────────────────────────────────
pdflatex = shutil.which("pdflatex")

if pdflatex:
    print("pdflatex found — compiling PDF (pass 1)...")
    r1 = subprocess.run(
        [pdflatex, "-interaction=nonstopmode",
         "-output-directory", str(OUTDIR), str(TEX)],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    print("Compiling PDF (pass 2)...")
    r2 = subprocess.run(
        [pdflatex, "-interaction=nonstopmode",
         "-output-directory", str(OUTDIR), str(TEX)],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    if PDF.exists():
        size_kb = PDF.stat().st_size // 1024
        print(f"\nSUCCESS (pdflatex): {PDF}  [{size_kb} KB]")
        for ext in [".aux", ".log", ".out", ".toc"]:
            f = OUTDIR / (TEX.stem + ext)
            if f.exists():
                f.unlink()
        print("Auxiliary files cleaned.")
        sys.exit(0)
    else:
        print("pdflatex compilation failed — falling back to reportlab")
        log = r2.stdout + r2.stderr
        print("\n".join(log.split("\n")[-20:]))
else:
    print("pdflatex not found — using reportlab fallback")


# ── reportlab fallback ────────────────────────────────────────────────────────
print("Building PDF with reportlab...")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_W, PAGE_H = A4
MARGIN = 2.54 * cm

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "ReportTitle",
    parent=styles["Normal"],
    fontSize=15,
    fontName="Helvetica-Bold",
    leading=20,
    alignment=TA_CENTER,
    spaceAfter=4,
)
meta_style = ParagraphStyle(
    "Meta",
    parent=styles["Normal"],
    fontSize=10,
    fontName="Helvetica",
    leading=14,
    alignment=TA_CENTER,
    spaceAfter=2,
)
h1_style = ParagraphStyle(
    "H1",
    parent=styles["Normal"],
    fontSize=13,
    fontName="Helvetica-Bold",
    leading=18,
    spaceBefore=14,
    spaceAfter=4,
)
h2_style = ParagraphStyle(
    "H2",
    parent=styles["Normal"],
    fontSize=11,
    fontName="Helvetica-Bold",
    leading=15,
    spaceBefore=10,
    spaceAfter=3,
)
h3_style = ParagraphStyle(
    "H3",
    parent=styles["Normal"],
    fontSize=10.5,
    fontName="Helvetica-BoldOblique",
    leading=14,
    spaceBefore=8,
    spaceAfter=2,
)
body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10,
    fontName="Helvetica",
    leading=14,
    alignment=TA_JUSTIFY,
    spaceBefore=3,
    spaceAfter=3,
)
quote_style = ParagraphStyle(
    "Quote",
    parent=styles["Normal"],
    fontSize=10,
    fontName="Helvetica-Oblique",
    leading=14,
    leftIndent=20,
    rightIndent=20,
    spaceBefore=4,
    spaceAfter=4,
)
bullet_style = ParagraphStyle(
    "Bullet",
    parent=styles["Normal"],
    fontSize=10,
    fontName="Helvetica",
    leading=14,
    leftIndent=16,
    firstLineIndent=-8,
    spaceBefore=1,
    spaceAfter=1,
)
ref_style = ParagraphStyle(
    "Ref",
    parent=styles["Normal"],
    fontSize=9.5,
    fontName="Helvetica",
    leading=13,
    leftIndent=18,
    firstLineIndent=-18,
    spaceBefore=3,
    spaceAfter=3,
)

caption_style = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontSize=9,
    fontName="Helvetica-Oblique",
    leading=12,
    alignment=TA_CENTER,
    spaceBefore=2,
    spaceAfter=8,
)

TABLE_CELL = ParagraphStyle(
    "TC",
    parent=styles["Normal"],
    fontSize=9,
    fontName="Helvetica",
    leading=12,
)
TABLE_HEADER = ParagraphStyle(
    "TH",
    parent=styles["Normal"],
    fontSize=9,
    fontName="Helvetica-Bold",
    leading=12,
)


def rl_inline(raw: str) -> str:
    raw = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", raw)
    raw = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", raw)
    raw = re.sub(r"`(.+?)`",       r"<font name='Courier'>\1</font>", raw)
    raw = raw.replace("—", "\u2014").replace("–", "\u2013")
    return raw


def make_figure(img_name: str, caption: str, max_width_cm: float = 14.5) -> list:
    p = PLOTS / img_name
    if not p.exists():
        return [Paragraph(f"[Figure not found: {img_name}]", caption_style)]
    max_w = max_width_cm * cm
    img = Image(str(p))
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth  = min(max_w, img.imageWidth)
    img.drawHeight = img.drawWidth * aspect
    if img.drawHeight > 12 * cm:
        img.drawHeight = 12 * cm
        img.drawWidth  = img.drawHeight / aspect
    return [img, Paragraph(caption, caption_style)]


# ── Figure injection markers ──
FIG1_MARKER = "##FIG_CONFUSION##"
FIG2_MARKER = "##FIG_ATTENTION##"
FIG3_MARKER = "##FIG_CLASSDIST##"

FIG1_CAPTION = (
    "Figure 1: Confusion matrices for TF-IDF + Logistic Regression (left) "
    "and DistilBERT (right) on AG News test samples. "
    "DistilBERT shows notably fewer Business/World confusions."
)
FIG2_CAPTION = (
    "Figure 2: Token-level attention weights from the final DistilBERT layer, "
    "averaged over all heads from the [CLS] token. Elevated weights on "
    "semantically significant tokens demonstrate contextual sensitivity "
    "absent from bag-of-words representations."
)
FIG3_CAPTION = (
    "Figure 3: AG News class distribution (train and test splits). "
    "The balanced 25% per class eliminates class imbalance as a "
    "confounding variable in the ML vs DL comparison."
)

FIGURE_MAP = {
    FIG1_MARKER: ("04_confusion_matrices_comparison.png", FIG1_CAPTION),
    FIG2_MARKER: ("05_attention_visualisation.png",       FIG2_CAPTION),
    FIG3_MARKER: ("01_class_distribution.png",            FIG3_CAPTION),
}


def parse_md_to_story(md: str) -> list:
    story = []
    lines = md.split("\n")
    i = 0
    in_table  = False
    table_rows: list[list] = []
    in_bullet = False
    bullet_buf: list[str] = []

    def flush_bullets():
        nonlocal in_bullet, bullet_buf
        if bullet_buf:
            for b in bullet_buf:
                story.append(Paragraph("\u2022\u2002" + rl_inline(b), bullet_style))
            bullet_buf = []
        in_bullet = False

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        col_count = max(len(r) for r in table_rows)
        avail_w   = PAGE_W - 2 * MARGIN
        col_w     = avail_w / col_count
        formatted = []
        for ri, row in enumerate(table_rows):
            st = TABLE_HEADER if ri == 0 else TABLE_CELL
            formatted.append([Paragraph(c, st) for c in row])
        t = Table(formatted, colWidths=[col_w] * col_count,
                  repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#3B4A6B")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#F5F7FA"), colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(Spacer(1, 4))
        story.append(t)
        story.append(Spacer(1, 6))
        table_rows = []
        in_table   = False

    while i < len(lines):
        line = lines[i]

        # Skip front-matter
        if (line.startswith("**Student:**") or
                line.startswith("**Module:**") or
                line.startswith("**Word count:**")):
            i += 1
            continue

        # Figure markers
        stripped = line.strip()
        if stripped in FIGURE_MAP:
            flush_bullets()
            flush_table()
            img_name, caption = FIGURE_MAP[stripped]
            story.extend(make_figure(img_name, caption))
            i += 1
            continue

        # Headings
        if line.startswith("# ") and not line.startswith("## "):
            flush_bullets(); flush_table()
            story.append(Paragraph(rl_inline(line[2:].strip()), h1_style))
            i += 1; continue
        if line.startswith("## ") and not line.startswith("### "):
            flush_bullets(); flush_table()
            story.append(Paragraph(rl_inline(line[3:].strip()), h2_style))
            i += 1; continue
        if line.startswith("### "):
            flush_bullets(); flush_table()
            story.append(Paragraph(rl_inline(line[4:].strip()), h3_style))
            i += 1; continue

        # HR
        if stripped in ("---", "***", "___"):
            flush_bullets(); flush_table()
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.grey, spaceAfter=4))
            i += 1; continue

        # Table rows
        if "|" in line and stripped.startswith("|"):
            flush_bullets()
            if not in_table:
                in_table = True
                table_rows = []
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                i += 1; continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            table_rows.append(cells)
            i += 1; continue
        else:
            if in_table:
                flush_table()

        # Blockquote
        if line.startswith("> "):
            flush_bullets()
            story.append(Paragraph(rl_inline(line[2:].strip()), quote_style))
            i += 1; continue

        # Bullet
        if line.startswith("- ") or line.startswith("* "):
            in_bullet = True
            bullet_buf.append(line[2:].strip())
            i += 1; continue
        else:
            if in_bullet:
                flush_bullets()

        # Empty line
        if stripped == "":
            flush_bullets(); flush_table()
            story.append(Spacer(1, 3))
            i += 1; continue

        # Paragraph
        is_ref = (len(story) > 0 and
                  any(getattr(x, "text", "").strip().startswith("## References")
                      for x in story[-5:] if hasattr(x, "text")))
        st = ref_style if is_ref else body_style
        story.append(Paragraph(rl_inline(stripped), st))
        i += 1

    flush_bullets()
    flush_table()
    return story


# ── Inject figure markers into markdown ──────────────────────────────────────
md_marked = md_text

md_marked = md_marked.replace(
    "### 2.4 Justification for Deep Learning in This Project",
    FIG1_MARKER + "\n### 2.4 Justification for Deep Learning in This Project",
)
md_marked = md_marked.replace(
    "### 2.3 Empirical Comparison",
    FIG2_MARKER + "\n### 2.3 Empirical Comparison",
)
md_marked = md_marked.replace(
    "### 4.2 Retrieval-Augmented Generation",
    FIG3_MARKER + "\n### 4.2 Retrieval-Augmented Generation",
)


# ── Page template with header/footer ─────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setStrokeColor(colors.HexColor("#AAAAAA"))
    canvas.line(MARGIN, PAGE_H - MARGIN + 4 * mm,
                PAGE_W - MARGIN, PAGE_H - MARGIN + 4 * mm)
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 6 * mm,
                      "Sourabha K Kallapur")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 6 * mm,
                           "WMG9B7 Individual Assessment")
    canvas.line(MARGIN, MARGIN - 4 * mm,
                PAGE_W - MARGIN, MARGIN - 4 * mm)
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 7 * mm,
                           str(doc.page))
    canvas.restoreState()


# ── Assemble document ─────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    str(PDF),
    pagesize=A4,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=MARGIN + 8 * mm,
    bottomMargin=MARGIN + 6 * mm,
    title="FinSight: A Critical Reflection on Deep Learning for Financial News Risk Intelligence",
    author="Sourabha K Kallapur",
)

story: list = []

# Cover block
story.append(Spacer(1, 6))
story.append(Paragraph(
    "FinSight: A Critical Reflection on Deep Learning<br/>"
    "for Financial News Risk Intelligence",
    title_style,
))
story.append(Spacer(1, 4))
story.append(Paragraph("Student: Sourabha K Kallapur", meta_style))
story.append(Paragraph(
    "Module: WMG9B7 \u2014 Artificial Intelligence and Deep Learning",
    meta_style,
))
story.append(Paragraph("Word count: 2703 (excluding references)", meta_style))
story.append(Spacer(1, 4))
story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#3B4A6B"),
                         spaceAfter=10))

story.extend(parse_md_to_story(md_marked))

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

if PDF.exists():
    size_kb = PDF.stat().st_size // 1024
    print(f"\nSUCCESS (reportlab): {PDF}")
    print(f"File size: {size_kb} KB")
else:
    print("\nPDF build failed.")
    sys.exit(1)
