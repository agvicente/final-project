#!/usr/bin/env python3
"""
Gera apresentação PowerPoint para reunião com orientador.
Uso: python docs/meeting/generate_meeting_pptx.py
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_PATH = SCRIPT_DIR / "2026-03-19-advisor-meeting.pptx"

PLOTS_C02 = PROJECT_ROOT / "experiments" / "results" / "campaign-02" / "plots"
PLOTS_C03 = PROJECT_ROOT / "experiments" / "results" / "campaign-03" / "plots"

# Colors
DARK_BLUE = RGBColor(0x1B, 0x3A, 0x5C)
MEDIUM_BLUE = RGBColor(0x2E, 0x75, 0xB6)
LIGHT_BLUE = RGBColor(0xD6, 0xE4, 0xF0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK_GRAY = RGBColor(0x40, 0x40, 0x40)
GREEN = RGBColor(0x27, 0xAE, 0x60)
RED = RGBColor(0xE7, 0x4C, 0x3C)
ORANGE = RGBColor(0xF3, 0x9C, 0x12)

# Slide dimensions (widescreen 16:9)
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)


def set_slide_bg(slide, color):
    """Set slide background color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 bold=False, color=BLACK, alignment=PP_ALIGN.LEFT,
                 font_name="Calibri"):
    """Add a text box to slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf


def add_paragraph(tf, text, font_size=18, bold=False, color=BLACK,
                  alignment=PP_ALIGN.LEFT, space_before=Pt(6)):
    """Add paragraph to existing text frame."""
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = "Calibri"
    p.alignment = alignment
    p.space_before = space_before
    return p


def add_table(slide, left, top, width, height, rows, cols):
    """Add table to slide."""
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    return table_shape.table


def style_header_row(table, color=DARK_BLUE, font_color=WHITE):
    """Style the header row of a table."""
    for i, cell in enumerate(table.rows[0].cells):
        cell.fill.solid()
        cell.fill.fore_color.rgb = color
        for p in cell.text_frame.paragraphs:
            p.font.color.rgb = font_color
            p.font.bold = True
            p.font.size = Pt(14)
            p.font.name = "Calibri"


def style_data_cell(cell, font_size=13, bold=False, color=BLACK,
                    alignment=PP_ALIGN.CENTER):
    """Style a data cell."""
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.alignment = alignment
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE


def set_cell(table, row, col, text, font_size=13, bold=False, color=BLACK):
    """Set cell text with styling."""
    cell = table.cell(row, col)
    cell.text = text
    style_data_cell(cell, font_size=font_size, bold=bold, color=color)


def add_title_bar(slide, title_text):
    """Add a colored title bar at the top."""
    # Title background
    from pptx.util import Emu
    shape = slide.shapes.add_shape(
        1,  # Rectangle
        Inches(0), Inches(0),
        SLIDE_WIDTH, Inches(1.2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BLUE
    shape.line.fill.background()

    # Title text
    add_text_box(slide, Inches(0.5), Inches(0.15), Inches(12), Inches(0.9),
                 title_text, font_size=32, bold=True, color=WHITE)


def create_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT

    # Use blank layout
    blank_layout = prs.slide_layouts[6]

    # ===== SLIDE 1: CAPA =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, DARK_BLUE)

    add_text_box(slide, Inches(1), Inches(1.5), Inches(11), Inches(1.5),
                 "Progresso Experimental",
                 font_size=44, bold=True, color=WHITE,
                 alignment=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(1), Inches(3.0), Inches(11), Inches(1.2),
                 "Detecção de Intrusão em IoT com\nClustering Evolutivo em Streaming",
                 font_size=28, color=LIGHT_BLUE,
                 alignment=PP_ALIGN.CENTER)

    tf = add_text_box(slide, Inches(1), Inches(5.0), Inches(11), Inches(1.5),
                      "Augusto — Mestrado PPGEE/UFMG",
                      font_size=20, color=WHITE,
                      alignment=PP_ALIGN.CENTER)
    add_paragraph(tf, "Reunião com Orientador — 19 de março de 2026",
                  font_size=18, color=LIGHT_BLUE,
                  alignment=PP_ALIGN.CENTER)
    add_paragraph(tf, "137 experimentos | 3 campanhas | 4 steps de ablation",
                  font_size=16, color=LIGHT_BLUE,
                  alignment=PP_ALIGN.CENTER)

    # ===== SLIDE 2: AGENDA =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Agenda")

    items = [
        ("1.", "Pipeline Streaming", "PCAP → Kafka → MicroTEDAclus"),
        ("2.", "Metodologia", "Ablation study com configuração cumulativa"),
        ("3.", "Campaign-01", "Baseline — anomaly rate invariante"),
        ("4.", "Campaign-02", "3 hipóteses: GT, features, janelas"),
        ("5.", "Campaign-03 S4", "Features comportamentais"),
        ("6.", "Consolidado", "Melhor resultado por ataque"),
        ("7.", "Insights", "5 pontos-chave"),
        ("8.", "Próximos passos", "3 opções para discussão"),
    ]
    for i, (num, title, desc) in enumerate(items):
        y = Inches(1.6) + Inches(i * 0.65)
        add_text_box(slide, Inches(1.0), y, Inches(0.5), Inches(0.5),
                     num, font_size=20, bold=True, color=MEDIUM_BLUE)
        tf = add_text_box(slide, Inches(1.5), y, Inches(10), Inches(0.5),
                          title, font_size=20, bold=True, color=DARK_GRAY)
        add_paragraph(tf, desc, font_size=16, color=DARK_GRAY)

    # ===== SLIDE 3: PIPELINE =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Pipeline Streaming")

    # Pipeline diagram as text boxes with arrows
    stages = [
        ("PCAPs\nCICIoT2023", Inches(0.5)),
        ("Kafka\nProducer", Inches(2.8)),
        ("Kafka\nBroker", Inches(5.1)),
        ("Flow\nConsumer", Inches(7.4)),
        ("MicroTEDA\nclus", Inches(9.7)),
        ("Métricas\nPrequential", Inches(11.5)),
    ]
    for label, x in stages:
        shape = slide.shapes.add_shape(
            5,  # Rounded rectangle
            x, Inches(2.0), Inches(1.8), Inches(1.2)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = MEDIUM_BLUE
        shape.line.fill.background()
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER

    # Arrows between stages
    for i in range(len(stages) - 1):
        x_start = stages[i][1] + Inches(1.8)
        x_end = stages[i+1][1]
        x_mid = (x_start + x_end) / 2
        add_text_box(slide, Emu(int(x_mid - Inches(0.2))), Inches(2.3),
                     Inches(0.5), Inches(0.5), "→",
                     font_size=28, bold=True, color=DARK_BLUE,
                     alignment=PP_ALIGN.CENTER)

    # Details below
    details = [
        ("5 tipos de ataque", "50k benign + 50k attack packets"),
        ("17 features per-flow", "Timeout 60s, event-time"),
        ("Clustering evolutivo", "Tipicalidade (Maia 2020)"),
        ("Test-then-train", "Ground truth por IP"),
    ]
    for i, (title, desc) in enumerate(details):
        x = Inches(0.8) + Inches(i * 3.2)
        tf = add_text_box(slide, x, Inches(4.0), Inches(2.8), Inches(1.5),
                          title, font_size=16, bold=True, color=DARK_BLUE)
        add_paragraph(tf, desc, font_size=14, color=DARK_GRAY)

    # ===== SLIDE 4: METODOLOGIA =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Metodologia — Ablation Study Cumulativo")

    # Table showing progression
    table = add_table(slide, Inches(0.5), Inches(1.5),
                      Inches(12.3), Inches(4.5), 6, 5)

    headers = ["Step", "Variável", "Opções Testadas", "Resultado", "Decisão"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h
    style_header_row(table)

    data = [
        ["C01", "Algoritmo", "TEDA vs MicroTEDAclus", "MicroTEDAclus 26x melhor", "MicroTEDAclus"],
        ["C02-S1", "Ground Truth", "Phase vs IP", "ICMP: 4%→27%", "IP GT"],
        ["C02-S2", "Features/flow", "v1(17) vs v2(25) vs v3(32)", "Zero impacto (±1pp)", "v1 (Occam)"],
        ["C02-S3", "Granularidade", "Per-flow vs Window (5-60s)", "Recall 15-20x melhor", "Window"],
        ["C03-S4", "Features/janela", "v1(12) vs v2(19 comportam.)", "Misto: 2/5 melhor, 2/5 pior", "Em análise"],
    ]
    for r, row_data in enumerate(data):
        for c, val in enumerate(row_data):
            set_cell(table, r+1, c, val)
            if r < 3:
                table.cell(r+1, 4).fill.solid()
                table.cell(r+1, 4).fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)

    add_text_box(slide, Inches(0.5), Inches(6.2), Inches(12), Inches(0.8),
                 "Cada step congela a melhor decisão e passa para o próximo → configuração cumulativa",
                 font_size=16, bold=True, color=MEDIUM_BLUE,
                 alignment=PP_ALIGN.CENTER)

    # ===== SLIDE 5: C01 BASELINE =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Campaign-01 — Baseline (17 runs)")

    # Key finding
    shape = slide.shapes.add_shape(5, Inches(0.5), Inches(1.5),
                                   Inches(12.3), Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xFD, 0xED, 0xED)
    shape.line.fill.background()
    add_text_box(slide, Inches(0.8), Inches(1.6), Inches(11.5), Inches(1.0),
                 "Achado principal: Anomaly rate invariante (~3.5%) com ou sem ataque",
                 font_size=22, bold=True, color=RED)

    # Results table
    table = add_table(slide, Inches(0.5), Inches(3.0),
                      Inches(8), Inches(2.5), 4, 4)

    for i, h in enumerate(["Cenário", "FPR / Recall", "Alvo", "Status"]):
        table.cell(0, i).text = h
    style_header_row(table)

    c01_data = [
        ["Benigno (FPR)", "3.5%", "≤ 5%", "Aprovado"],
        ["Todos ataques (Recall)", "~3-4%", "≥ 80%", "Reprovado"],
        ["TEDA vs MicroTEDAclus", "26x melhor", "—", "MicroTEDAclus"],
    ]
    for r, row_data in enumerate(c01_data):
        for c, val in enumerate(row_data):
            color = GREEN if val == "Aprovado" else (RED if val == "Reprovado" else BLACK)
            set_cell(table, r+1, c, val, color=color,
                     bold=(val in ("Aprovado", "Reprovado")))

    # Diagnosis
    tf = add_text_box(slide, Inches(0.5), Inches(5.8), Inches(12), Inches(1.5),
                      "Diagnóstico: O detector identifica outliers naturais do tráfego, não ataques.",
                      font_size=18, bold=True, color=DARK_BLUE)
    add_paragraph(tf, "Flows de ataque DDoS são estatisticamente indistinguíveis de flows benignos IoT.",
                  font_size=16, color=DARK_GRAY)
    add_paragraph(tf, "Problema é de representação (features), não de algoritmo.",
                  font_size=16, color=DARK_GRAY)

    # ===== SLIDE 6: C02 — 3 HIPÓTESES =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Campaign-02 — 3 Hipóteses (72 runs)")

    table = add_table(slide, Inches(0.3), Inches(1.5),
                      Inches(12.7), Inches(4.5), 4, 5)

    for i, h in enumerate(["Step", "Variável", "Resultado", "Decisão", "Impacto"]):
        table.cell(0, i).text = h
    style_header_row(table)

    s_data = [
        ["S1 — GT por IP", "phase → IP",
         "ICMP: 4%→27% Recall\nResto inalterado\nFPR estável ~3-4%",
         "Adotar IP GT",
         "Corrige medição\nmas não resolve\ndetecção"],
        ["S2 — Features", "v1(17)→v2(25)→v3(32)",
         "Zero impacto (±1pp)\nem todos os ataques",
         "Manter v1\n(Occam's razor)",
         "Features per-flow\nsaturaram"],
        ["S3 — Janelas", "Per-flow → Window\n(5s, 10s, 30s, 60s)",
         "SYN: 3%→54% (15x)\nRecon: 4%→45% (10x)\nMAS FPR explode (58% @60s)",
         "Direção certa\nfeatures insuficientes",
         "Muda a pergunta\nfundamental"],
    ]
    for r, row_data in enumerate(s_data):
        for c, val in enumerate(row_data):
            set_cell(table, r+1, c, val, font_size=12)

    # Highlight
    add_text_box(slide, Inches(0.5), Inches(6.3), Inches(12), Inches(0.8),
                 'S3 muda a pergunta: "Este flow é anômalo?" → "Este IP tem comportamento anômalo?"',
                 font_size=18, bold=True, color=MEDIUM_BLUE,
                 alignment=PP_ALIGN.CENTER)

    # ===== SLIDE 7: C02-S3 JANELAS (com plot) =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "C02-S3 — Recall por Janela Temporal")

    plot_path = PLOTS_C02 / "04_s3_recall_by_window.png"
    if plot_path.exists():
        slide.shapes.add_picture(
            str(plot_path),
            Inches(0.3), Inches(1.4),
            Inches(7.5), Inches(5.5)
        )
    else:
        add_text_box(slide, Inches(1), Inches(3), Inches(6), Inches(1),
                     f"[Plot não encontrado: {plot_path.name}]",
                     font_size=16, color=RED)

    # Key points on the right
    tf = add_text_box(slide, Inches(8.2), Inches(1.5), Inches(4.8), Inches(5.5),
                      "Melhorias de Recall:", font_size=20, bold=True, color=DARK_BLUE)
    points = [
        ("DDoS-SYN:", "3.5% → 53.9% (@30s)"),
        ("Recon:", "4.5% → 45.3% (@10s)"),
        ("Mirai:", "1.7% → 33.3% (@60s)"),
        ("", ""),
        ("Problema:", ""),
        ("FPR explode", "58% @60s benigno"),
        ("", ""),
        ("Trade-off severo:", ""),
        ("Bom Recall →", "FPR inaceitável"),
        ("Bom FPR →", "Recall insuficiente"),
    ]
    for title, val in points:
        if title:
            text = f"{title} {val}" if val else title
            bold = title.endswith(":") and not val
            color = RED if "Problema" in title or "FPR" in title else DARK_GRAY
            add_paragraph(tf, text, font_size=15, bold=bold, color=color)
        else:
            add_paragraph(tf, "", font_size=8)

    # ===== SLIDE 8: C03-S4 FEATURES COMPORTAMENTAIS (com plot) =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "C03-S4 — Features Comportamentais v1 vs v2 (48 runs)")

    plot_path = PLOTS_C03 / "01_s4_recall_v1_vs_v2_w10s.png"
    if plot_path.exists():
        slide.shapes.add_picture(
            str(plot_path),
            Inches(0.3), Inches(1.4),
            Inches(7.5), Inches(5.5)
        )
    else:
        add_text_box(slide, Inches(1), Inches(3), Inches(6), Inches(1),
                     f"[Plot não encontrado: {plot_path.name}]",
                     font_size=16, color=RED)

    # v2 features list + verdict
    tf = add_text_box(slide, Inches(8.2), Inches(1.5), Inches(4.8), Inches(2.5),
                      "7 features comportamentais (v2):",
                      font_size=16, bold=True, color=DARK_BLUE)
    features = [
        "dst_port_entropy",
        "dst_ip_entropy",
        "flows_per_second",
        "unanswered_ratio",
        "fwd_only_ratio",
        "small_packet_ratio",
        "syn_only_ratio",
    ]
    for f in features:
        add_paragraph(tf, f"  {f}", font_size=13, color=DARK_GRAY)

    # Verdict table
    tf2 = add_text_box(slide, Inches(8.2), Inches(4.5), Inches(4.8), Inches(2.8),
                       "Veredicto @w=10s:", font_size=16, bold=True, color=DARK_BLUE)
    verdicts = [
        ("ICMP:", "v2 desbloqueia (0→50%)", GREEN),
        ("Recon:", "v2 melhor (39→45%)", GREEN),
        ("SYN:", "v1 melhor (38% vs 31%)", RED),
        ("Mirai:", "v1 melhor (46% vs 38%)", RED),
        ("TCP:", "Ambos 0%", DARK_GRAY),
        ("FPR:", "v1=2.9% → v2=14.3%", RED),
    ]
    for label, desc, color in verdicts:
        add_paragraph(tf2, f"{label} {desc}", font_size=14, color=color)

    # ===== SLIDE 9: CONSOLIDADO =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Consolidado — Melhor Resultado por Ataque")

    table = add_table(slide, Inches(0.5), Inches(1.5),
                      Inches(12.3), Inches(3.5), 7, 6)

    for i, h in enumerate(["Ataque", "Campanha", "Config", "Recall", "F1", "FPR"]):
        table.cell(0, i).text = h
    style_header_row(table)

    consolidated = [
        ["DDoS-ICMP", "C03-S4", "v2 / w10s / r0=0.10", "50.0%", "5.6%", "15.7%"],
        ["DDoS-SYN", "C03-S4", "v2 / w30s / r0=0.05", "61.5%", "21.6%", "36.1%"],
        ["DDoS-TCP", "—", "Indetectável", "0.0%", "0.0%", "—"],
        ["Mirai", "C03-S4", "v1 / w10s / r0=0.10", "46.2%", "23.1%", "15.5%"],
        ["Recon", "C03-S4", "v2 / w10s / r0=0.05", "49.1%", "43.7%", "12.9%"],
        ["Benigno (FPR)", "C02-S1", "flow-level / r0=0.10", "—", "—", "3.5%"],
    ]
    for r, row_data in enumerate(consolidated):
        for c, val in enumerate(row_data):
            bold = (r == 4)  # Highlight Recon row
            color = GREEN if val == "43.7%" else (RED if val == "0.0%" or val == "Indetectável" else BLACK)
            set_cell(table, r+1, c, val, bold=bold, color=color)

    # Highlight box
    shape = slide.shapes.add_shape(5, Inches(0.5), Inches(5.3),
                                   Inches(12.3), Inches(1.8))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    shape.line.fill.background()

    tf = add_text_box(slide, Inches(0.8), Inches(5.4), Inches(11.5), Inches(1.5),
                      "Destaque: Recon-PortScan F1 = 43.7%",
                      font_size=22, bold=True, color=GREEN)
    add_paragraph(tf, "Melhor resultado não-supervisionado da dissertação (Recall 49.1%, Precision 39.4%)",
                  font_size=16, color=DARK_GRAY)
    add_paragraph(tf, "Alternativa: r0=0.15 → Precision 56.7%, FPR apenas 4.2%, F1=42.0%",
                  font_size=16, color=DARK_GRAY)

    # ===== SLIDE 10: INSIGHTS =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "5 Insights Principais")

    insights = [
        ("1. Detecção per-flow é fundamentalmente limitada",
         "Flows de ataque DDoS são estatisticamente indistinguíveis de flows benignos IoT. "
         "Mais features per-flow (17→32) não ajudam — o problema é estrutural."),
        ("2. Janelas temporais são a direção certa",
         'Mudam a pergunta de "este flow é anômalo?" para "este IP tem comportamento anômalo?". '
         "Melhorias de 10-20x no Recall."),
        ("3. Curse of dimensionality",
         "~210 vetores com 19 features → MicroTEDAclus não converge. "
         "Poucos vetores de ataque (2-55) se perdem no ruído."),
        ("4. Não existe config única ótima",
         "ICMP precisa de v2, Mirai/SYN funcionam melhor com v1. "
         "DDoS-TCP é indistinguível em qualquer configuração."),
        ("5. Resultado positivo: Recon F1=43.7%",
         "Demonstra que o pipeline Kafka → MicroTEDAclus funciona para certos tipos de ataque. "
         "Comparável com IDS não-supervisionados da literatura."),
    ]
    for i, (title, desc) in enumerate(insights):
        y = Inches(1.5) + Inches(i * 1.15)
        tf = add_text_box(slide, Inches(0.5), y, Inches(12.3), Inches(1.1),
                          title, font_size=19, bold=True, color=DARK_BLUE)
        add_paragraph(tf, desc, font_size=15, color=DARK_GRAY)

    # ===== SLIDE 11: PRÓXIMOS PASSOS =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Próximos Passos — 3 Opções")

    # Option A
    shape = slide.shapes.add_shape(5, Inches(0.3), Inches(1.5),
                                   Inches(4.0), Inches(5.0))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xE3, 0xF2, 0xFD)
    shape.line.fill.background()

    tf = add_text_box(slide, Inches(0.5), Inches(1.6), Inches(3.6), Inches(4.8),
                      "Opção A", font_size=24, bold=True, color=MEDIUM_BLUE)
    add_paragraph(tf, "S5 + Escrita", font_size=18, bold=True, color=DARK_BLUE)
    add_paragraph(tf, "~2 + 5 semanas", font_size=14, color=DARK_GRAY)
    add_paragraph(tf, "", font_size=8)
    add_paragraph(tf, "Two-Stage Detection:", font_size=15, bold=True, color=DARK_GRAY)
    add_paragraph(tf, "Stage 1: per-flow (FPR baixo)", font_size=13, color=DARK_GRAY)
    add_paragraph(tf, "Stage 2: concentração por IP", font_size=13, color=DARK_GRAY)
    add_paragraph(tf, "", font_size=8)
    add_paragraph(tf, "Equilibra profundidade\ne segurança de prazo", font_size=14, bold=True, color=GREEN)

    # Option B
    shape = slide.shapes.add_shape(5, Inches(4.6), Inches(1.5),
                                   Inches(4.0), Inches(5.0))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
    shape.line.fill.background()

    tf = add_text_box(slide, Inches(4.8), Inches(1.6), Inches(3.6), Inches(4.8),
                      "Opção B", font_size=24, bold=True, color=GREEN)
    add_paragraph(tf, "Consolidar + Escrever", font_size=18, bold=True, color=DARK_BLUE)
    add_paragraph(tf, "~6-7 semanas", font_size=14, color=DARK_GRAY)
    add_paragraph(tf, "", font_size=8)
    add_paragraph(tf, "Resultados atuais como", font_size=15, bold=True, color=DARK_GRAY)
    add_paragraph(tf, "contribuição válida:", font_size=15, bold=True, color=DARK_GRAY)
    add_paragraph(tf, "137 exps + ablation rigoroso", font_size=13, color=DARK_GRAY)
    add_paragraph(tf, "Resultados negativos documentados", font_size=13, color=DARK_GRAY)
    add_paragraph(tf, "", font_size=8)
    add_paragraph(tf, "Menos risco de prazo\nMais tempo para revisão", font_size=14, bold=True, color=GREEN)

    # Option C
    shape = slide.shapes.add_shape(5, Inches(8.9), Inches(1.5),
                                   Inches(4.0), Inches(5.0))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0xFF, 0xF3, 0xE0)
    shape.line.fill.background()

    tf = add_text_box(slide, Inches(9.1), Inches(1.6), Inches(3.6), Inches(4.8),
                      "Opção C", font_size=24, bold=True, color=ORANGE)
    add_paragraph(tf, "S5 + S6 + Escrita", font_size=18, bold=True, color=DARK_BLUE)
    add_paragraph(tf, "~3 + 4 semanas", font_size=14, color=DARK_GRAY)
    add_paragraph(tf, "", font_size=8)
    add_paragraph(tf, "Two-Stage + Threshold", font_size=15, bold=True, color=DARK_GRAY)
    add_paragraph(tf, "adaptativo", font_size=15, bold=True, color=DARK_GRAY)
    add_paragraph(tf, "Maximiza profundidade", font_size=13, color=DARK_GRAY)
    add_paragraph(tf, "experimental", font_size=13, color=DARK_GRAY)
    add_paragraph(tf, "", font_size=8)
    add_paragraph(tf, "Maior risco de prazo\n(defesa ~maio 2026)", font_size=14, bold=True, color=RED)

    # ===== SLIDE 12: CRONOGRAMA =====
    slide = prs.slides.add_slide(blank_layout)
    set_slide_bg(slide, WHITE)
    add_title_bar(slide, "Cronograma até Defesa (~8 semanas)")

    table = add_table(slide, Inches(0.3), Inches(1.5),
                      Inches(12.7), Inches(5.0), 9, 4)

    for i, h in enumerate(["Semana", "Opção A (recomendada)", "Opção B", "Opção C"]):
        table.cell(0, i).text = h
    style_header_row(table)

    timeline = [
        ["S5 (Mar 23-29)", "S5 experimentos", "Cap. Metodologia", "S5 experimentos"],
        ["S6 (Mar 30 - Abr 5)", "S5 análise", "Cap. Resultados", "S6 experimentos"],
        ["S7 (Abr 6-12)", "Cap. Metodologia", "Cap. Discussão", "S6 análise"],
        ["S8 (Abr 13-19)", "Cap. Resultados", "Revisão orientador", "Cap. Metodologia"],
        ["S9 (Abr 20-26)", "Cap. Discussão", "Ajustes finais", "Cap. Resultados"],
        ["S10 (Abr 27 - Mai 3)", "Revisão orientador", "Preparação defesa", "Cap. Discussão"],
        ["S11 (Mai 4-10)", "Ajustes finais", "DEFESA", "Revisão orientador"],
        ["S12 (Mai 11-17)", "DEFESA", "", "DEFESA"],
    ]
    for r, row_data in enumerate(timeline):
        for c, val in enumerate(row_data):
            bold = "DEFESA" in val
            color = GREEN if "DEFESA" in val else BLACK
            set_cell(table, r+1, c, val, font_size=12, bold=bold, color=color)
            if "DEFESA" in val:
                table.cell(r+1, c).fill.solid()
                table.cell(r+1, c).fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)

    # Save
    prs.save(str(OUTPUT_PATH))
    print(f"Apresentação salva em: {OUTPUT_PATH}")
    print(f"Total de slides: {len(prs.slides)}")


if __name__ == "__main__":
    create_presentation()
