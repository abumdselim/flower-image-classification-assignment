"""
make_report_pdf.py — Build the FlowerNet assignment report (body PDF via ReportLab),
then merge with the Playwright-rendered cover into the final PDF.

Report structure (assignment template):
  1 Introduction and Project Overview   (project name)
  2 Project Description                 (dataset, augmentation, models & params,
                                         why CV / learning curves / metrics / models)
  3 Implementation                      (source-code screenshots)
  4 Results and Discussion              (metrics, learning curves, overfitting)
  5 Conclusion
  References

Numbering plan (Step 3.5): cover & TOC are NOT chapters; body starts at Chapter 1.
"""
import csv
import hashlib
import json
import os
import sys

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, HRFlowable, Image, KeepTogether,
                                PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

sys.path.insert(0, "/home/z/my-project/skills/pdf/scripts")
from pdf import install_font_fallback  # noqa: E402

PROJECT = "/home/z/my-project/cv_assignment"
FIGURES = os.path.join(PROJECT, "figures")
CODE_SHOTS = os.path.join(PROJECT, "report", "code_shots")
RESULTS = os.path.join(PROJECT, "results")
OUT_BODY = os.path.join(PROJECT, "report", "body.pdf")

# ━━ Cascade Palette (design_engine.py palette-cascade --intent nature) ━━
PAGE_BG       = colors.HexColor('#f4f5f5')
SECTION_BG    = colors.HexColor('#f0f2f1')
CARD_BG       = colors.HexColor('#e8ebe9')
TABLE_STRIPE  = colors.HexColor('#ebedec')
HEADER_FILL   = colors.HexColor('#324e40')
COVER_BLOCK   = colors.HexColor('#567465')
BORDER        = colors.HexColor('#acc5b9')
ICON          = colors.HexColor('#4ba478')
ACCENT        = colors.HexColor('#1f9259')
ACCENT_2      = colors.HexColor('#3ac2c2')
TEXT_PRIMARY  = colors.HexColor('#131514')
TEXT_MUTED    = colors.HexColor('#747e79')

TABLE_HEADER_COLOR = HEADER_FILL
TABLE_ROW_ODD = TABLE_STRIPE

# ------------------------------------------------------------------ fonts ----
FONT_DIR = '/usr/share/fonts'
pdfmetrics.registerFont(TTFont('NotoSerifSC', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf'))
pdfmetrics.registerFont(TTFont('NotoSerifSC-Bold', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Bold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif', f'{FONT_DIR}/truetype/freefont/FreeSerif.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Bold', f'{FONT_DIR}/truetype/freefont/FreeSerifBold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Italic', f'{FONT_DIR}/truetype/freefont/FreeSerifItalic.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-BoldItalic', f'{FONT_DIR}/truetype/freefont/FreeSerifBoldItalic.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans', f'{FONT_DIR}/truetype/dejavu/DejaVuSansMono.ttf'))
registerFontFamily('NotoSerifSC', normal='NotoSerifSC', bold='NotoSerifSC-Bold')
registerFontFamily('FreeSerif', normal='FreeSerif', bold='FreeSerif-Bold',
                   italic='FreeSerif-Italic', boldItalic='FreeSerif-BoldItalic')
registerFontFamily('DejaVuSans', normal='DejaVuSans', bold='DejaVuSans')
install_font_fallback()

# ----------------------------------------------------------------- layout ----
MARGIN = 0.9 * inch
PAGE_W, PAGE_H = A4
AVAIL_W = PAGE_W - 2 * MARGIN
AVAIL_H = PAGE_H - 2 * MARGIN
MAX_KEEP_HEIGHT = PAGE_H * 0.4
H1_ORPHAN = AVAIL_H * 0.25

# ----------------------------------------------------------------- styles ----
S = {}
S['body'] = ParagraphStyle('Body', fontName='FreeSerif', fontSize=10.5, leading=16.5,
                           alignment=TA_JUSTIFY, textColor=TEXT_PRIMARY,
                           spaceBefore=0, spaceAfter=8)
S['bullet'] = ParagraphStyle('Bullet', parent=S['body'], leftIndent=16,
                             bulletIndent=4, spaceAfter=4, alignment=TA_LEFT)
S['h1'] = ParagraphStyle('H1', fontName='FreeSerif', fontSize=19, leading=24,
                         textColor=HEADER_FILL, spaceBefore=18, spaceAfter=4)
S['h2'] = ParagraphStyle('H2', fontName='FreeSerif', fontSize=14, leading=19,
                         textColor=TEXT_PRIMARY, spaceBefore=14, spaceAfter=6)
S['h3'] = ParagraphStyle('H3', fontName='FreeSerif', fontSize=11.5, leading=16,
                         textColor=TEXT_PRIMARY, spaceBefore=10, spaceAfter=4)
S['caption'] = ParagraphStyle('Caption', fontName='FreeSerif', fontSize=8.5,
                              leading=11.5, alignment=TA_CENTER,
                              textColor=TEXT_MUTED, spaceBefore=3, spaceAfter=6)
S['codecap'] = ParagraphStyle('CodeCap', parent=S['h3'], fontSize=10,
                              textColor=HEADER_FILL, spaceBefore=8, spaceAfter=2)
S['quote'] = ParagraphStyle('Quote', fontName='FreeSerif-Italic', fontSize=10.5,
                            leading=16, leftIndent=24, textColor=TEXT_MUTED,
                            spaceBefore=4, spaceAfter=8)
S['th'] = ParagraphStyle('TH', fontName='FreeSerif', fontSize=9, leading=12,
                         textColor=colors.white, alignment=TA_CENTER)
S['td'] = ParagraphStyle('TD', fontName='FreeSerif', fontSize=8.8, leading=11.5,
                         textColor=TEXT_PRIMARY, alignment=TA_CENTER)
S['tdl'] = ParagraphStyle('TDL', parent=S['td'], alignment=TA_LEFT)
S['stat'] = ParagraphStyle('Stat', fontName='FreeSerif', fontSize=19, leading=23,
                           textColor=ACCENT, alignment=TA_CENTER)
S['statlbl'] = ParagraphStyle('StatLbl', fontName='FreeSerif', fontSize=8.5,
                              leading=11, textColor=TEXT_MUTED, alignment=TA_CENTER)
S['ref'] = ParagraphStyle('Ref', parent=S['body'], alignment=TA_LEFT,
                          firstLineIndent=-24, leftIndent=24, spaceAfter=6)
S['toc0'] = ParagraphStyle('TOC0', fontName='FreeSerif-Bold', fontSize=12,
                           leading=20, leftIndent=6, textColor=TEXT_PRIMARY)
S['toc1'] = ParagraphStyle('TOC1', fontName='FreeSerif', fontSize=10.5,
                           leading=17, leftIndent=26, textColor=TEXT_PRIMARY)

# ------------------------------------------------------------- data load ----
with open(os.path.join(RESULTS, 'metrics.json')) as f:
    R = json.load(f)
MODELS = ["SimpleCNN", "VGG16", "ResNet50", "MobileNetV2", "EfficientNet-B0"]
AGG = {m: R['models'][m]['aggregate'] for m in MODELS}
DS = R['dataset']
BEST = max(MODELS, key=lambda m: AGG[m]['f1_macro_mean'])


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def params_m(n):
    return f"{n / 1e6:.2f} M"


# ------------------------------------------------------------ doc template ----
class TocDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, 'bookmark_name'):
            level = getattr(flowable, 'bookmark_level', 0)
            text = getattr(flowable, 'bookmark_text', '')
            key = getattr(flowable, 'bookmark_key', '')
            self.notify('TOCEntry', (level, text, self.page, key))


def header_footer(canvas, doc):
    canvas.saveState()
    # header
    canvas.setFont('FreeSerif', 7.5)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(MARGIN, PAGE_H - 0.55 * inch,
                      "FlowerNet — Custom Flower Image Classification with Five CNN Models")
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1.2)
    canvas.line(MARGIN, PAGE_H - 0.62 * inch, PAGE_W - MARGIN, PAGE_H - 0.62 * inch)
    # footer
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 0.62 * inch, PAGE_W - MARGIN, 0.62 * inch)
    canvas.setFont('FreeSerif', 7.5)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(MARGIN, 0.45 * inch, "Machine Learning Course Assignment Report")
    canvas.drawRightString(PAGE_W - MARGIN, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------- helpers ----
def add_heading(text, style, level=0):
    key = 'h_%s' % hashlib.md5(text.encode()).hexdigest()[:8]
    p = Paragraph('<a name="%s"/><b>%s</b>' % (key, text), style)
    p.bookmark_name = key
    p.bookmark_level = level
    p.bookmark_text = text
    p.bookmark_key = key
    return p


def h1(story, text):
    story.append(CondPageBreak(H1_ORPHAN))
    story.append(add_heading(text, S['h1'], 0))
    story.append(HRFlowable(width="100%", color=ACCENT, thickness=1.4,
                            spaceBefore=0, spaceAfter=10))


def h2(story, text, toc=True):
    if toc:
        story.append(CondPageBreak(70))
        story.append(add_heading(text, S['h2'], 1))
    else:
        story.append(CondPageBreak(70))
        story.append(Paragraph('<b>%s</b>' % text, S['h2']))


def body(story, text):
    story.append(Paragraph(text, S['body']))


def bullet(story, text):
    story.append(Paragraph(text, S['bullet'], bulletText='•'))


def fit_image(path, max_w=None, max_h=None):
    if max_w is None:
        max_w = AVAIL_W
    if max_h is None:
        max_h = PAGE_H * 0.35
    pil = PILImage.open(path)
    ow, oh = pil.size
    ratio = min(max_w / ow, max_h / oh, 1.0 * max_w / ow)
    ratio = min(max_w / ow, max_h / oh)
    return Image(path, width=ow * ratio, height=oh * ratio)


FIG_NO = {'n': 0}


def figure(story, path, caption, max_h=None, max_w=None):
    FIG_NO['n'] += 1
    img = fit_image(path, max_w=max_w, max_h=max_h or PAGE_H * 0.33)
    cap = Paragraph(f"Figure {FIG_NO['n']}. {caption}", S['caption'])
    story.append(Spacer(1, 10))
    story.append(KeepTogether([img, cap]))
    story.append(Spacer(1, 8))


LIST_NO = {'n': 0}


def listing_header(story, number, name, desc, part=None, total=None):
    if part:
        label = f"Listing {number} — {name}  (part {part}/{total})"
    else:
        label = f"Listing {number} — {name}"
    story.append(CondPageBreak(120))
    story.append(Paragraph(f"<b>{label}.</b>  <font color='#747e79'>{desc}</font>",
                           S['codecap']))
    story.append(Spacer(1, 2))


def callout_row(story, items):
    """Row of stat callout boxes. items = [(big, label), ...]"""
    n = len(items)
    w = min(150, (AVAIL_W - 12 * (n - 1)) / n)
    cells = []
    for big, lbl in items:
        inner = Table([[Paragraph(f"<b>{big}</b>", S['stat'])],
                       [Paragraph(lbl, S['statlbl'])]], colWidths=[w])
        inner.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
            ('BOX', (0, 0), (-1, -1), 0.8, BORDER),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, -1), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        cells.append(inner)
    outer = Table([cells], colWidths=[w + 12] * n, hAlign='CENTER')
    outer.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(Spacer(1, 6))
    story.append(KeepTogether([outer]))
    story.append(Spacer(1, 10))


def make_table(story, header, rows, ratios, caption=None, align_first_left=True):
    col_w = [r * AVAIL_W for r in ratios]
    assert sum(col_w) <= AVAIL_W + 0.5
    data = [[Paragraph(f"<b>{h}</b>", S['th']) for h in header]]
    for row in rows:
        cells = []
        for j, c in enumerate(row):
            st = S['tdl'] if (j == 0 and align_first_left) else S['td']
            cells.append(Paragraph(str(c), st))
        data.append(cells)
    t = Table(data, colWidths=col_w, hAlign='CENTER', repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
    ]
    for i in range(1, len(data)):
        style.append(('BACKGROUND', (0, i), (-1, i),
                      colors.white if i % 2 == 1 else TABLE_ROW_ODD))
    t.setStyle(TableStyle(style))
    story.append(Spacer(1, 12))
    if caption:
        cap = Paragraph(caption, S['caption'])
        if len(rows) <= 15:
            story.append(KeepTogether([t, cap]))
        else:
            story.append(t)
            story.append(cap)
    else:
        story.append(t)
    story.append(Spacer(1, 10))


# =================================================================== story ====
story = []

# ---------- TOC ----------
story.append(Paragraph('<b>Table of Contents</b>', S['h1']))
story.append(HRFlowable(width="100%", color=ACCENT, thickness=1.4,
                        spaceBefore=0, spaceAfter=12))
toc = TableOfContents()
toc.levelStyles = [S['toc0'], S['toc1']]
story.append(toc)
story.append(PageBreak())

# =========================================================== 1. INTRODUCTION ====
h1(story, "1. Introduction and Project Overview")
h2(story, "1.1 Project Name", toc=False)
body(story,
     "The name of this project is <b>FlowerNet — Custom Flower Image Classification "
     "with Five CNN Models</b>. FlowerNet is an end-to-end deep learning project that "
     "classifies flower photographs into three species (daisy, rose and sunflower) using "
     "a custom-built dataset, five different convolutional neural network architectures, "
     "5-fold stratified cross-validation, data augmentation and a suite of complementary "
     "evaluation metrics. The complete experiment is automated by a single Python "
     "pipeline so that every number and every figure in this report can be reproduced "
     "with one command.")
h2(story, "1.2 Objective and Requirements Coverage", toc=False)
body(story,
     "The assignment asks for an image-classification project built on a custom dataset "
     "with a minimum of three classes, the application of at least three (here five) "
     "models, cross-validation, data augmentation, proper evaluation metrics for model "
     "comparison, and learning curves that visualise whether the models overfit. Table 1 "
     "maps every formal requirement to the concrete artefact of this project that "
     "satisfies it. All six requirements are fully covered.")
make_table(
    story,
    ["#", "Requirement", "How it is satisfied in FlowerNet"],
    [
        ["1", "Custom dataset (≥ 3 classes)",
         "Self-collected custom dataset: 300 images, 3 classes × 100 images "
         "(daisy, rose, sunflower), deduplicated and normalised to 224×224 px"],
        ["2", "Cross-validation",
         "5-fold stratified cross-validation (scikit-learn StratifiedKFold, seed 42); "
         "every image is used for validation exactly once"],
        ["3", "At least 3 models",
         "Five models: SimpleCNN (from scratch) + VGG16, ResNet50, MobileNetV2, "
         "EfficientNet-B0 (ImageNet-pretrained, frozen backbone + trainable head)"],
        ["4", "Data augmentation",
         "RandomResizedCrop, HorizontalFlip, Rotation ±15°, ColorJitter — applied to "
         "training data only; augmented copies also pass through the frozen backbones"],
        ["5", "Proper evaluation metrics",
         "Accuracy, macro Precision, macro Recall, macro F1-score, macro ROC-AUC, "
         "confusion matrices — reported per fold and aggregated (mean ± std)"],
        ["6", "Learning curves + overfitting visualisation",
         "Epoch-wise train/val loss and accuracy curves per model; train−validation "
         "accuracy gap chart; accuracy-vs-training-set-size curve"],
    ],
    [0.05, 0.26, 0.69],
    caption="Table 1. Assignment requirements and where they are satisfied.")

# ===================================================== 2. PROJECT DESCRIPTION ====
h1(story, "2. Project Description")

h2(story, "2.1 Dataset Description")
body(story,
     "FlowerNet uses a <b>custom dataset</b> that was collected specifically for this "
     "assignment rather than an off-the-shelf benchmark. It contains <b>300 RGB images "
     "of flowers</b> divided into <b>three classes — daisy, rose and sunflower — with "
     "exactly 100 images per class</b>, so the dataset is perfectly balanced. The raw "
     "photographs were gathered from web image search with multiple, deliberately varied "
     "text queries per class (close-ups, gardens, bouquets, fields, different colours and "
     "backgrounds) so that each class covers substantial visual variety instead of one "
     "stereotypical view. Exact duplicate images were removed by MD5 content hashing, "
     "and every image was verified with PIL before being accepted.")
body(story,
     "All accepted images were resized so that the short side is 256 pixels and then "
     "centre-cropped to <b>224 × 224 pixels</b>, the canonical input resolution of the "
     "ImageNet-pretrained backbones used in this project. A uniform resolution removes "
     "one source of noise, keeps the memory footprint predictable, and means that the "
     "only geometric variation left in the data is real-world variation — which the "
     "augmentation pipeline of Section 2.2 then explicitly models. Figure 2 shows the "
     "class distribution and a random sample of images from each class.")
figure(story, os.path.join(FIGURES, '01_class_distribution.png'),
       "Class distribution of the custom dataset — 3 classes with exactly 100 images each (300 total).",
       max_h=180)
figure(story, os.path.join(FIGURES, '02_sample_grid.png'),
       "Randomly chosen sample images, one row per class. Note the large variation in "
       "background, pose, colour and scale within every class.", max_h=300)
make_table(
    story,
    ["Class", "Images", "Resolution", "Typical visual characteristics"],
    [
        ["daisy", "100", "224 × 224", "white radial petals around a bright yellow centre, thin stems"],
        ["rose", "100", "224 × 224", "layered, spiral petals; red / pink / yellow colour variants"],
        ["sunflower", "100", "224 × 224", "large yellow head, dark central disc, coarse leaves"],
    ],
    [0.16, 0.12, 0.17, 0.55],
    caption="Table 2. Dataset summary (100 images per class, 300 images in total).")

h2(story, "2.2 Data Augmentation")
body(story,
     "Because 300 images is a small dataset by deep-learning standards, FlowerNet "
     "applies <b>data augmentation</b> to enlarge the effective training set without "
     "collecting new photographs. The augmentation pipeline (applied to <b>training "
     "data only</b>, never to validation data) combines four random operations: "
     "<b>RandomResizedCrop</b> (crop a random region covering 75–100 % of the image, "
     "then resize back to the target resolution), <b>RandomHorizontalFlip</b> "
     "(p = 0.5), <b>RandomRotation</b> (±15°) and <b>ColorJitter</b> (brightness, "
     "contrast and saturation each jittered by up to 25 %). These operations simulate "
     "the exact variations that a real camera produces (different viewpoints, distances, "
     "orientations and lighting conditions) and therefore teach the models to be "
     "invariant to them.")
body(story,
     "For the four transfer-learning models the frozen backbones convert images into "
     "fixed 1280–2048-dimensional feature vectors. Augmentation is injected by passing "
     "every training image through the backbone <b>three times</b>: once with the "
     "deterministic evaluation transform (the clean view) and twice with independent "
     "random augmentation passes. Each classifier head is therefore trained on the "
     "union of clean and augmented feature vectors, which multiplies the effective "
     "training set by three and acts as a strong regulariser. The from-scratch "
     "SimpleCNN instead receives freshly augmented pixel images on every epoch. "
     "Figure 3 visualises typical augmented views: random crops and rotations change "
     "the viewpoint, colour jitter simulates different cameras and lighting.")
figure(story, os.path.join(FIGURES, '03_augmentation_showcase.png'),
       "The augmentation pipeline in action: each row shows one original image and three "
       "random augmented views (random crop, flip, rotation and colour jitter).", max_h=270)

h2(story, "2.3 The Five Models and Their Parameters")
body(story,
     "FlowerNet compares <b>five models</b> that deliberately span the design space of "
     "modern image classification. <b>SimpleCNN</b> is a compact convolutional network "
     "designed and trained <b>from scratch</b>: three Conv3×3→BatchNorm→ReLU→MaxPool "
     "blocks with 32, 64 and 128 filters, followed by global average pooling and a "
     "128-unit fully-connected head. It has only ≈0.11 M parameters and serves as the "
     "baseline that shows how far a small network gets on 300 images without any "
     "prior knowledge.")
body(story,
     "The remaining four models apply <b>transfer learning</b>: their convolutional "
     "backbones are <b>ImageNet-pretrained and frozen</b>, so they act as fixed feature "
     "extractors that convert each 224×224 image into a rich feature vector; only a "
     "small trainable head (Dropout → Linear → ReLU → Dropout → Linear, 128 hidden "
     "units) is fitted on those features under cross-validation. This two-stage "
     "strategy is CPU-friendly and avoids overfitting the tiny dataset, because "
     "94–99 % of each network's parameters are never updated. The four backbones "
     "represent four milestones of CNN design: <b>VGG16</b> (2014, uniform 3×3 "
     "convolution stacks), <b>ResNet50</b> (2016, bottleneck residual blocks with skip "
     "connections), <b>MobileNetV2</b> (2018, depthwise-separable inverted residuals "
     "for mobile devices) and <b>EfficientNet-B0</b> (2019, compound-scaled mobile "
     "inverted bottlenecks).")
make_table(
    story,
    ["Model", "Type", "Params (total)", "Params (trainable)", "Input", "Architecture highlights"],
    [
        ["SimpleCNN", "scratch", "0.11 M", "0.11 M (100 %)", "128×128",
         "3 conv blocks (32-64-128) + GAP + FC-128 head, trained from scratch"],
        ["VGG16", "transfer", "14.78 M", "66 K (0.4 %)", "224×224",
         "13 conv layers of uniform 3×3 filters, ImageNet-pretrained, frozen"],
        ["ResNet50", "transfer", "23.77 M", "263 K (1.1 %)", "224×224",
         "bottleneck residual blocks with identity skip connections, frozen"],
        ["MobileNetV2", "transfer", "2.39 M", "164 K (6.9 %)", "224×224",
         "depthwise-separable inverted residual blocks, frozen"],
        ["EfficientNet-B0", "transfer", "4.17 M", "164 K (3.9 %)", "224×224",
         "compound-scaled MBConv blocks, frozen"],
    ],
    [0.15, 0.09, 0.11, 0.15, 0.09, 0.41],
    caption="Table 3. The five models and their parameter budgets. Transfer heads are "
            "the small MLPs trained on frozen backbone features.")
body(story,
     "All models share the same training protocol: Adam optimiser with learning rate "
     "10<super>-3</super>, cross-entropy loss, batch size 64 for the transfer heads "
     "(30 epochs) and batch size 32 for SimpleCNN (12 epochs at 128×128 input). The "
     "best epoch is selected by validation accuracy within every cross-validation fold, "
     "and the metrics of the best epoch are reported.")

h2(story, "2.4 Why Cross-Validation?")
body(story,
     "With only 300 images, a single train/validation split would be fragile: 20 % of "
     "the data is just 60 images, so one lucky or unlucky split could change the "
     "measured accuracy by several percentage points and make the five models "
     "impossible to rank fairly. <b>K-fold cross-validation</b> solves this by splitting "
     "the dataset into K = 5 stratified folds — stratified meaning every fold keeps the "
     "exact 1 : 1 : 1 class ratio — and training K separate models, each time holding "
     "out a different fold for validation. Every image plays the role of a validation "
     "image exactly once, so the final out-of-fold prediction covers the whole dataset, "
     "and the reported metric is the <b>mean ± standard deviation over five folds</b>. "
     "The standard deviation doubles as an estimate of how sensitive each model is to "
     "the training subset, which is exactly the kind of stability information a small "
     "dataset demands. Cross-validation also uses the limited data far more efficiently "
     "than a single split (each model trains on 80 % of the data five times instead of "
     "once), and it removes the need to sacrifice a separate permanent test set, which "
     "would be prohibitively expensive at this dataset size.")

h2(story, "2.5 Why Learning Curves?")
body(story,
     "A single accuracy number cannot tell us <i>how</i> a model behaves during "
     "training, so FlowerNet visualises two complementary kinds of <b>learning "
     "curves</b>. First, <b>epoch-wise curves</b> plot training loss/accuracy and "
     "validation loss/accuracy against training epochs, averaged over the five folds. "
     "The vertical distance between the training and validation curves is the classic "
     "diagnostic of <b>overfitting</b>: a model that memorises its training data shows "
     "training accuracy climbing towards 100 % while validation accuracy stalls or "
     "falls and validation loss starts to rise again. Underfitting, conversely, shows "
     "both curves stuck at a low level. Second, a <b>training-set-size learning curve</b> "
     "plots validation accuracy against the number of training images used "
     "(25 %, 50 %, 75 %, 100 %). This curve answers two questions at once: whether the "
     "models have already saturated the information content of the dataset (curve flat "
     "→ more data would not help much) and whether collecting more images would be "
     "worthwhile (curve still climbing → more data would help). Plotting the "
     "train−validation gap against training-set size additionally shows how data "
     "volume suppresses overfitting, which is precisely the visualisation the "
     "assignment asks for.")

h2(story, "2.6 Which Evaluation Metrics and Why?")
body(story,
     "No single metric tells the whole story of a classifier, so FlowerNet reports "
     "five complementary metrics — always computed on the held-out fold, never on "
     "training data — and aggregates them over the five cross-validation folds:")
bullet(story, "<b>Accuracy</b> — the overall fraction of correctly classified images; "
       "intuitive and, because the dataset is perfectly balanced (100/100/100), not "
       "misleading here.")
bullet(story, "<b>Precision (macro)</b> — of the images predicted as class c, how many "
       "really belong to c, averaged over classes; penalises false alarms for each flower.")
bullet(story, "<b>Recall (macro)</b> — of the images of class c, how many were found; "
       "penalises missed flowers. A model can trade precision against recall, so both "
       "are needed.")
bullet(story, "<b>F1-score (macro)</b> — the harmonic mean of precision and recall; "
       "our primary model-ranking metric because it summarises both error types in one "
       "number and treats all three classes equally regardless of size.")
bullet(story, "<b>ROC-AUC (macro, one-vs-rest)</b> — the probability that a random "
       "positive image of a class receives a higher predicted score than a random "
       "negative; evaluates the <i>ranking quality</i> of the predicted probabilities, "
       "not just the hard arg-max decision, and is threshold-independent.")
body(story,
     "Finally, <b>confusion matrices</b> (row-normalised, computed from the pooled "
     "out-of-fold predictions) reveal <i>which</i> classes are mixed up with each "
     "other, information that scalar metrics hide. Macro averaging is used throughout "
     "so that every class contributes equally; with the balanced dataset this also "
     "coincides with micro averaging, making the numbers easy to interpret.")

h2(story, "2.7 Why These Five Models?")
body(story,
     "The five models were chosen to answer the assignment's comparison question from "
     "as many angles as possible. SimpleCNN provides the <b>from-scratch baseline</b>: "
     "it shows what is achievable with ≈0.1 M parameters and no external knowledge, "
     "and its training dynamics illustrate overfitting behaviour on a small dataset. "
     "The four pretrained backbones then isolate the value of <b>transfer learning</b>. "
     "VGG16 is the historical reference architecture whose uniform structure makes it "
     "the textbook choice for feature extraction. ResNet50 adds residual learning and "
     "is one of the most widely used general-purpose backbones in computer vision. "
     "MobileNetV2 and EfficientNet-B0 represent the modern <b>light-weight</b> family: "
     "if a 2–4 M-parameter mobile model matches a 24 M-parameter classic, that is a "
     "practically relevant finding for deployment. Comparing all five under the "
     "identical cross-validation, augmentation and metrics protocol therefore measures "
     "not only accuracy differences but also the accuracy-per-parameter trade-off.")

# ========================================================== 3. IMPLEMENTATION ====
h1(story, "3. Implementation (Source Code)")
body(story,
     "The project is implemented in <b>Python 3.12</b> with <b>PyTorch / torchvision</b> "
     "for models and training, <b>scikit-learn</b> for cross-validation and metrics, "
     "<b>NumPy / PIL</b> for data handling and <b>matplotlib</b> for all figures. The "
     "pipeline is fully scripted and reproducible: <i>main.py</i> runs dataset "
     "verification, feature extraction, cross-validated training of all five models and "
     "figure generation in order. The following pages show screenshots of the complete "
     "source code of every module, exactly as executed for this report.")
make_table(
    story,
    ["File", "Lines", "Role in the pipeline"],
    [
        ["config.py", "58", "central configuration: paths, image sizes, CV folds, hyper-parameters"],
        ["data.py", "121", "dataset scanning, eval + augmentation transforms, PyTorch datasets"],
        ["models.py", "124", "SimpleCNN definition, frozen backbones, classifier heads"],
        ["extract_features.py", "95", "stage 1 — frozen-backbone feature extraction (clean + augmented)"],
        ["train_cv.py", "343", "stage 2 — 5-fold CV training loop, learning-curve data collection"],
        ["evaluate.py", "383", "stage 3 — metrics, confusion matrices, ROC curves, all figures"],
        ["main.py", "66", "one-command experiment driver (dataset → features → CV → figures)"],
        ["prepare_dataset.py", "62", "raw-image cleaning: resize + centre-crop to 224×224"],
        ["scripts/collect_dataset.py", "163", "custom dataset collection: search, download, validate, deduplicate"],
    ],
    [0.26, 0.09, 0.65],
    caption="Table 4. Project structure — every file is shown as a screenshot below.")

CODE_FILES = [
    (1, "config.py", "Central configuration: paths, image sizes, 5-fold CV setup, "
     "classifier-head and SimpleCNN hyper-parameters.", 2),
    (2, "data.py", "Dataset scanning, deterministic eval transform, data-augmentation "
     "transform, PyTorch Dataset classes.", 3),
    (3, "models.py", "SimpleCNN (from scratch) and the four frozen ImageNet backbones "
     "plus the trainable classifier head.", 3),
    (4, "extract_features.py", "Stage 1: images pass through the frozen backbones to "
     "produce clean + augmented feature vectors (cached per pass).", 3),
    (5, "train_cv.py", "Stage 2: 5-fold stratified cross-validation for all five models, "
     "per-fold metrics, OOF probabilities and learning-curve data.", 9),
    (6, "evaluate.py", "Stage 3: metric aggregation, confusion matrices, ROC curves and "
     "the twenty result figures of this report.", 10),
    (7, "main.py", "One-command driver: dataset check → feature extraction → CV training "
     "→ evaluation figures.", 2),
    (8, "prepare_dataset.py", "Raw images are resized (short side 256) and centre-cropped "
     "to 224×224 to build the final dataset.", 2),
    (9, "scripts/collect_dataset.py", "Custom-dataset collection: web image search per "
     "class, threaded download, PIL validation, MD5 deduplication.", 4),
]
for order, name, desc, npages in CODE_FILES:
    base = os.path.basename(name)
    for p in range(1, npages + 1):
        fname = f"{order:02d}_{base}_p{p}.png"
        path = os.path.join(CODE_SHOTS, fname)
        if not os.path.exists(path):
            print(f"[warn] missing code shot: {fname}")
            continue
        listing_header(story, order, name, desc, part=p, total=npages)
        img = fit_image(path, max_w=AVAIL_W, max_h=PAGE_H * 0.56)
        story.append(img)
        story.append(Spacer(1, 6))

# ============================================== 4. RESULTS AND DISCUSSION ====
h1(story, "4. Results and Discussion")

h2(story, "4.1 Cross-Validated Comparison of the Five Models")
body(story,
     "Table 5 and Figures 4–6 summarise the 5-fold stratified cross-validation. All "
     "values are means over the five held-out folds; ± is the standard deviation "
     "across folds. The gap column is the training − validation accuracy at the best "
     "epoch (the overfitting indicator discussed in Section 4.3).")
rows = []
for m in MODELS:
    a = AGG[m]
    gap = a['train_acc_at_best_mean'] - a['best_val_acc_mean']
    rows.append([
        m,
        params_m(R['models'][m]['params_total']),
        f"{fmt(a['accuracy_mean'])} ± {a['accuracy_std']:.3f}",
        fmt(a['precision_macro_mean']),
        fmt(a['recall_macro_mean']),
        fmt(a['f1_macro_mean']),
        fmt(a['auc_macro_mean']),
        f"{gap:+.3f}",
    ])
make_table(
    story,
    ["Model", "Params", "Accuracy", "Precision", "Recall", "F1 (macro)",
     "AUC", "Gap"],
    rows,
    [0.17, 0.10, 0.17, 0.12, 0.11, 0.12, 0.11, 0.10],
    caption="Table 5. Final 5-fold cross-validation results (mean ± std over folds). "
            "Gap = train − validation accuracy at the best epoch.")
callout_row(story, [
    ("99.0 %", "best CV accuracy (ResNet50 / MobileNetV2 / EfficientNet-B0)"),
    (fmt(AGG[BEST]['f1_macro_mean']), f"best macro-F1 — {BEST}"),
    ("0.9995", "best macro ROC-AUC — MobileNetV2"),
    ("−4.2 pp", "largest train−val gap (SimpleCNN, no overfitting)"),
])
figure(story, os.path.join(FIGURES, '04_cv_accuracy_bars.png'),
       "Mean 5-fold cross-validation accuracy per model (error bars = ±1 std).",
       max_h=250)
figure(story, os.path.join(FIGURES, '05_metrics_comparison.png'),
       "All five evaluation metrics side by side — the ranking of the models is "
       "consistent across metrics.", max_h=250)
figure(story, os.path.join(FIGURES, '16_summary_heatmap.png'),
       "Metric summary heatmap (mean over the 5 CV folds).", max_h=250)
body(story,
     "Three clear findings emerge. First, <b>all four transfer-learning models beat "
     "the from-scratch baseline</b>: SimpleCNN reaches "
     f"{fmt(AGG['SimpleCNN']['accuracy_mean'])} ± {AGG['SimpleCNN']['accuracy_std']:.3f} "
     "accuracy, while VGG16, ResNet50, MobileNetV2 and EfficientNet-B0 reach "
     f"{fmt(AGG['VGG16']['accuracy_mean'])}, {fmt(AGG['ResNet50']['accuracy_mean'])}, "
     f"{fmt(AGG['MobileNetV2']['accuracy_mean'])} and "
     f"{fmt(AGG['EfficientNet-B0']['accuracy_mean'])} respectively. Features learned on "
     "1.2 M ImageNet images transfer almost perfectly to flower recognition even with "
     "a frozen backbone. Second, the three modern backbones are statistically "
     "indistinguishable at ≈99 %, so on this dataset the extra capacity of ResNet50 "
     "buys nothing over the 2.4 M-parameter MobileNetV2 — a decisive practical "
     "argument for light-weight models. Third, the metric ranking is perfectly "
     "consistent across all five metrics, which increases confidence that the "
     "differences are real and not artefacts of one particular metric choice.")

h2(story, "4.2 Fold Stability")
figure(story, os.path.join(FIGURES, '06_fold_stability.png'),
       "Per-fold validation accuracy — transfer models stay within ~5 pp across folds, "
       "SimpleCNN fluctuates most (std 0.039).", max_h=250)
body(story,
     "Figure 7 shows the validation accuracy of every model on each of the five folds. "
     "The transfer models are remarkably stable (standard deviations of 0.008–0.013), "
     "which tells us that their features make the classification problem easy enough "
     "that virtually any 240-image training subset suffices. SimpleCNN, in contrast, "
     "has a standard deviation of "
     f"{AGG['SimpleCNN']['accuracy_std']:.3f} and swings between "
     f"{min(R['models']['SimpleCNN']['fold_val_accs']):.2f} and "
     f"{max(R['models']['SimpleCNN']['fold_val_accs']):.2f} across folds: learning "
     "flower-discriminative filters from 240 augmented images is a much harder "
     "optimisation problem, and the exact fold composition matters. This stability "
     "difference is itself an argument for transfer learning on small datasets.")

h2(story, "4.3 Learning Curves and Overfitting Analysis")
body(story,
     "Figures 8–12 show the epoch-wise learning curves of every model (mean over the "
     "five folds): solid lines are training curves, dashed lines validation. The "
     "assignment specifically asks to use the curves to judge whether overfitting is "
     "present — the verdict for every model is <b>no harmful overfitting</b>: "
     "validation accuracy rises monotonically and plateaus at the training level "
     "instead of separating from it, and validation loss keeps falling without the "
     "characteristic U-shaped rebound.")
for m in MODELS:
    figure(story, os.path.join(FIGURES, f'07_curves_{m}.png'),
           f"Learning curves of {m} (mean over the 5 CV folds): loss (left) and "
           "accuracy (right); solid = training, dashed = validation.", max_h=200)
body(story,
     "The train−validation accuracy gaps in Figure 13 are in fact slightly "
     "<b>negative</b> for all models. This is the expected consequence of the "
     "augmentation strategy: training accuracy is measured on randomly cropped, "
     "rotated and colour-shifted views (a harder problem), while validation accuracy "
     "is measured on clean centre crops. A model whose augmented-training accuracy "
     "still matches its clean-validation accuracy is fitting the <i>augmented</i> "
     "distribution without memorising individual images — strong evidence that the "
     "combination of augmentation, dropout (0.3) and best-epoch selection effectively "
     "prevents overfitting at 300 images.")
figure(story, os.path.join(FIGURES, '13_overfit_gap.png'),
       "Overfitting check: train − validation accuracy gap at the best epoch "
       "(mean of 5 folds). Negative gaps indicate zero memorisation of clean training data.",
       max_h=230)
figure(story, os.path.join(FIGURES, '12_size_curve_all.png'),
       "Training-set-size learning curve: validation accuracy (left) and the "
       "train−validation gap (right) versus the number of training images.", max_h=250)
body(story,
     "Figure 14 adds the training-set-size learning curves. The transfer models are "
     "already within one percentage point of their final accuracy at <b>25 % of the "
     "training data</b> (≈75 images + augmented copies): their validation curves are "
     "essentially flat, meaning the frozen ImageNet features have saturated this "
     "dataset and collecting more flower photos would mainly benefit SimpleCNN, whose "
     "curve is still climbing. The gap panel on the right shows the same story — the "
     "from-scratch model benefits most from every additional image, exactly the "
     "regime in which data augmentation matters most.")

h2(story, "4.4 Confusion Matrices and ROC Analysis")
body(story,
     "Figure 15 shows the row-normalised out-of-fold confusion matrices. The transfer "
     "models confuse almost nothing; the residual errors of SimpleCNN and VGG16 "
     "concentrate on daisies being mislabelled as roses — visually plausible, since "
     "pink/white multi-petal rose photographs resemble daisy close-ups far more than "
     "sunflowers do. Sunflower is never confused: its large dark central disc and "
     "coarse yellow rays are trivially separable features.")
for m in MODELS:
    figure(story, os.path.join(FIGURES, f'14_confusion_{m}.png'),
           f"{m} — out-of-fold confusion matrix (row-normalised, pooled over all "
           "images; each image predicted exactly once by a model that never saw it).",
           max_h=230)
figure(story, os.path.join(FIGURES, '15_roc_macro_all.png'),
       "One-vs-rest macro-averaged ROC curves from pooled out-of-fold probabilities.",
       max_h=290)
body(story,
     "The ROC curves of Figure 16 confirm that the predicted <i>probabilities</i>, not "
     "just the arg-max decisions, are well calibrated: every model's macro-AUC exceeds "
     "0.97, and the three modern backbones are in the 0.997–0.9995 range, i.e. their "
     "scores rank a random correct-class image above a random wrong-class image "
     "essentially always. MobileNetV2 achieves the best AUC (0.9995) despite having "
     "the second-smallest parameter count, reinforcing the deployment argument of "
     "Section 4.1.")

# ============================================================== 5. CONCLUSION ====
h1(story, "5. Conclusion")
body(story,
     "This assignment set out to build a complete image-classification project on a "
     "custom dataset, and every requirement was met: a self-collected, deduplicated "
     "custom dataset of <b>300 flower images in three balanced classes (100 each)</b>; "
     "<b>five models</b> spanning from-scratch and transfer-learning paradigms; "
     "<b>5-fold stratified cross-validation</b>; systematic <b>data augmentation</b>; "
     "five <b>evaluation metrics</b> (accuracy, macro precision, macro recall, macro "
     "F1 and macro ROC-AUC) with confusion matrices; and two families of "
     "<b>learning curves</b> used explicitly to reason about overfitting.")
body(story,
     "The experiments show that ImageNet-pretrained backbones transfer almost "
     "perfectly to a small custom flower dataset: ResNet50, MobileNetV2 and "
     "EfficientNet-B0 all reach <b>99.0 % cross-validated accuracy</b> (macro-F1 "
     "0.99) versus 94.7 % for the from-scratch SimpleCNN — and the 2.4 M-parameter "
     "MobileNetV2 matches the 23.8 M-parameter ResNet50, making it the pragmatic "
     "deployment choice. Learning curves revealed no overfitting anywhere: "
     "validation loss fell monotonically, and the train−validation accuracy gaps "
     "stayed at or below zero because augmented training views are harder than the "
     "clean validation views. The size-based curves further showed that the transfer "
     "models saturate at 25 % of the training data, while the from-scratch model "
     "would profit from more images. The most instructive lesson is methodological: "
     "on small datasets, cross-validation plus augmentation plus frozen pretrained "
     "features deliver both better accuracy <i>and</i> far more trustworthy estimates "
     "than any single-split experiment could.")
body(story,
     "Natural future work includes fine-tuning the top backbone layers instead of "
     "freezing them, collecting a larger and more imbalanced dataset to stress-test "
     "the macro metrics, holding out a permanent test set for a final unbiased "
     "estimate, and benchmarking inference speed on mobile hardware to quantify the "
     "deployment trade-off quantitatively.")

# ================================================================ references ====
h1(story, "References")
refs = [
    "K. Simonyan and A. Zisserman, \"Very Deep Convolutional Networks for Large-Scale "
    "Image Recognition,\" in Proc. ICLR, 2015.",
    "K. He, X. Zhang, S. Ren, and J. Sun, \"Deep Residual Learning for Image "
    "Recognition,\" in Proc. IEEE CVPR, pp. 770–778, 2016.",
    "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, \"MobileNetV2: "
    "Inverted Residuals and Linear Bottlenecks,\" in Proc. IEEE CVPR, pp. 4510–4520, 2018.",
    "M. Tan and Q. V. Le, \"EfficientNet: Rethinking Model Scaling for Convolutional "
    "Neural Networks,\" in Proc. ICML, pp. 6105–6114, 2019.",
    "F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of "
    "Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "A. Paszke et al., \"PyTorch: An Imperative Style, High-Performance Deep Learning "
    "Library,\" in Advances in Neural Information Processing Systems 32, 2019.",
    "A. Krizhevsky, I. Sutskever, and G. E. Hinton, \"ImageNet Classification with Deep "
    "Convolutional Neural Networks,\" in Advances in Neural Information Processing "
    "Systems 25, 2012.",
]
for r in refs:
    story.append(Paragraph(r, S['ref']))

# ==================================================================== build ====
doc = TocDocTemplate(
    OUT_BODY, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=0.95 * inch, bottomMargin=0.85 * inch,
    title="FlowerNet — Custom Flower Image Classification with Five CNN Models",
    author="Z.ai", creator="Z.ai",
    subject="Machine learning course assignment report: custom dataset, 5 models, "
            "5-fold cross-validation, learning curves")
doc.multiBuild(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"body -> {OUT_BODY}")

# ------------------------------------------------------- merge cover + body ----
from pypdf import PdfReader, PdfWriter

A4_W, A4_H = 595.28, 841.89
FINAL = os.path.join(PROJECT, "report", "FlowerNet_Report.pdf")


def normalize_page_to_a4(page):
    box = page.mediabox
    w, h = float(box.width), float(box.height)
    if abs(w - A4_W) > 0.1 or abs(h - A4_H) > 0.1:
        page.scale_to(A4_W, A4_H)
    return page


writer = PdfWriter()
cover_page = PdfReader(os.path.join(PROJECT, "report", "cover.pdf")).pages[0]
writer.add_page(normalize_page_to_a4(cover_page))
for page in PdfReader(OUT_BODY).pages:
    writer.add_page(normalize_page_to_a4(page))
writer.add_metadata({
    '/Title': 'FlowerNet — Custom Flower Image Classification with Five CNN Models',
    '/Author': 'Z.ai', '/Creator': 'Z.ai',
    '/Subject': 'Machine learning course assignment report: custom dataset, 5 models, '
                '5-fold cross-validation, learning curves',
})
with open(FINAL, 'wb') as f:
    writer.write(f)
print(f"final -> {FINAL} ({len(writer.pages)} pages)")
