import os
import sys
import subprocess

def install_and_import(package):
    try:
        import pptx
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        import pptx
        return pptx

pptx = install_and_import('python-pptx')
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def add_slide_with_accent(prs, title_text):
    slide_layout = prs.slide_layouts[6] # Blank
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 77, 77) # Teal
    
    # Orange accent bar
    line = slide.shapes.add_shape(6, Inches(0.5), Inches(1), Inches(1.5), Inches(0.05)) # Rectangle
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(255, 102, 77) # Orange
    line.line.visible = False
    
    return slide

def add_column_text(slide, x, y, w, h, header, content_points):
    # Header
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.5))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = header
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 102, 102)
    
    # Underline for header
    line = slide.shapes.add_shape(6, Inches(x), Inches(y + 0.5), Inches(w), Inches(0.01))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(200, 200, 200)
    line.line.visible = False
    
    # Content
    content_box = slide.shapes.add_textbox(Inches(x), Inches(y + 0.6), Inches(w), Inches(h - 0.6))
    tf = content_box.text_frame
    tf.word_wrap = True
    for point in content_points:
        p = tf.add_paragraph()
        p.text = point
        p.font.size = Pt(16)
        p.level = 0
        p.space_before = Pt(8)

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.33) # Widescreen
    prs.slide_height = Inches(7.5)

    # 1. Title Slide
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Main heading
    title = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(12.33), Inches(1.5)) # Wider width
    tf = title.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Stretching ESCNet: Frequency-Domain Feature Enhancement\nand Bayesian Uncertainty Estimation"
    p.font.size = Pt(32) # Smaller so it perfectly fits
    p.font.bold = True
    p.font.color.rgb = RGBColor(40, 60, 150) # Dark Blue
    p.alignment = PP_ALIGN.CENTER
    
    subtitle = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(0.8))
    tf = subtitle.text_frame
    p = tf.paragraphs[0]
    p.text = "for Camouflaged Object Detection"
    p.font.size = Pt(24)
    p.font.italic = False
    p.font.color.rgb = RGBColor(0, 150, 130) # Green Matching Reference
    p.alignment = PP_ALIGN.CENTER

    # Info block
    info = slide.shapes.add_textbox(Inches(2), Inches(3.8), Inches(9.33), Inches(3))
    tf = info.text_frame
    
    def add_center_para(tf, text, size=18, bold=False):
        p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(size)
        p.font.bold = bold
        p.alignment = PP_ALIGN.CENTER
        return p

    add_center_para(tf, "Prince Kumar Dubey (230101096)", 22, True)
    tf.add_paragraph().space_before = Pt(20) # Spacer
    add_center_para(tf, "Under the supervision of:", 16)
    add_center_para(tf, "Dr. N Kishorjit Singh", 18, True)
    tf.add_paragraph().space_before = Pt(20) # Spacer
    add_center_para(tf, "Department of CSE", 16)
    add_center_para(tf, "IIIT Manipur", 16)
    tf.add_paragraph().space_before = Pt(20) # Spacer
    add_center_para(tf, "April 2026", 18, True)

    # 2. Introduction, Objective & Contribution
    slide = add_slide_with_accent(prs, "Introduction, Objective & Contribution")
    add_column_text(slide, 0.5, 1.5, 4, 5, "Introduction", [
        "Camouflage breaks standard visual assumptions",
        "Targets match surroundings grain-by-grain",
        "Critical for medical (polyps) and security uses",
        "Manual inspection is slow and subjective"
    ])
    add_column_text(slide, 4.7, 1.5, 4, 5, "Objective", [
        "Build an automated detection pipeline for camouflaged targets",
        "Engineer Frequency Enhancement Module (FEM)",
        "Train a Bayesian decoder for aleatoric uncertainty",
        "Integrate reliability-based explanations for every prediction"
    ])
    add_column_text(slide, 8.9, 1.5, 4, 5, "Key Contributions", [
        "Dual-domain feature extraction using FFT-based frequency branch",
        "Uncertainty-aware segmentation head (log_var output)",
        "State-of-the-art results on COD10K (0.860 S-measure)",
        "End-to-end deployable dashboard with overlay visualizations"
    ])

    # 3. Background & Existing Work (4 quadrants)
    slide = add_slide_with_accent(prs, "Background & Existing Work")
    
    # Q1: Top Left
    add_column_text(slide, 0.5, 1.5, 6, 2.5, "What Others Have Done (COD10K)", [
        "Chanu et al. (2023): Reported 99.20% acc on spatial only",
        "Chen et al. (2023): DWT + Random Forest approach",
        "Most methods rely purely on RGB intensity"
    ])
    
    # Q2: Top Right
    add_column_text(slide, 6.8, 1.5, 6, 2.5, "Why Our Project Is Needed", [
        "Camouflage detection is a high-stakes task (medical/defense)",
        "Single-classifier approaches are brittle to noise",
        "Spatial context alone misses fine-grained texture cues"
    ])
    
    # Q3: Bottom Left
    add_column_text(slide, 0.5, 4.3, 6, 2.5, "Gaps in These Works", [
        "Reliance on spatial domain only; frequency not combined",
        "Narrow pipeline (single extractor -> single classifier)",
        "No per-prediction confidence/reliability scores"
    ])
    
    # Q4: Bottom Right (Address Gaps - with Teal highlight)
    # Header badge for Q4
    badge = slide.shapes.add_shape(6, Inches(6.8), Inches(4.3), Inches(6), Inches(0.5))
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(175, 225, 225) # Light Teal
    badge.line.visible = False
    
    tf = badge.text_frame
    p = tf.paragraphs[0]
    p.text = "How Bayesian & FFT Addresses Gaps"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 77, 77)
    
    content_box = slide.shapes.add_textbox(Inches(6.8), Inches(4.9), Inches(6), Inches(2))
    tf = content_box.text_frame
    for point in [
        "Gap 1: spatial only -> FFT capture high-freq textures",
        "Gap 2: single classifier -> PVT-v2 transformer depth",
        "Gap 3: black box outputs -> Bayesian Aleatoric Uncertainty"
    ]:
        p = tf.add_paragraph()
        p.text = point
        p.font.size = Pt(16)
        p.space_before = Pt(8)

    # 4. Proposed System & Architecture (Flowchart)
    slide = add_slide_with_accent(prs, "Proposed System & Architecture")
    
    # Left column: System Architecture Subtitle
    add_column_text(slide, 0.5, 1.2, 5.5, 5, "System Architecture", [])
    
    def add_flow_box(slide, x, y, w, h, text, color, label=None):
        shape = slide.shapes.add_shape(6, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.color.rgb = RGBColor(100,100,100)
        tf = shape.text_frame
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0,0,0)
        p.alignment = PP_ALIGN.CENTER
        if label:
            p2 = tf.add_paragraph()
            p2.text = label
            p2.font.size = Pt(10)
            p2.font.color.rgb = RGBColor(50,50,50)
            p2.alignment = PP_ALIGN.CENTER
        return shape

    def add_arrow(slide, start_shape, end_shape, from_edge=3, to_edge=1):
        from pptx.enum.shapes import MSO_CONNECTOR
        connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 0, 0, 0, 0)
        try:
            connector.begin_connect(start_shape, from_edge)
            connector.end_connect(end_shape, to_edge)
        except Exception:
            pass # fall back cleanly if connection index fails

    # Pastel colors perfectly matching presentation theme
    c_blue = RGBColor(214, 234, 248)
    c_green = RGBColor(213, 245, 227)
    c_orange = RGBColor(250, 215, 161)
    c_yellow = RGBColor(252, 243, 207)
    c_pink = RGBColor(250, 219, 216)
    c_purple = RGBColor(232, 218, 239)
    c_cyan = RGBColor(209, 242, 235)

    # Main Center Column (Natively drawn to match PPT font)
    box_w = 2.0
    box_h = 0.5
    start_x = 2.5
    start_y = 1.6
    gap = 0.75
    offset = 2.2
    
    s_in = add_flow_box(slide, start_x, start_y, box_w, box_h, "Input Image x", c_blue, "(COD10K Dataset)")
    s_pvt = add_flow_box(slide, start_x, start_y+gap, box_w, box_h, "PVT-v2-B5 Backbone", c_green, "(Encoder)")
    s_asa = add_flow_box(slide, start_x, start_y+2*gap, box_w, box_h, "ASA Adaptors", c_orange, "(Alignment)")
    s_fuse = add_flow_box(slide, start_x, start_y+3*gap, box_w, box_h, "Fused Features", c_yellow, "(Spatial + Freq)")
    s_dec = add_flow_box(slide, start_x, start_y+4*gap, box_w, box_h, "4-Scale Decoder", c_pink, "(Gated Fusion)")
    s_mask = add_flow_box(slide, start_x, start_y+5*gap, box_w, box_h, "Predicted Masks", c_purple, "{m1, ..., m4}")
    s_loss = add_flow_box(slide, start_x, start_y+6*gap, box_w, box_h, "Total Loss L_total", c_cyan, "(IoU+BCE+NLL)")

    # Side Columns
    s_fft = add_flow_box(slide, start_x - offset, start_y+2*gap, box_w-0.5, box_h, "FFT Branch", c_cyan, "(Freq Cues)")
    s_edge = add_flow_box(slide, start_x + offset, start_y+3*gap, box_w-0.5, box_h, "AETP Edge", c_orange, "(Def. Conv)")
    s_unc = add_flow_box(slide, start_x + offset, start_y+5*gap, box_w-0.5, box_h, "Log-Var Head", c_cyan, "(Uncertainty)")

    # Arrows
    add_arrow(slide, s_in, s_pvt)
    add_arrow(slide, s_pvt, s_asa)
    add_arrow(slide, s_asa, s_fuse)
    add_arrow(slide, s_fuse, s_dec)
    add_arrow(slide, s_dec, s_mask)
    add_arrow(slide, s_mask, s_loss)
    
    # Side straight arrows to avoid overlap
    add_arrow(slide, s_pvt, s_fft, 3, 1)    # Bottom PVT -> Top FFT
    add_arrow(slide, s_fft, s_asa, 2, 0)    # Right FFT -> Left ASA
    add_arrow(slide, s_fuse, s_edge, 2, 0)  # Right Fuse -> Left Edge
    add_arrow(slide, s_mask, s_unc, 2, 0)   # Right Mask -> Left Unc

    # Right column: Working of the Pipeline & Tools
    add_column_text(slide, 6.5, 1.2, 6.5, 5, "Working of the Pipeline", [
        "1. Input Image feeds into the PVT-v2-B5 backbone to extract global context.",
        "2. FFT Branch injects frequency-domain texture information to refine camo edges.",
        "3. ASA Adaptors map multi-scale features to a uniform 128-channel space.",
        "4. AETP Module applies deformable convolution to extract sharp edge priors.",
        "5. Gated Decoder fuses features level-by-level to recover the segmentation mask.",
        "6. Log-Var Head outputs per-pixel aleatoric uncertainty scores."
    ])

    # Tool Text
    add_column_text(slide, 6.5, 4.6, 6.5, 2, "Tools & Libraries", [
        "Language: Python 3.11, Framework: PyTorch 2.3",
        "Backbone: PVT-v2-B5 (Pretrained on ImageNet-22K)",
        "Hardware: Apple Silicon (MPS Backend)",
        "Dashboard: Flask REST API + Vanilla HTML/JS"
    ])
    
    # 5. Results & Analysis
    slide = add_slide_with_accent(prs, "Results & Analysis")

    add_column_text(slide, 0.5, 1.3, 6, 2, "Quantitative Analysis (COD10K Test)", [])

    # Insert PPTX Table (Both Rows)
    headers = ["Method", "Smeasure", "wFmeasure", "meanFm", "meanEm", "MAE", "MBA"]
    row1 = ["new_epoch_25", "860", "282", "301", "884", "030", ""]
    row2 = ["new_epoch_25_bayesian", "860", "282", "300", "883", "030", "774"]
    
    shape = slide.shapes.add_table(3, 7, Inches(0.5), Inches(1.9), Inches(5.8), Inches(1.2))
    table = shape.table
    for i, texts in enumerate([headers, row1, row2]):
        for j, text in enumerate(texts):
            cell = table.cell(i, j)
            cell.text = text
            cp = cell.text_frame.paragraphs[0]
            if j == 0:
                cp.font.size = Pt(10) # slightly smaller for long name
            else:
                cp.font.size = Pt(12)
            cp.font.color.rgb = RGBColor(0, 0, 0)
            if i == 0:
                cp.font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(220, 240, 240)

    add_column_text(slide, 0.5, 4.0, 6, 3, "Key Observations", [
        "S-Measure and MAE metrics remained entirely resilient during uncertainty integration.",
        "Bayesian approach successfully mapped Mean Boundary Accuracy (MBA=0.774), giving confidence tracking natively on blurred object seams."
    ])

    if os.path.exists('bayesian_curve.png'):
        add_column_text(slide, 6.8, 1.3, 6, 2, "Bayesian Reliability Curve", [])
        slide.shapes.add_picture('bayesian_curve.png', Inches(6.8), Inches(1.9), width=Inches(5.5))
        
        # Curve explanation
        expl = slide.shapes.add_textbox(Inches(6.8), Inches(5.4), Inches(6.0), Inches(1.5))
        tf = expl.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = "Graph Interpretation:"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 102, 102)
        p2 = tf.add_paragraph()
        p2.text = "The curve represents the empirical relationship between the model's confidence and its accuracy. The Y-axis tracks expected accuracy, and the X-axis plots raw predictive uncertainty (σ²). Ascending gradients confirm the model properly isolates low-confidence predictions to challenging camouflaged boundary regions."
        p2.font.size = Pt(13)

    # 6. Conclusion
    slide = add_slide_with_accent(prs, "Conclusion & Future Scope")
    
    # Left Column
    add_column_text(slide, 0.5, 1.3, 6, 2.5, "Summary of Work", [
        "End-to-end camouflaged object detection pipeline via PVT-v2.",
        "Extracted high-frequency texture cues using FFT for edge refinement.",
        "Aleatoric uncertainty modeled via log-variance head.",
        "Deployed as a Flask REST API with interactive result dashboard."
    ])
    
    add_column_text(slide, 0.5, 4.2, 6, 2.5, "Key Findings", [
        "0.860 S-measure, 0.030 MAE on COD10K benchmark.",
        "Bayesian head successfully captures edge ambiguity (MBA = 0.774).",
        "Dual-branch FFT cleanly isolates boundary textures vs standard CNNs."
    ])
    
    # Right Column
    add_column_text(slide, 6.8, 1.3, 6, 2.5, "Limitations", [
        "Evaluated on a single benchmark (COD10K); medical application pending.",
        "Deformable (AETP) convolutions lack native Apple Silicon support without fallback.",
        "Offline processing on static images, not sequential video streaming."
    ])
    
    add_column_text(slide, 6.8, 4.2, 6, 2.5, "Future Scope", [
        "Extend the pipeline to real-time video stream camouflage tracking.",
        "Compile network ops to CoreML/ONNX for native edge-device deployment.",
        "Incorporate semantic priors aimed at biological patterns (e.g. polyps)."
    ])

    # 7. Thank You
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title = slide.shapes.add_textbox(Inches(4), Inches(2.5), Inches(5), Inches(1))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Thank You"
    p.font.size = Pt(60)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 77, 77)
    p.alignment = PP_ALIGN.CENTER
    
    line = slide.shapes.add_shape(6, Inches(5.4), Inches(3.8), Inches(2.5), Inches(0.08))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(255, 102, 77)
    line.line.visible = False

    prs.save('ESCNet_Final_Presentation.pptx')
    print("Presentation saved as ESCNet_Final_Presentation.pptx")

if __name__ == "__main__":
    create_presentation()
