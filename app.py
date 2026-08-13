import io
import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0

st.set_page_config(
    page_title="TeaLeaf AI | Disease & Pest Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
RESNET_PATH = BASE / "models" / "best_resnet50.pth"
EFF_PATH = BASE / "models" / "best_efficientnet_b0.pth"
UNI_LOGO = BASE / "assets" / "university_logo.png"
CLASS_DIR = BASE / "assets" / "classes"

CLASSES = [
    "Algal Leaf Rust", "Brown Blight", "Gray Spot", "Healthy Leaves",
    "Helopeltis", "Looper Caterpillar", "Red Spider"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TEA_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 220">
<rect width="220" height="220" rx="48" fill="#063e24"/>
<circle cx="110" cy="95" r="69" fill="none" stroke="#b7e36b" stroke-width="5"/>
<path d="M111 164C79 141 70 107 94 76c17-22 42-30 60-31-1 25-10 52-31 70-12 10-23 15-34 20z" fill="#8bcf32"/>
<path d="M110 163c1-38 11-70 39-103" fill="none" stroke="#f1f8d7" stroke-width="6" stroke-linecap="round"/>
<path d="M82 113c16-4 29 0 40 10M91 91c11-2 19 2 27 9" fill="none" stroke="#f1f8d7" stroke-width="4" stroke-linecap="round"/>
<circle cx="158" cy="58" r="5" fill="#b7e36b"/><circle cx="178" cy="48" r="4" fill="#b7e36b"/>
<path d="M163 58h27M182 48h12" stroke="#b7e36b" stroke-width="3"/>
</svg>'''


def svg_data():
    return "data:image/svg+xml;base64," + base64.b64encode(TEA_SVG.encode()).decode()


st.markdown("""
<style>
.stApp { background:#f7faf7; }
.block-container { max-width:1540px; padding:82px 18px 22px !important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#053b21 0%,#07532d 58%,#064321 100%); min-width:238px; max-width:238px; }
section[data-testid="stSidebar"] > div { padding:20px 14px; } [data-testid="stHeader"] { z-index:999; }
section[data-testid="stSidebar"] * { color:white !important; }
.side-brand { text-align:center; }
.side-brand img { width:105px; height:105px; }
.side-brand .name { font-size:24px; font-weight:800; margin-top:3px; }
.side-brand .tag { font-size:12px; color:#d6eed9 !important; }
.side-head { color:#c7e955 !important; font-weight:800; font-size:14px; margin:20px 0 7px; letter-spacing:.3px; }
.side-link { padding:9px 10px; border-radius:9px; margin:3px 0; font-size:15px; }
.side-link.active { background:#2b8a44; font-weight:700; }
.side-info { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12); border-radius:13px; padding:12px; font-size:12px; line-height:1.65; }
.side-info b { font-size:14px; }
.header-card { background:white; border:1px solid #dce8dc; border-radius:14px; padding:11px 18px; box-shadow:0 2px 10px rgba(16,70,32,.04); }
.header-title { text-align:center; color:#0c4a26 !important; font-weight:800; font-size:30px; line-height:1.12; }
.header-sub { text-align:center; color:#2c7d41 !important; font-size:19px; margin-top:8px; }
.card { background:#fff; border:1px solid #dce8dc; border-radius:14px; padding:15px; box-shadow:0 3px 13px rgba(16,70,32,.045); height:100%; }
.card-head { color:#154b28 !important; font-weight:800; font-size:16px; margin-bottom:10px; }
.result { background:linear-gradient(135deg,#eff9ed,#fff); border:1px solid #cce3ca; border-radius:12px; padding:20px; }
.result-label { font-size:14px; color:#111 !important; }
.result-name { font-size:31px; font-weight:800; color:#2a7d25; margin:7px 0 15px; }
.conf-label { font-size:15px; color:#111 !important; }
.conf { color:#26791f; font-size:38px; font-weight:800; }
.bar { height:12px; background:#dfe4df; border-radius:20px; overflow:hidden; margin:2px 0 14px; }
.bar > div { height:100%; background:#2f8d23; border-radius:20px; }
.pill { display:inline-block; background:#e4f3df; color:#276f27; padding:6px 11px; border-radius:9px; font-weight:700; }
.compare { color:#111 !important; width:100%; border-collapse:separate; border-spacing:0; border:1px solid #e0e4e0; border-radius:10px; overflow:hidden; font-size:14px; }
.compare th { background:#fafafa !important; color:#111 !important; padding:11px; text-align:left; border-bottom:1px solid #e0e4e0; font-weight:800; }
.compare td { padding:12px 11px; border-bottom:1px solid #e6e9e6; color:#111 !important; font-weight:600; }
.compare tr:last-child td { border-bottom:none; }
.green { color:#287e2a !important; font-weight:800; }
.final { background:linear-gradient(135deg,#eff9ec,#fff); border:1px solid #c8e2c7; border-radius:11px; padding:12px; text-align:center; margin-top:11px; }
.final-small { color:#111 !important; font-size:13px; }
.final-big { color:#2a7d25; font-size:25px; font-weight:800; margin:4px; }
.upload-note { border:1.5px dashed #6f963d; border-radius:11px; padding:11px; text-align:center; color:#37613e; background:#fbfefb; }
.class-card { background:white; border:1px solid #dbe8da; border-radius:11px; padding:6px; text-align:center; }
.class-card img { width:100%; height:72px; object-fit:cover; border-radius:8px; }
.class-name { color:#111 !important; font-size:11px; font-weight:700; margin-top:4px; min-height:27px; }
.footer { background:#f0f8f0; border:1px solid #d3e6d3; border-radius:11px; padding:11px 15px; color:#355b40; font-size:13px; margin-top:13px; }
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed #6f963d !important; background:#fbfefb !important; border-radius:11px !important; color:#111 !important; } [data-testid="stFileUploaderDropzone"] * { color:#111 !important; }
[data-testid="stFileUploader"] section { padding:0 !important; }
.stMarkdown, .stText, .stCaption { color:#111; }
[data-testid="stDataFrame"] * { color:#111 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = resnet50(weights=None)
    r.fc = nn.Linear(r.fc.in_features, len(CLASSES))
    r.load_state_dict(torch.load(RESNET_PATH, map_location=device, weights_only=True))
    r.to(device).eval()

    e = efficientnet_b0(weights=None)
    e.classifier[1] = nn.Linear(e.classifier[1].in_features, len(CLASSES))
    e.load_state_dict(torch.load(EFF_PATH, map_location=device, weights_only=True))
    e.to(device).eval()
    return r, e, device


resnet_model, efficient_model, device = load_models()


def predict(model, image):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]), probs


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(
        f'<div class="side-brand"><img src="{svg_data()}"><div class="name">TeaLeaf AI</div>'
        '<div class="tag">Smart Leaf, Healthy Harvest</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-head">MODEL SELECTION</div>', unsafe_allow_html=True)
    mode = st.radio(
        "model", ["ResNet50", "EfficientNet-B0", "Compare Both Models"],
        index=2, label_visibility="collapsed"
    )
    st.markdown('<div class="side-head">UPLOAD</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-link">☁️ &nbsp; Upload Leaf Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-head">DATASET INFO</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-link">▤ &nbsp; Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-link">🍃 &nbsp; Recognizable Classes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-info"><b>ABOUT DATASET</b><br>'
        '🖼️ Total Images: 10,643<br>◉ Classes: 7<br>📍 Source: Bangladeshi Tea Gardens<br>⬛ Image Size: 224×224</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="margin-top:28px;font-size:12px;color:#d7efd9!important">Developed with TeaLeafHUB Team</div>', unsafe_allow_html=True)

# ==================== HEADER ====================
left, center, right = st.columns([1.0, 6.2, 1.0])
with left:
    st.markdown(f'<div style="text-align:center;padding-top:8px"><img src="{svg_data()}" width="82"></div>', unsafe_allow_html=True)
with center:
    st.markdown(
        '<div class="header-card"><div class="header-title">'
        'Tea Leaf Disease and Pest Classification Using Transfer Learning<br>'
        '<span style="font-size:25px">on a Novel Bangladeshi Dataset</span></div>'
        '<div class="header-sub">🍃 Tea Leaf Disease Analysis 🍃</div></div>',
        unsafe_allow_html=True,
    )
with right:
    st.image(str(UNI_LOGO), width=145)

st.write("")

# ==================== UPLOAD ====================
uploaded = st.file_uploader(
    "Upload Leaf Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded is None:
    st.markdown('<div class="upload-note">☁️ <b>Drag & Drop or Browse Files</b><br><span style="font-size:12px">Upload a tea leaf image to start analysis</span></div>', unsafe_allow_html=True)
    st.markdown("### 🍃 RECOGNIZABLE CLASSES (7)")
    cols = st.columns(7)
    for col, name in zip(cols, CLASSES):
        p = CLASS_DIR / f"{name}.jpg"
        with col:
            st.markdown('<div class="class-card">', unsafe_allow_html=True)
            if p.exists():
                st.image(str(p), use_container_width=True)
            st.markdown(f'<div class="class-name">{name}</div></div>', unsafe_allow_html=True)
    st.stop()

image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

# ==================== REAL MODEL INFERENCE ====================
preds = {}
with st.spinner("Analyzing tea leaf..."):
    if mode in ["ResNet50", "Compare Both Models"]:
        preds["ResNet50"] = predict(resnet_model, image)
    if mode in ["EfficientNet-B0", "Compare Both Models"]:
        preds["EfficientNet-B0"] = predict(efficient_model, image)

primary_name = next(iter(preds))
primary_pred, primary_conf, primary_probs = preds[primary_name]

# ==================== TOP THREE CARDS ====================
a, b, c = st.columns([1.10, 1.04, 1.00], gap="small")

with a:
    st.markdown('<div class="card"><div class="card-head">🌿 UPLOADED IMAGE</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('<div class="upload-note">📁 Drag & Drop or Browse Files<br><span style="font-size:12px">Supported formats: JPG, JPEG, PNG</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with b:
    st.markdown('<div class="card"><div class="card-head">🌿 PREDICTION RESULT</div>', unsafe_allow_html=True)
    pct = primary_conf * 100
    st.markdown(
        f'<div class="result"><div class="result-label">Predicted Disease / Pest</div>'
        f'<div class="result-name">{primary_pred} 🌿</div>'
        f'<div class="conf-label">Confidence</div><div class="conf">{pct:.2f}%</div>'
        f'<div class="bar"><div style="width:{pct:.2f}%"></div></div>'
        f'<span class="pill">⚙ Model Used &nbsp; {primary_name}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with c:
    st.markdown('<div class="card"><div class="card-head">⚖️ MODEL COMPARISON</div>', unsafe_allow_html=True)
    if len(preds) == 2:
        rows = "".join(
            f'<tr><td>{n}</td><td>{v[0]}</td><td class="green">{v[1]*100:.2f}%</td></tr>'
            for n, v in preds.items()
        )
        st.markdown(
            '<table class="compare"><thead><tr><th>Model</th><th>Prediction</th><th>Confidence</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True
        )
        final_name, final_tuple = max(preds.items(), key=lambda kv: kv[1][1])
        final_pred, final_conf, _ = final_tuple
        agree = preds["ResNet50"][0] == preds["EfficientNet-B0"][0]
        reason = "Both models agree" if agree else "Based on higher confidence"
        st.markdown(
            f'<div class="final"><div class="final-small">🏆 Final Prediction ({reason})</div>'
            f'<div class="final-big">{final_pred}</div></div>', unsafe_allow_html=True
        )
    else:
        st.markdown(f'<div class="final"><div class="final-small">Final Prediction</div><div class="final-big">{primary_pred}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== CHART + CLASS GALLERY ====================
d1, d2 = st.columns([1.75, 1.0], gap="small")

with d1:
    st.markdown('<div class="card"><div class="card-head">📊 CLASS WISE PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
    chart_model = st.selectbox("Chart model", list(preds.keys()), label_visibility="collapsed")
    probs = preds[chart_model][2]
    fig = go.Figure(go.Bar(
        x=CLASSES, y=[x * 100 for x in probs],
        text=[f"{x*100:.2f}%" for x in probs], textposition="outside",
        marker_color="#2f8f23"
    ))
    fig.update_layout(
        height=345, margin=dict(l=45, r=10, t=18, b=70),
        yaxis=dict(title="Probability (%)", range=[0, max(100, float(max(probs)*100)+8)]),
        xaxis=dict(title="Disease / Pest Classes"),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        font=dict(color="#18281b", size=11)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with d2:
    st.markdown('<div class="card"><div class="card-head">🍃 RECOGNIZABLE CLASSES (7)</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, name in enumerate(CLASSES):
        p = CLASS_DIR / f"{name}.jpg"
        with cols[i % 3]:
            st.markdown('<div class="class-card" style="margin-bottom:7px">', unsafe_allow_html=True)
            if p.exists():
                st.image(str(p), use_container_width=True)
            st.markdown(f'<div class="class-name">{i+1}. {name}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">ⓘ &nbsp; This prediction is trained by ML models through real world dataset TeaLeafHub and should be used as a decision support tool.'
    ' <span style="float:right;font-size:20px">🌿</span></div>',
    unsafe_allow_html=True,
)
