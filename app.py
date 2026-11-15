"""Overhear - a listening assistant that shows its work."""
import io, os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import librosa
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from huggingface_hub import hf_hub_download

HF_REPO = "Snehith450/overhear-weights"    # EDIT: your Hugging Face repo name
CUTOFFS = {"mel": 0.80, "fused": 0.70}           # EDIT only if your validation tables say so

SR = 22050
CLIP = SR * 5
HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["dog", "crying_baby", "door_wood_knock", "door_wood_creaks", "clock_alarm",
           "siren", "car_horn", "glass_breaking", "crackling_fire", "thunderstorm"]
NAMES = {"dog": "Dog barking", "crying_baby": "Baby crying", "door_wood_knock": "Door knock",
         "door_wood_creaks": "Door creaking", "clock_alarm": "Alarm clock", "siren": "Siren",
         "car_horn": "Car horn", "glass_breaking": "Glass breaking",
         "crackling_fire": "Crackling fire", "thunderstorm": "Thunderstorm"}
LEVEL = {"siren": "HIGH", "glass_breaking": "HIGH", "crackling_fire": "HIGH", "crying_baby": "HIGH",
         "car_horn": "MEDIUM", "door_wood_knock": "MEDIUM", "clock_alarm": "MEDIUM", "dog": "MEDIUM",
         "thunderstorm": "LOW", "door_wood_creaks": "LOW"}

BG, INK, MUTED, LINE, AMBER = "#12100E", "#EAE3D2", "#8C8576", "#2B2823", "#E8A33D"
CMAP = LinearSegmentedColormap.from_list("overhear", [BG, "#3A2712", "#9A5F14", AMBER, "#F7EACB"])
plt.rcParams["font.family"] = "monospace"

st.set_page_config(page_title="Overhear", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background:#12100E; color:#EAE3D2; font-family:'JetBrains Mono',monospace; }
[data-testid="stHeader"], [data-testid="stToolbar"], footer, #MainMenu { display:none; }
.block-container { max-width:1180px; padding-top:2.4rem; padding-bottom:4rem; }
.wordmark { font-family:'Instrument Serif',serif; font-size:3.6rem; line-height:1; letter-spacing:-0.01em; }
.tag, .label { font-size:0.68rem; letter-spacing:0.18em; color:#8C8576; text-transform:uppercase; }
.tag { margin:0.5rem 0 1.4rem; }
.label { margin:1.3rem 0 0.6rem; border-top:1px solid #2B2823; padding-top:0.7rem; }
.verdict { font-family:'Instrument Serif',serif; font-size:3.8rem; line-height:1.05; margin:0.2rem 0 0.5rem; }
.verdict.unsure { font-style:italic; color:#8C8576; }
.meta { font-size:0.8rem; color:#8C8576; line-height:1.7; }
.lvl-high { color:#E8A33D; }
.row { display:grid; grid-template-columns:150px 1fr 44px; gap:12px; align-items:center; font-size:0.78rem; padding:5px 0; }
.bar { height:2px; background:#2B2823; display:block; }
.bar i { display:block; height:2px; background:#E8A33D; }
.pc { text-align:right; color:#8C8576; }
.note { font-size:0.7rem; color:#8C8576; line-height:1.7; margin-top:0.4rem; }
.serif2 { font-family:'Instrument Serif',serif; font-size:2rem; margin:1.6rem 0 0.4rem; }
.body { font-size:0.85rem; line-height:1.8; color:#CFC8B8; max-width:820px; }
table.lab { width:100%; border-collapse:collapse; font-size:0.8rem; margin:0.6rem 0; }
table.lab th { text-align:left; color:#8C8576; font-weight:400; letter-spacing:0.12em; font-size:0.68rem; border-bottom:1px solid #2B2823; padding:8px 4px; }
table.lab td { padding:8px 4px; border-bottom:1px solid #2B2823; }
button[data-baseweb="tab"] { font-family:'JetBrains Mono',monospace; letter-spacing:0.14em; text-transform:uppercase; font-size:0.72rem; color:#8C8576; }
button[data-baseweb="tab"][aria-selected="true"] { color:#EAE3D2; }
div[data-baseweb="tab-highlight"] { background:#E8A33D; }
[data-testid="stFileUploaderDropzone"] { background:transparent; border:1px solid #2B2823; border-radius:2px; }
.stRadio label, .stSelectbox label, .stRadio p { color:#EAE3D2 !important; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


# ---------- model ----------
def make_branch():
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.fc = nn.Identity()
    return m


class Net(nn.Module):
    def __init__(self, use_mel, use_rp, n_classes=10):
        super().__init__()
        self.use_mel, self.use_rp = use_mel, use_rp
        self.mel_net = make_branch() if use_mel else None
        self.rp_net = make_branch() if use_rp else None
        dim = 512 * (int(use_mel) + int(use_rp))
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(dim, n_classes))

    def forward(self, mel, rp):
        feats = []
        if self.use_mel: feats.append(self.mel_net((mel - 0.5) / 0.25))
        if self.use_rp: feats.append(self.rp_net((rp - 0.5) / 0.25))
        return self.head(torch.cat(feats, dim=1))


@st.cache_resource(show_spinner="Loading model...")
def load_model(key):
    fname, use_rp = ("overhear_fused.pt", True) if key == "fused" else ("overhear_mel.pt", False)
    path = hf_hub_download(HF_REPO, fname)
    m = Net(True, use_rp)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.float().eval()


# ---------- the two pictures (same recipe as training) ----------
def make_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=512,
                                         n_mels=128, fmin=20, fmax=8000)
    db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    return ((db + 80) / 80).astype(np.float32)


def make_rp(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=2048, hop_length=512)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
    D = cdist(mfcc.T, mfcc.T)
    D = D / (D.max() + 1e-8)
    return cv2.resize(D.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)


def prepare(wav):
    """Model expects 5 seconds: take the loudest 5 s of long clips, pad short ones."""
    if len(wav) > CLIP:
        starts = range(0, len(wav) - CLIP + 1, SR // 2)
        best = max(starts, key=lambda s: np.sqrt(np.mean(wav[s:s + CLIP] ** 2)))
        wav = wav[best:best + CLIP]
    elif len(wav) < CLIP:
        wav = np.pad(wav, (0, CLIP - len(wav)))
    return wav.astype(np.float32)


def analyse(model, wav):
    mel, rp = make_mel(wav), make_rp(wav)
    mel_t, rp_t = torch.tensor(mel)[None, None], torch.tensor(rp)[None, None]
    with torch.no_grad():
        probs = torch.softmax(model(mel_t, rp_t), dim=1)[0].numpy()
    return mel, rp, mel_t, rp_t, probs


def gradcam(model, mel_t, rp_t, class_idx):
    """Where the Mel branch concentrated for the chosen class."""
    store = {}
    layer = model.mel_net.layer4[-1]

    def fwd(module, inp, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = layer.register_forward_hook(fwd)
    with torch.enable_grad():
        model.zero_grad()
        out = model(mel_t, rp_t)
        out[0, class_idx].backward()
    handle.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=mel_t.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy()


# ---------- drawing ----------
def panel(title, w=4.2, h=2.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=6, length=0)
    ax.set_title(title, color=MUTED, fontsize=7, loc="left", pad=8)
    return fig, ax


def fig_wave(wav):
    fig, ax = panel("WAVEFORM")
    ax.plot(np.linspace(0, 5, len(wav)), wav, color=AMBER, lw=0.35)
    ax.set_yticks([]); ax.set_xlim(0, 5)
    fig.tight_layout()
    return fig


def fig_mel(mel, cam):
    fig, ax = panel("SPECTRUM + ATTENTION CONTOURS")
    ax.imshow(mel, origin="lower", aspect="auto", cmap=CMAP, extent=[0, 5, 0, 128])
    if cam.max() > 0:
        ax.contour(cam, levels=[0.5, 0.8], colors=["#9FD8CF", "#F7EACB"], linewidths=[0.6, 1.2],
                   origin="lower", extent=[0, 5, 0, 128])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def fig_rp(rp):
    fig, ax = panel("RECURRENCE FINGERPRINT")
    ax.imshow(rp, origin="lower", aspect="equal", cmap=CMAP, extent=[0, 5, 0, 5])
    fig.tight_layout()
    return fig


def fig_noise(tbl):
    fig, ax = plt.subplots(figsize=(7.2, 3), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    style = {"A: Mel only": (INK, "-"), "B: RP only": (MUTED, "--"), "C: Mel + RP (fused)": (AMBER, "-")}
    for col in tbl.columns:
        c, ls = style.get(col, (INK, "-"))
        ax.plot(tbl.index, tbl[col], color=c, ls=ls, lw=1.4, marker="o", ms=3, label=col)
    ax.axhline(10, color=LINE, lw=0.8)
    ax.text(0, 12, "chance level", color=MUTED, fontsize=6)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=LINE, lw=0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=6, length=0)
    ax.set_ylabel("test accuracy %", color=MUTED, fontsize=6)
    ax.set_xlabel("background noise: clean to loudest", color=MUTED, fontsize=6)
    ax.legend(frameon=False, labelcolor=INK, fontsize=6)
    fig.tight_layout()
    return fig


def verdict_html(probs, cutoff):
    order = np.argsort(probs)[::-1]
    i = int(order[0]); conf = probs[i]
    if conf >= cutoff:
        lvl = LEVEL[CLASSES[i]]
        return ('<div class="label">VERDICT</div>'
                f'<div class="verdict">{NAMES[CLASSES[i]]}</div>'
                f'<div class="meta">{conf*100:.0f}% confident &nbsp;&middot;&nbsp; '
                f'alert level <b class="lvl-{lvl.lower()}">{lvl}</b></div>')
    a, b = int(order[0]), int(order[1])
    return ('<div class="label">VERDICT</div>'
            '<div class="verdict unsure">Not sure.</div>'
            f'<div class="meta">Closest: {NAMES[CLASSES[a]]} {probs[a]*100:.0f}% &middot; '
            f'{NAMES[CLASSES[b]]} {probs[b]*100:.0f}%. '
            'Below the confidence cutoff, so Overhear will not guess. '
            'Try a cleaner or louder recording.</div>')


def bars_html(probs, k=4):
    rows = ""
    for i in np.argsort(probs)[::-1][:k]:
        p = probs[i] * 100
        rows += (f'<div class="row"><span>{NAMES[CLASSES[i]]}</span>'
                 f'<span class="bar"><i style="width:{p:.1f}%"></i></span>'
                 f'<span class="pc">{p:.0f}%</span></div>')
    return '<div class="label">LIKELIHOODS</div>' + rows


# ---------- page ----------
st.markdown('<div class="wordmark">Overhear</div>'
            '<div class="tag">A listening assistant that shows its work</div>', unsafe_allow_html=True)
tab_listen, tab_lab = st.tabs(["Listen", "Lab notes"])

with tab_listen:
    left, right = st.columns([5, 7], gap="large")
    wav = None
    with left:
        st.markdown('<div class="label" style="border-top:none;margin-top:0">MODEL</div>', unsafe_allow_html=True)
        mode = st.radio("Model", ["Mel (default)", "Fusion (Mel + recurrence plot)"], label_visibility="collapsed")
        st.markdown('<div class="label">SOURCE</div>', unsafe_allow_html=True)
        sources = ["Sample", "Upload", "Record"] if hasattr(st, "audio_input") else ["Sample", "Upload"]
        src = st.radio("Source", sources, horizontal=True, label_visibility="collapsed")
        try:
            if src == "Sample":
                pick = st.selectbox("Sample", CLASSES, index=CLASSES.index("siren"),
                                    format_func=lambda c: NAMES[c], label_visibility="collapsed")
                path = os.path.join(HERE, "samples", f"{pick}.wav")
                wav, _ = librosa.load(path, sr=SR, mono=True)
                st.audio(path)
            elif src == "Upload":
                f = st.file_uploader("Audio file", type=["wav", "mp3", "ogg", "flac"], label_visibility="collapsed")
                if f:
                    raw = f.getvalue()
                    wav, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
                    st.audio(raw, format=f.type or "audio/wav")
            else:
                rec = st.audio_input("Record a sound")
                if rec:
                    raw = rec.getvalue()
                    wav, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
        except Exception as e:
            st.error(f"Could not read that audio: {e}")
            wav = None
        st.markdown('<div class="note">Overhear analyses 5 seconds: the loudest 5 s of a long clip, '
                    'padded if shorter. Prototype for demonstration, not a certified safety device.</div>',
                    unsafe_allow_html=True)

    with right:
        if wav is None or len(wav) == 0:
            st.markdown('<div class="label" style="border-top:none;margin-top:0">VERDICT</div>'
                        '<div class="verdict unsure">Waiting for a sound.</div>', unsafe_allow_html=True)
        elif np.sqrt(np.mean(wav ** 2)) < 1e-4:
            st.markdown('<div class="label" style="border-top:none;margin-top:0">VERDICT</div>'
                        '<div class="verdict unsure">Too quiet.</div>'
                        '<div class="meta">That recording is almost silent.</div>', unsafe_allow_html=True)
        else:
            key = "fused" if mode.startswith("Fusion") else "mel"
            model = load_model(key)
            wav = prepare(wav)
            mel, rp, mel_t, rp_t, probs = analyse(model, wav)
            cam = gradcam(model, mel_t, rp_t, int(probs.argmax()))
            st.markdown(verdict_html(probs, CUTOFFS[key]) + bars_html(probs), unsafe_allow_html=True)

    if wav is not None and len(wav) > 0 and np.sqrt(np.mean(wav ** 2)) >= 1e-4:
        st.markdown('<div class="label">EVIDENCE</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        c1.pyplot(fig_wave(wav), clear_figure=True)
        c2.pyplot(fig_mel(mel, cam), clear_figure=True)
        c3.pyplot(fig_rp(rp), clear_figure=True)
        st.markdown('<div class="note">Contours mark where the model concentrated for its top guess. '
                    'They are coarse by design: the network\'s last layer is a 4 x 7 grid. '
                    'In the fingerprint, dark means two moments sounded alike. It drives the verdict '
                    'only in Fusion mode.</div>', unsafe_allow_html=True)

with tab_lab:
    st.markdown('<div class="serif2">Does a recurrence plot help?</div>'
                '<div class="body">Zeng, Zhou and Cai (2026) propose fusing a Mel-spectrogram with a recurrence '
                'plot (RP) for environmental sound classification. I tested that idea on 10 household alert '
                'sounds from ESC-50, using its official folds so no clip appears in both training and test.</div>',
                unsafe_allow_html=True)
    st.markdown("""
<table class="lab"><tr><th>MODEL</th><th>TEST ACCURACY, MEAN +/- SD OF 3 RUNS</th></tr>
<tr><td>Mel only</td><td>96.7 +/- 0.6 %</td></tr>
<tr><td>Recurrence plot only</td><td>77.5 +/- 3.7 %</td></tr>
<tr><td>Mel + recurrence plot (fused)</td><td>95.0 +/- 1.0 %</td></tr></table>
""", unsafe_allow_html=True)
    st.markdown('<div class="serif2">Under noise</div>'
                '<div class="body">I mixed in background noise (rain, wind, engine, vacuum cleaner, washing '
                'machine) at several levels and re-tested.</div>', unsafe_allow_html=True)
    try:
        st.pyplot(fig_noise(pd.read_csv(os.path.join(HERE, "noise_table.csv"), index_col=0)), clear_figure=True)
    except Exception:
        st.markdown('<div class="note">noise_table.csv not found.</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="serif2">What I conclude</div>
<div class="body">The recurrence plot carries real information on its own, but adding it gave no reliable gain,
on clean audio or in noise. The small differences are within the noise of an 80-clip test. That is why Overhear
defaults to the Mel model and offers Fusion as a switch you can try. Each model has its own "not sure" cutoff
({CUTOFFS['mel']*100:.0f}% for Mel, {CUTOFFS['fused']*100:.0f}% for Fusion). On clean test audio every confident
Mel answer was correct. In heavy noise (5 dB) confidence is less trustworthy: at a 70% cutoff the Mel model was
wrong on about 1 in 4 of its answers.</div>
<div class="serif2">Limits I would fix next</div>
<div class="body">Only 10 classes and 80 test clips (one clip is 1.25 points). The noise test used one trained
model per type. My RP is built from sound-tone features, not the raw waveform as in the paper. Next: repeat with
more seeds and all 50 classes, and test on real phone recordings.</div>
""", unsafe_allow_html=True)
