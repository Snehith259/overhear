"""Overhear - a listening assistant that shows its work."""
import gc, io, json, os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.vgg import make_layers, cfgs
import librosa
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from huggingface_hub import hf_hub_download

HF_REPO = "Snehith450/overhear-weights"    # EDIT: your Hugging Face repo name

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
ARCHS = ["resnet18", "vgg16", "cnngru", "sresnet"]
KEYS = ["ensemble"] + ARCHS
LABELS = {"ensemble": "Ensemble (all four models)", "resnet18": "ResNet-18", "vgg16": "VGG-16",
          "cnngru": "CNN-GRU", "sresnet": "SResNet (simplified)"}

BG, INK, MUTED, LINE, AMBER = "#12100E", "#EAE3D2", "#8C8576", "#2B2823", "#E8A33D"
CMAP = LinearSegmentedColormap.from_list("overhear", [BG, "#3A2712", "#9A5F14", AMBER, "#F7EACB"])
plt.rcParams["font.family"] = "monospace"


def load_config():
    try:
        with open(os.path.join(HERE, "overhear_config.json")) as f:
            return json.load(f)
    except Exception:
        return {"default": "resnet18", "cutoffs": {}, "clean_test": {}, "ensemble_clean": None}


CFG = load_config()

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
.verdict.likely { color:#CDBF9C; }
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


# ---------- the four architectures (same definitions as in training) ----------
norm = lambda x: (x - 0.5) / 0.25


def make_branch():
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.fc = nn.Identity()
    return m


def vgg_branch():
    feats = make_layers(cfgs["D"], batch_norm=True)     # VGG-16 conv part only (saves memory)
    feats[0] = nn.Conv2d(1, 64, 3, padding=1)
    return nn.Sequential(feats, nn.AdaptiveAvgPool2d(1), nn.Flatten())


class TwoStream(nn.Module):
    def __init__(self, kind, n_classes=10):
        super().__init__()
        mk = make_branch if kind == "resnet18" else vgg_branch
        self.mel, self.rp = mk(), mk()
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, n_classes))

    def forward(self, mel, rp):
        return self.head(torch.cat([self.mel(norm(mel)), self.rp(norm(rp))], dim=1))


class CNNGRU(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.mel, self.rp = vgg_branch(), vgg_branch()
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(512 + 512 + 256, n_classes))

    def forward(self, mel, rp):
        m, r = norm(mel), norm(rp)
        g, _ = self.gru(m.squeeze(1).transpose(1, 2))
        return self.head(torch.cat([self.mel(m), self.rp(r), g.mean(dim=1)], dim=1))


class SwiGLUGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.BatchNorm2d(c)
        self.expand = nn.Conv2d(c, 2 * c, 1)
        self.project = nn.Conv2d(c, c, 1)

    def forward(self, x):
        a, b = self.expand(self.norm(x)).chunk(2, dim=1)
        return x + self.project(a * F.silu(b))


class CrossFusion(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.mel_from_rp = nn.MultiheadAttention(c, heads, batch_first=True)
        self.rp_from_mel = nn.MultiheadAttention(c, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.g1, self.g2 = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))

    def forward(self, fm, fr):
        B, C, Hm, Wm = fm.shape
        _, _, Hr, Wr = fr.shape
        tm, tq = fm.flatten(2).transpose(1, 2), fr.flatten(2).transpose(1, 2)
        nm_, nr_ = self.n1(tm), self.n2(tq)
        am, _ = self.mel_from_rp(nm_, nr_, nr_)
        ar, _ = self.rp_from_mel(nr_, nm_, nm_)
        fm = fm + self.g1 * am.transpose(1, 2).reshape(B, C, Hm, Wm)
        fr = fr + self.g2 * ar.transpose(1, 2).reshape(B, C, Hr, Wr)
        return fm, fr


class SResNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.mel, self.rp = make_branch(), make_branch()
        chans = (64, 128, 256, 512)
        self.gm = nn.ModuleList([SwiGLUGate(c) for c in chans])
        self.gr = nn.ModuleList([SwiGLUGate(c) for c in chans])
        self.fuse = CrossFusion(128)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, n_classes))

    def _stem(self, net, x):
        return net.maxpool(net.relu(net.bn1(net.conv1(x))))

    def forward(self, mel, rp):
        a, b = self._stem(self.mel, norm(mel)), self._stem(self.rp, norm(rp))
        a, b = self.gm[0](self.mel.layer1(a)), self.gr[0](self.rp.layer1(b))
        a, b = self.gm[1](self.mel.layer2(a)), self.gr[1](self.rp.layer2(b))
        a, b = self.fuse(a, b)
        a, b = self.gm[2](self.mel.layer3(a)), self.gr[2](self.rp.layer3(b))
        a, b = self.gm[3](self.mel.layer4(a)), self.gr[3](self.rp.layer4(b))
        return self.head(torch.cat([a.mean((2, 3)), b.mean((2, 3))], dim=1))


def build(arch):
    if arch in ("resnet18", "vgg16"):
        return TwoStream(arch)
    return CNNGRU() if arch == "cnngru" else SResNet()


def build_loaded(arch):
    path = hf_hub_download(HF_REPO, f"overhear_{arch}.pt")
    m = build(arch)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.float().eval()


@st.cache_resource(show_spinner="Loading model...", max_entries=1)   # keeps only one model in memory
def load_cached(arch):
    return build_loaded(arch)


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
    """Models expect 5 seconds: take the loudest 5 s of long clips, pad short ones."""
    if len(wav) > CLIP:
        starts = range(0, len(wav) - CLIP + 1, SR // 2)
        best = max(starts, key=lambda s: np.sqrt(np.mean(wav[s:s + CLIP] ** 2)))
        wav = wav[best:best + CLIP]
    elif len(wav) < CLIP:
        wav = np.pad(wav, (0, CLIP - len(wav)))
    return wav.astype(np.float32)


def cam_layer(model, arch):
    if arch in ("resnet18", "sresnet"):
        return model.mel.layer4[-1]
    return model.mel[0][-1]                              # vgg16 and cnngru


def gradcam(model, arch, mel_t, rp_t, class_idx):
    """Where the Mel stream concentrated for the chosen class."""
    store = {}

    def fwd(module, inp, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = cam_layer(model, arch).register_forward_hook(fwd)
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


@st.cache_data(show_spinner=False, max_entries=24)
def analyse(wav, key):
    mel, rp = make_mel(wav), make_rp(wav)
    mel_t, rp_t = torch.tensor(mel)[None, None], torch.tensor(rp)[None, None]
    if key == "ensemble":
        plist, keep = [], None
        for a in ARCHS:                                  # one model at a time, to stay inside memory
            m = build_loaded(a)
            with torch.no_grad():
                plist.append(torch.softmax(m(mel_t, rp_t), dim=1)[0].numpy())
            if a == "resnet18":
                keep = m
            else:
                del m
                gc.collect()
        probs = np.mean(plist, axis=0)
        cam = gradcam(keep, "resnet18", mel_t, rp_t, int(probs.argmax()))   # attention from ResNet-18 member
        del keep
        gc.collect()
    else:
        m = load_cached(key)
        with torch.no_grad():
            probs = torch.softmax(m(mel_t, rp_t), dim=1)[0].numpy()
        cam = gradcam(m, key, mel_t, rp_t, int(probs.argmax()))
    return mel, rp, probs, cam


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
    ax.imshow(rp, origin="lower", aspect="auto", cmap=CMAP, extent=[0, 5, 0, 5])
    fig.tight_layout()
    return fig


def fig_noise(tbl):
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    style = {"Ensemble": (AMBER, "-", 2.0), "ResNet-18": (INK, "-", 1.1), "VGG-16": ("#B9AE97", "--", 1.1),
             "CNN-GRU": (MUTED, ":", 1.4), "SResNet": ("#C98A2B", "-.", 1.1)}
    for col in tbl.columns:
        c, ls, lw = style.get(col, (INK, "-", 1.1))
        ax.plot(tbl.index, tbl[col], color=c, ls=ls, lw=lw, marker="o", ms=2.5, label=col)
    ax.axhline(10, color=LINE, lw=0.8)
    ax.text(0, 12, "chance level", color=MUTED, fontsize=6)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=LINE, lw=0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=6, length=0)
    ax.set_ylabel("test accuracy %", color=MUTED, fontsize=6)
    ax.set_xlabel("background noise: clean to loudest", color=MUTED, fontsize=6)
    ax.legend(frameon=False, labelcolor=INK, fontsize=6, ncol=3)
    fig.tight_layout()
    return fig


def verdict_html(probs, sure, likely):
    order = np.argsort(probs)[::-1]
    a, b = int(order[0]), int(order[1])
    conf = probs[a]
    name, lvl = NAMES[CLASSES[a]], LEVEL[CLASSES[a]]
    head = '<div class="label">VERDICT</div>'
    if conf >= sure:
        return (head + f'<div class="verdict">{name}</div>'
                f'<div class="meta">{conf*100:.0f}% confident &nbsp;&middot;&nbsp; '
                f'alert level <b class="lvl-{lvl.lower()}">{lvl}</b></div>')
    if conf >= likely:
        return (head + f'<div class="verdict likely">Probably {name.lower()}.</div>'
                f'<div class="meta">{conf*100:.0f}% confident, below the bar for a firm answer. '
                f'Next closest: {NAMES[CLASSES[b]]} {probs[b]*100:.0f}%. '
                f'alert level <b class="lvl-{lvl.lower()}">{lvl}</b></div>')
    return (head + '<div class="verdict unsure">Not sure.</div>'
            f'<div class="meta">Closest: {name} {probs[a]*100:.0f}% &middot; '
            f'{NAMES[CLASSES[b]]} {probs[b]*100:.0f}%. '
            'Too uncertain to guess. Try a cleaner or louder recording.</div>')


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
        default_key = CFG.get("default", "resnet18")
        choice = st.radio("Model", KEYS, index=KEYS.index(default_key) if default_key in KEYS else 0,
                          format_func=lambda k: LABELS[k], label_visibility="collapsed")
        st.markdown('<div class="note">The ensemble averages all four models. It is the most careful option '
                    'and the slowest, especially on the first run.</div>', unsafe_allow_html=True)
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

    ready = False
    with right:
        if wav is None or len(wav) == 0:
            st.markdown('<div class="label" style="border-top:none;margin-top:0">VERDICT</div>'
                        '<div class="verdict unsure">Waiting for a sound.</div>', unsafe_allow_html=True)
        elif np.sqrt(np.mean(wav ** 2)) < 1e-4:
            st.markdown('<div class="label" style="border-top:none;margin-top:0">VERDICT</div>'
                        '<div class="verdict unsure">Too quiet.</div>'
                        '<div class="meta">That recording is almost silent.</div>', unsafe_allow_html=True)
        else:
            wav = prepare(wav)
            try:
                with st.spinner("Listening..."):
                    mel, rp, probs, cam = analyse(wav, choice)
                sure, likely = CFG.get("cutoffs", {}).get(choice, [0.80, 0.60])
                st.markdown(verdict_html(probs, sure, likely) + bars_html(probs), unsafe_allow_html=True)
                ready = True
            except Exception as e:
                st.error(f"Could not run the model. Check HF_REPO and the weight files. Details: {e}")

    if ready:
        st.markdown('<div class="label">EVIDENCE</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        c1.pyplot(fig_wave(wav), clear_figure=True)
        c2.pyplot(fig_mel(mel, cam), clear_figure=True)
        c3.pyplot(fig_rp(rp), clear_figure=True)
        who = "the ResNet-18 member of the ensemble" if choice == "ensemble" else LABELS[choice]
        st.markdown(f'<div class="note">Contours mark where {who} concentrated for the top guess. '
                    'They are coarse by design. In the fingerprint, dark means two moments sounded alike. '
                    'All four models read both pictures.</div>', unsafe_allow_html=True)

with tab_lab:
    st.markdown('<div class="serif2">Does a recurrence plot help, and does the architecture matter?</div>'
                '<div class="body">Zeng, Zhou and Cai (2026) fuse a Mel-spectrogram (a linear time-frequency view) '
                'with a recurrence plot (a non-linear view of how a sound repeats) for environmental sound '
                'classification. I rebuilt four model families around those two views and tested them on 10 '
                'household alert sounds from ESC-50, using its official folds so no clip appears in both '
                'training and test. Training clips also received noisy copies, using noise clips that are never '
                'used in evaluation.</div>', unsafe_allow_html=True)

    rows = "".join(f"<tr><td>{k}</td><td>{v} %</td></tr>" for k, v in CFG.get("clean_test", {}).items())
    if CFG.get("ensemble_clean") is not None:
        rows += f"<tr><td>Ensemble (all four)</td><td>{CFG['ensemble_clean']:.1f} %</td></tr>"
    st.markdown('<table class="lab"><tr><th>MODEL</th><th>CLEAN TEST ACCURACY, 80 CLIPS '
                '(MEAN +/- SD OF 3 TRAINING RUNS)</th></tr>' + rows + '</table>', unsafe_allow_html=True)

    st.markdown('<div class="serif2">Under noise</div>'
                '<div class="body">Background noise (rain, wind, engine, vacuum cleaner, washing machine) '
                'was mixed into the test clips at several levels.</div>', unsafe_allow_html=True)
    try:
        tbl = pd.read_csv(os.path.join(HERE, "noise_table.csv"), index_col=0)
        st.pyplot(fig_noise(tbl), clear_figure=True)
        noisy = tbl.drop(index="clean", errors="ignore")
        means = noisy.mean()
        spread = means.max() - means.min()
        verdict = ("The spread is small, so I treat the systems as comparable."
                   if spread < 3 else "The gap is larger than test-set noise, so it is probably real.")
        st.markdown(f'<div class="body">Averaged over the noisy levels, {means.idxmax()} scored highest '
                    f'({means.max():.1f}%) and {means.idxmin()} lowest ({means.min():.1f}%). '
                    f'The spread is {spread:.1f} points. {verdict}</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="note">noise_table.csv not found.</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="serif2">How the models differ</div>
<div class="body">
<b>ResNet-18</b> and <b>VGG-16</b>: two pretrained streams, one per picture, features joined.<br>
<b>CNN-GRU</b>: the VGG-16 streams plus a recurrent network that reads the sound as a sequence in time.<br>
<b>SResNet (simplified)</b>: ResNet-18 streams with SwiGLU gated residual blocks and cross-attention between the
two pictures. This is my own simplified reading of the paper's design, not the authors' code.<br>
<b>Ensemble</b>: averages the four models' probabilities.<br><br>
Confidence bars drive three outcomes: a firm answer, "probably", or "not sure". The two cutoffs for each system
were chosen on a validation fold with and without noise, then checked on the test fold.
</div>
<div class="serif2">Limits I would fix next</div>
<div class="body">Only 10 classes and 80 test clips (one clip is 1.25 points), so small differences between models
are not meaningful. My recurrence plot is built from sound-tone features, not the raw waveform as in the paper.
Next: repeat with more seeds and all 50 classes, and test on real phone recordings.</div>
""", unsafe_allow_html=True)
