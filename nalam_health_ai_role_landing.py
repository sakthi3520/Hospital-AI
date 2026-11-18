import ast
import base64
import csv
import difflib
import hashlib
import html
import io
import re
import time
import unicodedata
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote_plus
import json
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import streamlit as st

# Optional libraries (app still works without them)
#   pip install deep-translator SpeechRecognition
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except Exception:
    HAS_TRANSLATOR = False

try:
    import speech_recognition as sr
    HAS_SR = True
except Exception:
    HAS_SR = False

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

HAS_AUDIO_INPUT = hasattr(st, "audio_input")
try:
    import inspect as _inspect
    HAS_CHAT_MIC = "accept_audio" in _inspect.signature(st.chat_input).parameters
except Exception:
    HAS_CHAT_MIC = False

# ============================================================
# NALAM HEALTH AI  (v5)
# ============================================================

st.set_page_config(page_title="Nalam Health AI", page_icon="🏥", layout="wide")

# ---- change this one line to deploy for a different hospital ----
HOSPITAL_NAME = "Nalam General Hospital"

AGE_GROUPS = ["Child (0-12)", "Teen (13-19)", "Adult (20-40)", "Middle-aged (41-60)", "Senior (60+)"]
GENDERS = ["Male", "Female", "Other / Prefer not to say"]
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "I don't know"]


def age_group_for(age):
    if age <= 12:
        return "Child (0-12)"
    if age <= 19:
        return "Teen (13-19)"
    if age <= 40:
        return "Adult (20-40)"
    if age <= 60:
        return "Middle-aged (41-60)"
    return "Senior (60+)"

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("stage", "start")
ss.setdefault("profile", {"name": None, "age": None, "age_group": None, "gender": None, "blood_group": None})
ss.setdefault("pending_text", "")
ss.setdefault("last_audio", "")
ss.setdefault("consultation_choice", None)
ss.setdefault("consult_patient_id", None)
ss.setdefault("consult_patient_response", None)
ss.setdefault("consult_booking_response", None)
ss.setdefault("consult_selected_doctor", None)
ss.setdefault("consult_selected_slot", None)
ss.setdefault("role", None)  # None -> not chosen yet, "doctor" or "patient"


def new_conversation():
    ss["messages"] = []
    ss["stage"] = "start"
    ss["profile"] = {"name": None, "age": None, "age_group": None, "gender": None, "blood_group": None}
    ss["pending_text"] = ""
    ss["consultation_choice"] = None
    ss["consult_patient_id"] = None
    ss["consult_patient_response"] = None
    ss["consult_booking_response"] = None
    ss["consult_selected_doctor"] = None
    ss["consult_selected_slot"] = None
    ss["role"] = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Background photo: put a file named  hospital_bg.jpg / .png / .webp
# in the project folder, or in  assets/  or  img/   (any of these works)
BG_SEARCH_DIRS = [BASE_DIR, BASE_DIR / "assets", BASE_DIR / "img", BASE_DIR / "images"]

# 0 = photo fully visible, 1 = photo hidden  (lower = photo more visible)
PHOTO_OVERLAY_TOP = 0.58
PHOTO_OVERLAY_BOTTOM = 0.66

FILES = {
    "symptoms": DATA_DIR / "Diseases_and_Symptoms_dataset.csv",
    "description": DATA_DIR / "description.csv",
    "precautions": DATA_DIR / "precautions.csv",
    "medication": DATA_DIR / "medications.csv",
    "diet": DATA_DIR / "diets.csv",
    "workout": DATA_DIR / "workout.csv",
}

DISEASE_COL = "diseases"

# real trained ML model (created by running train_model.py once) — optional
MODEL_FILE = DATA_DIR / "model.pkl"
# our own from-scratch neural network (created by running train_own_nn.py) — optional, takes priority
OWN_NN_FILE = DATA_DIR / "own_nn.npz"

# where every visit is recorded, and the PIN staff use to view the daily report
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "patient_log.csv"
LOG_COLS = ["timestamp", "date", "name", "age", "gender", "blood_group", "symptoms_text", "condition"]
ADMIN_PIN = "staff123"  # change this for your hospital

# name shown in the selector -> (translate code, speech-recognition code)
LANGS = {
    "English": ("en", "en-IN"),
    "தமிழ் (Tamil)": ("ta", "ta-IN"),
    "हिन्दी (Hindi)": ("hi", "hi-IN"),
    "తెలుగు (Telugu)": ("te", "te-IN"),
    "ಕನ್ನಡ (Kannada)": ("kn", "kn-IN"),
    "മലയാളം (Malayalam)": ("ml", "ml-IN"),
}

SCRIPT_RANGES = {
    "ta": (0x0B80, 0x0BFF),
    "hi": (0x0900, 0x097F),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
}


# ============================================================
# HTML HELPERS
# Streamlit markdown treats 4-space indented lines as CODE and blank
# lines end an HTML block -> strip every line before rendering.
# ============================================================

def render(markup, target=st):
    flat = "".join(line.strip() for line in markup.splitlines())
    target.markdown(flat, unsafe_allow_html=True)


def esc(value):
    return html.escape(str(value))


# ============================================================
# BACKGROUND PHOTO
# ============================================================

def find_background():
    for folder in BG_SEARCH_DIRS:
        if folder.exists():
            for p in sorted(folder.glob("hospital_bg*")):
                if p.is_file():
                    return p
    return None


@st.cache_resource(show_spinner=False)
def photo_data_uri(path_str, mtime):
    data = Path(path_str).read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "png"
    elif data[:3] == b"\xff\xd8\xff":
        mime = "jpeg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "webp"
    else:
        mime = "jpeg"
    return f"data:image/{mime};base64," + base64.b64encode(data).decode()


def background_css():
    photo = find_background()
    if photo is not None:
        uri = photo_data_uri(str(photo), photo.stat().st_mtime)
        return (
            "background-image: "
            f"linear-gradient(rgba(240,250,252,{PHOTO_OVERLAY_TOP}), rgba(232,246,249,{PHOTO_OVERLAY_BOTTOM})), "
            f"url('{uri}');"
            "background-size: cover; background-position: center; "
            "background-repeat: no-repeat; background-attachment: fixed;"
        )

    cross = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='130' height='130' viewBox='0 0 130 130'>"
        "<g fill='#087f8c' fill-opacity='0.16'>"
        "<rect x='57' y='40' width='16' height='50' rx='4'/>"
        "<rect x='40' y='57' width='50' height='16' rx='4'/>"
        "</g></svg>"
    )
    cross_b64 = base64.b64encode(cross.encode()).decode()
    return (
        "background-image: "
        f"url('data:image/svg+xml;base64,{cross_b64}'), "
        "radial-gradient(circle at top left, rgba(0,150,165,0.22), transparent 45%), "
        "linear-gradient(135deg, #f2fafb, #e3f3f6, #f4fafb);"
        "background-attachment: fixed;"
    )


# ============================================================
# CSS
# ============================================================

render(f"""
<style>
.stApp {{ {background_css()} }}
header[data-testid="stHeader"] {{ background: rgba(255,255,255,0.78); }}
.stApp {{ color: #172f35; }}
[data-testid="stAppViewContainer"] > .main {{ background-color: rgba(245,251,252,0.12); }}
.block-container {{ max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }}

/* ---------- HERO ---------- */
.hero {{
    background: linear-gradient(135deg, rgba(8,127,140,0.96), rgba(10,156,168,0.96));
    padding: 34px 38px; border-radius: 24px; color: white;
    box-shadow: 0 14px 35px rgba(0,60,80,0.30); margin-bottom: 24px;
}}
.hero-title {{ font-size: 40px; font-weight: 800; letter-spacing: 1px; }}
.hero-subtitle {{ font-size: 19px; margin-top: 8px; }}
.hero-note {{ margin-top: 14px; font-size: 14px; opacity: 0.95; }}

/* ---------- INPUT ---------- */
.input-title {{
    display: inline-block; color: #087f8c; font-size: 24px; font-weight: 800;
    margin: 26px 0 8px; background: rgba(255,255,255,0.88);
    padding: 6px 16px; border-radius: 14px;
}}
.input-hint {{
    display: inline-block; color: #33494e; font-size: 14px; margin: 0 0 10px 0;
    background: rgba(255,255,255,0.88); padding: 5px 14px; border-radius: 12px;
}}
div[data-testid="stForm"] {{
    border: none !important; padding: 12px !important;
    background: rgba(255,255,255,0.72) !important; border-radius: 18px !important;
}}
div[data-testid="stTextInput"] input {{
    border-radius: 14px; border: 1px solid #b9dde2; background: #ffffff;
    font-size: 17px; padding: 14px 16px;
}}
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {{
    width: 100%; border-radius: 12px; border: none;
    background: linear-gradient(135deg, #087f8c, #0a9ca8);
    color: white; font-weight: 700; min-height: 48px;
}}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {{ color: white; filter: brightness(1.08); }}
div[data-testid="stAudioInput"] {{
    background: rgba(255,255,255,0.85); border-radius: 16px; padding: 6px;
}}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #ffffff, #eaf6f8);
    border-right: 1px solid #d3e9ed;
}}
.side-title {{ color: #087f8c; font-size: 22px; font-weight: 800; margin-bottom: 2px; }}
.side-sub {{ color: #5b6f74; font-size: 13px; margin-bottom: 10px; }}
.side-heading {{ color: #0b5560; font-size: 15px; font-weight: 800; margin: 18px 0 8px; }}
.emergency {{
    background: #fdecec; border: 1px solid #f3bcbc; border-radius: 14px;
    padding: 12px 14px; color: #7a1f1f; line-height: 1.7; font-size: 14px;
    margin-bottom: 14px;
}}
.emergency b {{ color: #b42323; }}
.dept {{
    display: flex; justify-content: space-between; gap: 10px; font-size: 13px;
    padding: 7px 0; border-bottom: 1px dashed #cfe3e7; color: #26393d;
}}
.dept span:last-child {{ color: #087f8c; font-weight: 700; text-align: right; }}
.tip {{ font-size: 13px; color: #33494e; line-height: 1.6; margin: 4px 0; }}

/* ---------- RESULT ---------- */
.success-box {{
    background: rgba(238,250,244,0.97); border: 1px solid #bfe6d1; border-radius: 14px;
    padding: 14px 17px; color: #1f6545; margin: 6px 0 12px;
}}
.note-box {{
    background: rgba(255,255,255,0.93); border: 1px solid #cfe6ea; border-radius: 14px;
    padding: 10px 15px; color: #33494e; margin: 8px 0; line-height: 1.6; font-size: 14px;
}}
.alert-box {{
    background: rgba(253,236,236,0.97); border: 1px solid #f0b3b3; border-radius: 14px;
    padding: 14px 17px; color: #7a1f1f; margin: 14px 0; line-height: 1.6;
}}
.section-heading {{
    display: inline-block; color: #0b5560; font-size: 24px; font-weight: 800;
    margin: 22px 0 8px; background: rgba(255,255,255,0.88);
    padding: 5px 16px; border-radius: 14px;
}}
.symptom-pill {{
    display: inline-block; background: #e7f7f8; color: #087f8c; border: 1px solid #c9e9ec;
    border-radius: 999px; padding: 7px 13px; margin: 4px 5px 4px 0; font-weight: 650;
}}
.other-pill {{
    display: inline-block; background: #ffffff; color: #0b5560; border: 1px solid #cfe6ea;
    border-radius: 999px; padding: 7px 14px; margin: 4px 6px 4px 0; font-weight: 650;
}}
.other-pill small {{ color: #6c7b80; font-weight: 600; margin-left: 6px; }}

.main-card {{
    background: rgba(255,255,255,0.98); border: 1px solid #d7ebee;
    border-left: 8px solid #0b8d9a; border-radius: 22px;
    padding: 26px 28px; margin: 14px 0 10px;
    box-shadow: 0 10px 28px rgba(20,80,90,0.16);
}}
.badge {{
    display: inline-block; background: #087f8c; color: white; border-radius: 999px;
    padding: 5px 14px; font-size: 13px; font-weight: 700; margin-bottom: 10px;
}}
.main-title {{ color: #087f8c; font-size: 32px; font-weight: 800; margin-bottom: 12px; text-transform: capitalize; }}
.score-row {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }}
.score-text {{ font-weight: 700; color: #0b5560; }}
.bar {{ flex: 1; min-width: 160px; height: 10px; background: #e3f1f3; border-radius: 999px; overflow: hidden; }}
.bar-fill {{ height: 100%; background: linear-gradient(90deg, #087f8c, #35c2cc); border-radius: 999px; }}
.matched {{ color: #33494e; line-height: 1.6; }}
.section-label {{ color: #087f8c; font-size: 17px; font-weight: 750; margin: 16px 0 6px; }}
.info-box {{
    background: rgba(244,251,252,0.98); border: 1px solid #d8edef; border-radius: 14px;
    padding: 14px 16px; line-height: 1.7; color: #26393d;
}}
.info-box ul {{ margin: 0; padding-left: 20px; }}
.caption {{
    display: inline-block; color: #33494e; font-size: 13px; margin-top: 8px; line-height: 1.5;
    background: rgba(255,255,255,0.88); padding: 4px 12px; border-radius: 10px;
}}

div[data-baseweb="tab-list"] {{
    background: rgba(255,255,255,0.85); border-radius: 14px; padding: 4px 8px;
}}
button[data-baseweb="tab"] {{ font-weight: 700; font-size: 16px; }}
.role-welcome {{ background: rgba(255,255,255,0.94); border: 1px solid #cce5e9; border-radius: 22px; padding: 24px; margin: 18px 0; box-shadow: 0 10px 28px rgba(20,80,90,0.12); }}
.role-welcome h2 {{ color: #075b65; margin: 0 0 8px 0; }}
.role-welcome p {{ color: #26393d; font-size: 16px; margin: 0; }}

.warning-box {{
    background: rgba(255,248,232,0.98); border: 1px solid #f2d99a; border-radius: 16px;
    padding: 18px 20px; margin-top: 6px; color: #5f4b1d; line-height: 1.6;
}}
.footer {{
    text-align: center; color: #26393d; font-size: 13px; padding: 16px 0; margin-top: 30px;
    background: rgba(255,255,255,0.85); border-radius: 16px;
}}

/* ---------- CHAT ---------- */
div[data-testid="stChatMessage"] {{
    background: rgba(255,255,255,0.93); border-radius: 18px;
    padding: 4px 6px; margin-bottom: 6px; box-shadow: 0 4px 14px rgba(20,80,90,0.06);
}}
div[data-testid="stChatMessage"] div[data-testid="stVerticalBlock"] {{ gap: 0.35rem; }}
div[data-testid="stChatMessage"] div.element-container {{ margin-bottom: 0 !important; }}
div[data-testid="stChatInput"] textarea {{
    border-radius: 16px !important;
}}
.chip-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 16px; }}
div[data-testid="column"] div.stButton > button {{
    background: rgba(255,255,255,0.9); color: #087f8c; border: 1px solid #b9dde2;
    font-weight: 650; min-height: 40px;
}}

/* ---------- TRENDY SIDEBAR ---------- */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{ gap: 0.85rem; }}
section[data-testid="stSidebar"] div.element-container {{ margin-bottom: 0 !important; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem; }}
.side-heading {{ margin: 4px 0 2px !important; }}
.hosp-badge {{
    display: inline-block; background: #0b5560; color: #eafcff; font-weight: 800;
    font-size: 13px; letter-spacing: .3px; padding: 6px 14px; border-radius: 999px;
    margin-bottom: 10px;
}}
.profile-card {{
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(135deg, #087f8c, #0a9ca8); color: white;
    border-radius: 16px; padding: 12px 16px; margin-bottom: 6px;
    box-shadow: 0 8px 20px rgba(0,80,95,0.25);
}}
.profile-avatar {{
    flex: none; width: 42px; height: 42px; border-radius: 50%;
    background: rgba(255,255,255,0.20); display: flex; align-items: center;
    justify-content: center; font-size: 22px;
}}
.profile-text {{ min-width: 0; }}
.profile-name {{ font-size: 16px; font-weight: 800; line-height: 1.2; }}
.profile-meta {{ font-size: 12px; opacity: 0.92; margin-top: 1px; }}
section[data-testid="stSidebar"] div.stButton:has(button[kind]) {{ margin-bottom: 4px; }}
section[data-testid="stSidebar"] div.stButton > button {{
    min-height: 34px !important; padding: 4px 10px !important; font-size: 13px !important;
}}
.patient-chip {{
    background: rgba(255,255,255,0.95); border: 1px solid #cfe6ea; border-radius: 12px;
    padding: 8px 14px; margin: 6px 0 14px; color: #0b5560; font-size: 13px; font-weight: 650;
}}
div[data-testid="stSidebar"] div[data-testid="stExpander"] {{
    border: 1px solid #d8edef; border-radius: 14px; background: #fbfeff; margin-bottom: 8px;
}}
div[data-testid="stSidebar"] div.stButton > button {{
    background: white; color: #087f8c; border: 1px solid #b9dde2; font-weight: 700;
}}
</style>
""")


# ============================================================
# TEXT HELPERS
# ============================================================

_PUNCT = re.compile(r"[!-/:-@\[-`{-~\u0964\u0965\u2026\u201c\u201d\u2018\u2019\u2013\u2014]")


def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_value(value):
    if pd.isna(value):
        return "Not available"
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "none", "null"}:
        return "Not available"
    return value


def has_phrase(text, phrase):
    """Whole-word match for English text, plain substring for Indian scripts."""
    if not phrase:
        return False
    if phrase.isascii():
        return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None
    return phrase in text


def has_non_ascii_letters(text):
    return any(ord(ch) > 127 and ch.isalpha() or 0x0900 <= ord(ch) <= 0x0DFF for ch in text)


def detect_script_lang(text):
    counts = {code: 0 for code in SCRIPT_RANGES}
    for ch in text:
        o = ord(ch)
        for code, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= o <= hi:
                counts[code] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None


def parse_list(value):
    value = safe_value(value)
    if value == "Not available":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [value]


# ============================================================
# TRANSLATION (needs internet + deep-translator)
# ============================================================

@st.cache_resource(show_spinner=False)
def translation_store():
    return {"cache": {}, "down_until": 0.0, "failed": 0}


def _translate_one(text, source, target):
    return GoogleTranslator(source=source, target=target).translate(text)


def to_english(text):
    """Any language -> English (used to understand what the user typed/said)."""
    if not HAS_TRANSLATOR or not text.strip():
        return ""
    store = translation_store()
    key = ("to_en", text)
    if key in store["cache"]:
        return store["cache"][key]
    if time.time() < store["down_until"]:
        return ""
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            out = ex.submit(_translate_one, text, "auto", "en").result(timeout=10)
        out = out or ""
        store["cache"][key] = out
        return out
    except Exception:
        store["down_until"] = time.time() + 45
        return ""


def translate_many(strings, lang):
    """English -> chosen language. Returns True if translation worked."""
    if lang == "en":
        return True
    if not HAS_TRANSLATOR:
        return False
    store = translation_store()
    cache = store["cache"]
    todo = sorted({s for s in strings if s and s.strip() and (s, lang) not in cache})
    if not todo:
        return True
    if time.time() < store["down_until"]:
        return False

    def work(s):
        try:
            cache[(s, lang)] = _translate_one(s, "en", lang) or s
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = list(ex.map(work, todo))
    if not any(ok):
        store["down_until"] = time.time() + 45
        return False
    return True


def tt(text, lang):
    if lang == "en":
        return text
    return translation_store()["cache"].get((text, lang), text)


# ============================================================
# VISIT LOG  (one row per patient visit -> used for the daily report)
# ============================================================

def log_visit(profile, disease, symptoms_text):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_FILE.exists()
    now = datetime.now()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(LOG_COLS)
        w.writerow([
            now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
            profile.get("name") or "", profile.get("age") if profile.get("age") is not None else "",
            profile.get("gender") or "", profile.get("blood_group") or "", symptoms_text, disease,
        ])
    load_log.clear()


@st.cache_data(ttl=5, show_spinner=False)
def load_log():
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=LOG_COLS)
    df = pd.read_csv(LOG_FILE, dtype=str)
    return df


# ============================================================
# SIDEBAR  (hospital related)
# ============================================================

with st.sidebar:
    render(f'<div class="hosp-badge">🏥 {esc(HOSPITAL_NAME)}</div>')

    p = ss.profile
    avatar = "🧑" if p["gender"] == "Male" else "👩" if p["gender"] == "Female" else "🧑‍⚕️"
    if p["name"]:
        age_txt = f'{p["age"]} yrs' if p.get("age") is not None else p["age_group"]
        meta_bits = [x for x in [age_txt, p["gender"]] if x]
        if p.get("blood_group") and p["blood_group"] != "I don't know":
            meta_bits.append(f'🩸 {p["blood_group"]}')
        meta = " • ".join(meta_bits)
        render(f"""
        <div class="profile-card">
            <div class="profile-avatar">{avatar}</div>
            <div class="profile-text">
                <div class="profile-name">{esc(p["name"])}</div>
                <div class="profile-meta">{esc(meta) if meta else "Chatting now"}</div>
            </div>
        </div>
        """)
    else:
        render("""
        <div class="profile-card">
            <div class="profile-avatar">🏥</div>
            <div class="profile-text">
                <div class="profile-name">Nalam Health AI</div>
                <div class="profile-meta">Your healthcare chat assistant</div>
            </div>
        </div>
        """)

    st.button("🆕 New Conversation", use_container_width=True, on_click=new_conversation)

    render('<div class="side-heading">🚨 Emergency numbers (India)</div>')
    render("""
    <div class="emergency">
        <b>108</b> &nbsp;Ambulance<br>
        <b>112</b> &nbsp;National emergency number<br>
        <b>104</b> &nbsp;Tamil Nadu health helpline
    </div>
    """)

    with st.expander("📍 Find hospitals near you"):
        city = st.text_input("City / area", placeholder="Example: Ambur", label_visibility="collapsed", key="city_input")
        query = f"hospitals near {city.strip()}" if city.strip() else "hospitals near me"
        st.link_button(
            "🗺️ Search on Google Maps",
            f"https://www.google.com/maps/search/{quote_plus(query)}",
            use_container_width=True,
        )

    with st.expander("🩺 Which doctor to meet?"):
        render("""
        <div class="dept"><span>Fever, cold, body pain</span><span>General Medicine</span></div>
        <div class="dept"><span>Chest pain, palpitations</span><span>Cardiology</span></div>
        <div class="dept"><span>Stomach pain, acidity</span><span>Gastroenterology</span></div>
        <div class="dept"><span>Skin rash, itching</span><span>Dermatology</span></div>
        <div class="dept"><span>Joint / bone pain</span><span>Orthopedics</span></div>
        <div class="dept"><span>Cough, breathing trouble</span><span>Pulmonology</span></div>
        <div class="dept"><span>Headache, dizziness</span><span>Neurology</span></div>
        <div class="dept"><span>Ear, nose, throat</span><span>ENT</span></div>
        <div class="dept"><span>Children's health</span><span>Pediatrics</span></div>
        """)

    with st.expander("⚠️ Go to the hospital immediately if"):
        render("""
        <div class="tip">• Severe chest pain or pressure</div>
        <div class="tip">• Difficulty in breathing</div>
        <div class="tip">• Sudden weakness, slurred speech, face drooping</div>
        <div class="tip">• Heavy bleeding or fainting</div>
        <div class="tip">• Very high fever that does not come down</div>
        """)

    with st.expander("💡 Daily health tips"):
        render("""
        <div class="tip">• Drink enough clean water</div>
        <div class="tip">• Sleep 7-8 hours</div>
        <div class="tip">• Wash hands before eating</div>
        <div class="tip">• Do a routine check-up once a year</div>
        """)

    with st.expander("📊 Patient Reports (staff only)"):
        if not ss.get("report_unlocked"):
            pin = st.text_input("Staff PIN", type="password", key="report_pin")
            if st.button("Unlock", key="unlock_reports", use_container_width=True):
                if pin == ADMIN_PIN:
                    ss["report_unlocked"] = True
                    st.rerun()
                else:
                    st.error("Wrong PIN.")
        else:
            log_df = load_log()
            if log_df.empty:
                st.caption("No patient visits recorded yet.")
            else:
                dates = sorted(log_df["date"].unique(), reverse=True)
                day = st.selectbox("Date", dates, key="report_date")
                day_df = log_df[log_df["date"] == day]
                counts = day_df["condition"].value_counts()
                st.markdown(f"**{len(day_df)} patient(s) on {day}**")
                for cond, n in counts.items():
                    st.markdown(f"- **{cond.title()}** — {n} patient{'s' if n != 1 else ''}")
                st.bar_chart(counts)
                st.download_button(
                    "⬇️ Download full log (CSV)", log_df.to_csv(index=False),
                    file_name="patient_log.csv", mime="text/csv", use_container_width=True,
                )
            if st.button("Lock", key="lock_reports", use_container_width=True):
                ss["report_unlocked"] = False
                st.rerun()

    with st.expander("⚙️ Settings"):
        other_count = st.slider("Other possible conditions to show", 0, 5, 3)
        out_choice = st.selectbox(
            "🌐 Answer language",
            ["Auto (same as I type)"] + list(LANGS.keys()),
            help="Auto = if you type/speak Tamil or Hindi etc., the answer comes in that language.",
        )
        voice_choice = st.selectbox("🎤 Voice language", list(LANGS.keys()))


# ============================================================
# HEADER
# ============================================================

render(f"""
<div class="hero">
    <div class="hero-title">🏥 NALAM HEALTH AI</div>
    <div class="hero-subtitle">Smart Healthcare Information Assistant &nbsp;•&nbsp; {esc(HOSPITAL_NAME)}</div>
    <div class="hero-note">Type or speak your symptoms in any language (English, Tamil, Tanglish, Hindi ...) and get dataset-based health information before you meet the doctor.</div>
</div>
""")


# ============================================================
# CHECK FILES
# ============================================================

missing_files = [str(p) for p in FILES.values() if not p.exists()]

if missing_files:
    st.error("❌ Required dataset files are missing.")
    for file in missing_files:
        st.write(f"- `{file}`")
    st.info("Make sure all six CSV files are inside the data folder.")
    st.stop()


# ============================================================
# LOAD DATA
#   freq  : rows = diseases, columns = symptoms,
#           value = how often that symptom appears for that disease (0 to 1)
#   prior : how common each disease is in the dataset
# ============================================================

@st.cache_data(show_spinner="Loading healthcare dataset...")
def load_data():
    symptom_df = pd.read_csv(FILES["symptoms"])

    if DISEASE_COL not in symptom_df.columns:
        raise ValueError(
            f"`{DISEASE_COL}` column not found. Available columns: {list(symptom_df.columns)[:15]}..."
        )

    symptom_cols = [c for c in symptom_df.columns if c != DISEASE_COL]
    freq = symptom_df.groupby(DISEASE_COL)[symptom_cols].mean(numeric_only=True)
    freq.index = [str(i) for i in freq.index]

    prior = symptom_df[DISEASE_COL].astype(str).value_counts(normalize=True).reindex(freq.index)

    return (
        freq,
        prior,
        pd.read_csv(FILES["description"]),
        pd.read_csv(FILES["precautions"]),
        pd.read_csv(FILES["medication"]),
        pd.read_csv(FILES["diet"]),
        pd.read_csv(FILES["workout"]),
    )


try:
    (freq_table, prior, description_df, precaution_df,
     medication_df, diet_df, workout_df) = load_data()
except Exception as e:
    st.error("❌ Dataset loading failed.")
    st.code(str(e))
    st.stop()

SYMPTOM_COLS = list(freq_table.columns)
NORM_TO_COL = {normalize_text(c): c for c in SYMPTOM_COLS}
DISEASES = list(freq_table.index)


# ============================================================
# REAL ML MODEL  (trained by train_model.py, optional)
# If data/model.pkl exists, its predictions replace the plain statistics
# formula below. Without it, the app still works using the statistics.
# ============================================================

@st.cache_resource(show_spinner=False)
def load_ml_model():
    if not (HAS_JOBLIB and MODEL_FILE.exists()):
        return None
    try:
        bundle = joblib.load(MODEL_FILE)
        if set(bundle.get("columns", [])) - set(SYMPTOM_COLS):
            return None  # model was trained on a different dataset version
        return bundle
    except Exception:
        return None


ML_MODEL = load_ml_model()


# ============================================================
# OUR OWN NEURAL NETWORK  (trained by train_own_nn.py, optional)
# Pure numpy forward pass — no ML library involved, even at inference time.
# Takes priority over the sklearn model above when both are present.
# ============================================================

def _nn_forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)          # ReLU
    z2 = a1 @ W2 + b2
    z2 = z2 - z2.max(axis=1, keepdims=True)
    e = np.exp(z2)
    return e / e.sum(axis=1, keepdims=True)  # softmax


@st.cache_resource(show_spinner=False)
def load_own_nn():
    if not OWN_NN_FILE.exists():
        return None
    try:
        data = np.load(OWN_NN_FILE, allow_pickle=True)
        columns = list(data["columns"])
        if set(columns) - set(SYMPTOM_COLS):
            return None  # trained on a different dataset version
        return {
            "W1": data["W1"], "b1": data["b1"], "W2": data["W2"], "b2": data["b2"],
            "columns": columns, "classes": list(data["classes"]),
            "accuracy": float(data["accuracy"]),
        }
    except Exception:
        return None


OWN_NN = load_own_nn()


# ============================================================
# LOOKUPS  (disease names differ slightly between CSV files,
# e.g. "...(COPD)" vs "COPD", so keys are matched loosely)
# ============================================================

def build_lookup(df, value_col, as_list=False):
    lookup = {}
    if "Disease" in df.columns and value_col in df.columns:
        for _, row in df.iterrows():
            lookup[normalize_text(row["Disease"])] = row[value_col]
    return lookup


def resolve_key(lookup, disease_key):
    if disease_key in lookup:
        return disease_key
    for k in lookup:
        if len(k) >= 4 and f" {k} " in f" {disease_key} ":
            return k
    close = difflib.get_close_matches(disease_key, list(lookup), n=1, cutoff=0.85)
    return close[0] if close else None


description_lookup = build_lookup(description_df, "Description")
medication_lookup = build_lookup(medication_df, "Medication")
diet_lookup = build_lookup(diet_df, "Diet")
workout_lookup = build_lookup(workout_df, "Workouts")

precaution_lookup = {}
if "Disease" in precaution_df.columns:
    p_cols = [c for c in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
              if c in precaution_df.columns]
    for _, row in precaution_df.iterrows():
        items = [safe_value(row[c]) for c in p_cols]
        precaution_lookup[normalize_text(row["Disease"])] = [i for i in items if i != "Not available"]


def info_for(disease):
    key = normalize_text(disease)

    def get(lookup, default):
        k = resolve_key(lookup, key)
        return lookup[k] if k is not None else default

    return {
        "description": safe_value(get(description_lookup, "Not available")),
        "precautions": get(precaution_lookup, []),
        "medication": parse_list(get(medication_lookup, "")),
        "diet": parse_list(get(diet_lookup, "")),
        "workout": parse_list(get(workout_lookup, "")),
    }


# ============================================================
# EVERYDAY WORDS (English, Tanglish, Hindi, Tamil, ...)  ->  dataset symptoms
# Each row: (label, [dataset symptom columns], "phrase|phrase|phrase")
# Columns that do not exist in the dataset are ignored automatically.
# ============================================================

CONCEPTS = [
    ("fever", ["fever"],
     "high temperature|temperature|feverish|body heat|body is hot|body is very hot|body hot|feel hot|feeling hot|kaichal|kaaichal|kaichchal|udambu soodu|udambu suduthu|"
     "bukhar|bukhaar|காய்ச்சல்|உடல் சூடு|बुखार|జ్వరం|ಜ್ವರ|പനി"),
    ("flu", ["flu-like syndrome", "fever", "ache all over"],
     "flu|viral fever|influenza|viral infection|viral bukhar|வைரஸ் காய்ச்சல்|वायरल बुखार"),
    ("headache", ["headache", "frontal headache"],
     "head pain|head ache|migraine|pain in head|pain in my head|head hurts|head is paining|thalai vali|thalaivali|thalai valikkudhu|thalai valikuthu|sar dard|sir dard|sardard|"
     "தலைவலி|தலை வலி|सिरदर्द|सर दर्द|తలనొప్పి|ತಲೆನೋವು|തലവേദന"),
    ("cold", ["coryza", "nasal congestion", "sneezing", "sinus congestion"],
     "cold|common cold|running nose|runny nose|blocked nose|stuffy nose|nose block|jalubu|jalubhu|salai|sali|sali pudichu|"
     "thumal|thummal|jukam|zukam|nazla|சளி|ஜலதோஷம்|மூக்கு ஒழுகுதல்|जुकाम|ज़ुकाम|सर्दी"),
    ("cough", ["cough"],
     "coughing|dry cough|irumal|irumbal|irumul|khansi|khaansi|இருமல்|खांसी|खासी|దగ్గు|ಕೆಮ್ಮು|ചുമ"),
    ("sore throat", ["sore throat"],
     "throat pain|pain in throat|throat hurts|throat is paining|throat infection|throat irritation|thondai vali|thondai pun|gale mein dard|gala dard|gala kharab|"
     "தொண்டை வலி|தொண்டை புண்|गले में दर्द|गला दर्द"),
    ("stomach pain", ["sharp abdominal pain", "burning abdominal pain", "upper abdominal pain", "lower abdominal pain"],
     "stomach pain|stomach ache|stomachache|tummy pain|tummy ache|belly pain|abdominal pain|stomach problem|stomach issue|"
     "stomach cramps|pain in stomach|pain in my stomach|pain in abdomen|pain in my abdomen|abdomen pain|stomach hurts|stomach is paining|stomach paining|belly hurts|tummy hurts|vayiru vali|vayiru valikkudhu|vayiru valikuthu|vayithu vali|pet dard|pet mein dard|pait dard|"
     "வயிற்று வலி|வயிறு வலி|வயிற்றுவலி|पेट दर्द|पेट में दर्द"),
    ("gas / acidity", ["heartburn", "regurgitation", "stomach bloating", "burning abdominal pain"],
     "gas|gas trouble|gas problem|gastric|acidity|acid reflux|indigestion|bloating|vayiru eriyudhu|vayiru erichal|"
     "nenju erichal|acidity problem|एसिडिटी|पेट में गैस|सीने में जलन|அசிடிட்டி|வயிற்று எரிச்சல்|நெஞ்செரிச்சல்"),
    ("heart problem",
     ["sharp chest pain", "burning chest pain", "chest tightness", "palpitations", "irregular heartbeat",
      "increased heart rate", "decreased heart rate"],
     "heart problem|heart problems|heart issue|heart issues|heart trouble|heart disease|heart pain|heart attack|"
     "heart condition|cardiac|heart related|weak heart|heart patient|idhaya noi|idhayam|idhaya prachanai|"
     "idhayathil vali|heart prachanai|dil ki bimari|dil ka dard|dil mein dard|heart ki problem|"
     "இதய பிரச்சனை|இதய நோய்|இதயம்|दिल की बीमारी|दिल का दर्द|हृदय रोग"),
    ("fast heartbeat", ["palpitations", "increased heart rate", "irregular heartbeat"],
     "heart beating fast|heart racing|racing heart|fast heartbeat|fast heart beat|heart beat fast|heartbeat fast|"
     "heart pounding|nenju padapadappu|padapadappu|dhadkan tez|धड़कन तेज|இதய துடிப்பு அதிகம்|படபடப்பு"),
    ("chest pain", ["sharp chest pain", "burning chest pain", "chest tightness"],
     "chest pain|chest problem|chest discomfort|pain in chest|pain in my chest|chest hurts|nenju vali|nenju valikkudhu|"
     "seene mein dard|seene me dard|chhati mein dard|நெஞ்சு வலி|நெஞ்சுவலி|சீனே|सीने में दर्द|छाती में दर्द"),
    ("breathing problem",
     ["shortness of breath", "difficulty breathing", "wheezing", "hurts to breath", "breathing fast",
      "abnormal breathing sounds"],
     "breathing problem|breathing problems|breathing difficulty|trouble breathing|breathless|breathlessness|"
     "short of breath|cant breathe|can t breathe|asthma|lung problem|moochu thinaral|moochu thiranal|moochu vittu|"
     "moochu prachanai|saans lene mein taklif|saans phulna|dam ghutna|மூச்சு திணறல்|மூச்சுத் திணறல்|"
     "மூச்சு விடுவதில் சிரமம்|सांस फूलना|सांस लेने में तकलीफ|साँस की तकलीफ"),
    ("dizziness", ["dizziness"],
     "dizzy|giddiness|giddy|feeling dizzy|light headed|lightheaded|head spinning|vertigo|mayakkam|maiyakkam|"
     "thalai suthu|thalai sutruthu|thalai suthudhu|chakkar|chakkar aana|மயக்கம்|தலை சுற்றல்|தலைச்சுற்றல்|"
     "चक्कर|चक्कर आना"),
    ("fainting", ["fainting"],
     "faint|fainted|blackout|passed out|unconscious|behosh|மயங்கி விழுந்தேன்|बेहोश"),
    ("fatigue", ["fatigue", "weakness", "sleepiness"],
     "tired|tiredness|feeling weak|weak|no energy|low energy|exhausted|sorvu|soarvu|kalaippu|kalaipu|udambu sorvu|"
     "sorvaga|udambu mudiyala|thakan|thakaan|kamzori|kamjori|சோர்வு|களைப்பு|உடல் சோர்வு|कमजोरी|थकान|थकावट"),
    ("vomiting", ["vomiting", "nausea"],
     "vomit|vomits|puking|throwing up|feel like vomiting|vaanthi|vandhi|vaandhi|vanthi|vandhi varuthu|matali|"
     "ulti|ulti aana|ji michlana|जी मिचलाना|வாந்தி|குமட்டல்|उल्टी"),
    ("diarrhea", ["diarrhea"],
     "loose motion|loose motions|loose stool|loose stools|watery stool|diarrhoea|vayiru pokku|vayiru poguthu|"
     "bedhi|bethi|pedhi|dast|வயிற்றுப்போக்கு|வயிற்று போக்கு|பேதி|दस्त|पतला पखाना"),
    ("constipation", ["constipation"],
     "hard stool|cant pass stool|malachikkal|malacchikkal|kabz|kabj|மலச்சிக்கல்|कब्ज|कब्ज़"),
    ("body pain", ["ache all over"],
     "body pain|body pains|body ache|body aches|muscle ache|muscle pain|muscle pains|udambu vali|udambu valikkudhu|"
     "udal vali|badan dard|body dard|sharir dard|shareer dard|உடல் வலி|உடம்பு வலி|बदन दर्द|शरीर दर्द"),
    ("joint pain", ["joint pain"],
     "joints pain|joint ache|arthritis|muttu vali|mootu vali|jodo mein dard|jodon ka dard|மூட்டு வலி|"
     "जोड़ों का दर्द|जोड़ों में दर्द"),
    ("back pain", ["back pain", "low back pain"],
     "backache|back ache|waist pain|mudhugu vali|muthugu vali|mudugu vali|kamar dard|peeth dard|pith dard|"
     "முதுகு வலி|இடுப்பு வலி|कमर दर्द|पीठ दर्द"),
    ("neck pain", ["neck pain"],
     "kazhuthu vali|kazhutthu vali|gardan dard|கழுத்து வலி|गर्दन दर्द"),
    ("shoulder pain", ["shoulder pain"],
     "thol vali|kandha dard|தோள்பட்டை வலி|कंधे में दर्द"),
    ("knee pain", ["knee pain"],
     "muzhangal vali|mulankal vali|ghutne mein dard|முழங்கால் வலி|घुटने में दर्द"),
    ("leg pain", ["leg pain"],
     "legs pain|kaal vali|kaalvali|pair dard|tang dard|கால் வலி|पैर दर्द|पैरों में दर्द"),
    ("hand pain", ["arm pain", "hand or finger pain", "wrist pain"],
     "hand pain|hands pain|kai vali|haath dard|hath dard|கை வலி|हाथ दर्द|हाथ में दर्द"),
    ("foot pain", ["foot or toe pain", "ankle pain"],
     "foot pain|heel pain|paadham vali|paatham vali|பாத வலி|पैर के तलवे में दर्द"),
    ("toothache", ["toothache", "gum pain"],
     "tooth pain|tooth ache|tooth hurts|pain in tooth|pal vali|dant dard|பல் வலி|दांत दर्द|दाँत दर्द"),
    ("ear pain", ["ear pain"],
     "earache|ear ache|pain in ear|ear hurts|kaadhu vali|kadhu vali|kaan dard|காது வலி|कान दर्द|कान में दर्द"),
    ("hearing problem", ["diminished hearing"],
     "hearing loss|cant hear|kaadhu ketkala|kam sunai|காது கேட்கவில்லை|कम सुनाई देना"),
    ("ringing in ear", ["ringing in ear"], "ringing in ears|tinnitus|kaadhula ringing"),
    ("eye problem",
     ["pain in eye", "eye redness", "lacrimation", "itchiness of eye", "swollen eye", "eye burns or stings",
      "foreign body sensation in eye"],
     "eye problem|eye issue|eye pain|red eye|red eyes|watery eyes|eye infection|kann vali|kan vali|kann sivappu|"
     "kann eriyudhu|aankh dard|aankh mein dard|aankhon mein jalan|கண் வலி|கண் சிவப்பு|கண்ணில் எரிச்சல்|"
     "आंख दर्द|आंख में दर्द|आंखों में जलन"),
    ("vision problem", ["diminished vision", "double vision", "spots or clouds in vision", "blindness"],
     "blurry vision|blurred vision|cant see|vision problem|poor vision|kann theriyala|dhundhla dikhna|"
     "கண் மங்கல்|धुंधला दिखना"),
    ("skin problem",
     ["skin rash", "itching of skin", "acne or pimples", "abnormal appearing skin", "skin irritation", "skin lesion",
      "skin dryness, peeling, scaliness, or roughness"],
     "skin problem|skin issue|skin infection|rash|rashes|itching|itchy|pimples|pimple|acne|thol prachanai|arippu|"
     "arikkudhu|tholil arippu|khujli|khujali|daane|अरिप्पु|खुजली|त्वचा की समस्या|चकत्ते|அரிப்பு|தோல் பிரச்சனை|"
     "தடிப்பு|முகப்பரு"),
    ("allergy", ["allergic reaction"], "allergy|allergic|allergies|ஒவ்வாமை|एलर्जी"),
    ("scalp problem", ["itchy scalp", "irregular appearing scalp"],
     "dandruff|scalp itching|thalai arippu|தலை அரிப்பு|रूसी"),
    ("urine problem", ["painful urination", "frequent urination", "blood in urine"],
     "urine problem|urinary problem|urine infection|uti|burning urination|burning urine|urine burning|"
     "siruneer prachanai|siruneer eriyudhu|sirunir eriyudhu|peshab mein jalan|peshab ki problem|"
     "சிறுநீர் எரிச்சல்|சிறுநீர் பிரச்சனை|पेशाब में जलन|पेशाब की समस्या"),
    ("period pain", ["painful menstruation"],
     "period pain|periods pain|menstrual pain|menstrual cramps|cramps in periods|masikkalam vali|maadhavidai vali|"
     "mathavidai vali|period dard|माहवारी में दर्द|पीरियड्स में दर्द|மாதவிடாய் வலி"),
    ("irregular periods", ["unpredictable menstruation", "frequent menstruation"],
     "irregular periods|irregular period|periods not regular|period late|masikkalam sariyilla|"
     "மாதவிடாய் ஒழுங்கின்மை|अनियमित पीरियड"),
    ("heavy periods", ["heavy menstrual flow", "long menstrual periods"], "heavy bleeding|heavy periods"),
    ("anxiety", ["anxiety and nervousness"],
     "anxiety|stress|stressed|tension|panic|worry|worried|nervous|padhattam|kavalai|ghabrahat|"
     "பதற்றம்|மன அழுத்தம்|கவலை|घबराहट|तनाव"),
    ("fear", ["fears and phobias"], "fear|afraid|phobia|bayam|payam|பயம்|डर"),
    ("depression", ["depression", "low self-esteem"],
     "sad|sadness|feeling low|feeling down|depressed|udasi|manachorvu|மனச்சோர்வு|உடம்பு சரியில்லை மனசு|उदासी"),
    ("insomnia", ["insomnia"],
     "sleeplessness|cant sleep|can t sleep|not able to sleep|no sleep|thookam varala|thookkam illa|thookam illai|"
     "neend nahi aati|नींद नहीं आती|தூக்கமின்மை|தூக்கம் வரவில்லை"),
    ("anger", ["excessive anger", "temper problems", "hostile behavior"],
     "anger|angry|kobam|gussa|கோபம்|गुस्सा"),
    ("memory problem", ["disturbance of memory"],
     "memory loss|forgetful|forgetting|maranthu|yaad nahi|மறதி|भूलने की बीमारी|याददाश्त"),
    ("numbness", ["paresthesia", "loss of sensation"],
     "numbness|tingling|pins and needles|maradhipu|sunn|सुन्न|மரத்துப்போதல்|மதமதப்பு"),
    ("seizure", ["seizures"],
     "seizure|fits|epilepsy|convulsions|valippu|daura|मिर्गी|வலிப்பு"),
    ("chills", ["chills"],
     "shivering|shiver|kulir|kulirudhu|thand lagna|ठंड लगना|குளிர்|நடுக்கம்"),
    ("sweating", ["sweating"],
     "sweat|perspiration|sweating a lot|viyarvai|vearvai|pasina|पसीना|வியர்வை"),
    ("weight gain", ["weight gain"],
     "gaining weight|overweight|obesity|எடை கூடுதல்|मोटापा|वजन बढ़ना"),
    ("loss of appetite", ["decreased appetite"],
     "loss of appetite|no appetite|not hungry|pasi illa|pasi edukkala|bhook nahi|भूख नहीं लगना|பசியின்மை"),
    ("swelling", ["leg swelling", "ankle swelling", "peripheral edema", "fluid retention", "foot or toe swelling"],
     "swelling in legs|swollen legs|swollen feet|kaal veekkam|kaal vikkam|paer sujan|पैरों में सूजन|கால் வீக்கம்|வீக்கம்"),
    ("jaundice", ["jaundice"],
     "yellow eyes|yellow skin|manjal kamalai|kamalai|peelia|पीलिया|மஞ்சள் காமாலை|காமாலை"),
    ("nosebleed", ["nosebleed"],
     "nose bleeding|bleeding from nose|mookkil ratham|nakseer|नाक से खून|மூக்கில் இரத்தம்"),
    ("cough with blood", ["hemoptysis"],
     "coughing blood|blood in cough|cough with blood|irumalil ratham|khoon ki khansi|இருமலில் இரத்தம்"),
    ("phlegm", ["coughing up sputum"],
     "phlegm|sputum|mucus cough|kabam|kaphum|balgam|கபம்|बलगम"),
    ("blood in stool", ["blood in stool", "rectal bleeding", "melena"],
     "bleeding in stool|ratham malam|potty mein khoon|மலத்தில் இரத்தம்|मल में खून"),
    ("mouth ulcer", ["mouth ulcer"],
     "mouth sores|mouth ulcers|vaai punn|vayil pun|muh ke chhale|வாய்ப்புண்|मुंह के छाले"),
    ("dry mouth", ["mouth dryness"], "dry mouth|வாய் வறட்சி|मुंह सूखना"),
    ("walking problem", ["problems with movement", "leg weakness", "leg pain"],
     "cant walk|difficulty walking|walking problem|trouble walking|nadakka mudiyala|नहीं चल पा रहा|நடக்க முடியவில்லை"),
    ("tonsils", ["swollen or red tonsils"], "tonsils|tonsillitis|டான்சில்|टॉन्सिल"),
]

STOPWORDS = {
    "i", "have", "has", "had", "am", "is", "are", "was", "my", "me", "a", "an", "the", "and", "or", "of", "in",
    "on", "at", "to", "with", "from", "for", "since", "feeling", "feel", "feels", "got", "getting", "having",
    "little", "very", "too", "much", "some", "now", "also", "there", "it", "its", "this", "that", "been", "do",
    "does", "not", "no", "during", "please", "help", "suffering", "problem", "problems", "when", "while",
    "because", "since", "days", "day", "week", "from", "last", "few", "many", "so", "like",
}

CANON = {
    "aches": "pain", "ache": "pain", "aching": "pain", "hurts": "pain", "hurt": "pain", "hurting": "pain",
    "paining": "pain", "painful": "pain", "pains": "pain", "swollen": "swelling", "swell": "swelling",
    "tummy": "abdominal", "belly": "abdominal", "stomach": "abdominal", "tired": "fatigue",
    "legs": "leg", "hands": "hand", "eyes": "eye", "ears": "ear", "knees": "knee", "arms": "arm",
    "fingers": "finger", "abdomen": "abdominal", "toes": "toe", "feet": "foot", "lips": "lip",
}

EMERGENCY_KEYS = ["chest pain", "chest tightness", "shortness of breath", "difficulty breathing", "fainting"]


def _tokens(text):
    out = []
    for w in normalize_text(text).split():
        w = CANON.get(w, w)
        if w not in STOPWORDS:
            out.append(w)
    return out


# ---- built once ----
CONCEPT_ROWS = []
for _label, _cols, _phrases in CONCEPTS:
    _real = [c for c in _cols if c in freq_table.columns]
    if _real:
        _plist = sorted({normalize_text(p) for p in _phrases.split("|") if p.strip()}, key=len, reverse=True)
        CONCEPT_ROWS.append({"label": _label, "cols": _real, "phrases": _plist})

ASCII_PHRASE_TO_CONCEPT = {}
for _i, _row in enumerate(CONCEPT_ROWS):
    for _p in _row["phrases"]:
        if _p.isascii() and len(_p) >= 5:
            ASCII_PHRASE_TO_CONCEPT[_p] = _i

COL_TOKENS = {col: _tokens(col) for col in SYMPTOM_COLS}


def _ngrams(text, max_n=3):
    words = text.split()
    grams = set()
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            grams.add(" ".join(words[i:i + n]))
    return grams


def _token_present(tok, text_tokens):
    if tok in text_tokens:
        return True
    if len(tok) >= 5:
        return bool(difflib.get_close_matches(tok, text_tokens, n=1, cutoff=0.85))
    return False


# ============================================================
# UNDERSTANDING WHAT THE USER WROTE
# ============================================================

def feature_from_cols(label, cols):
    if len(cols) == 1:
        series = freq_table[cols[0]]
    else:
        series = freq_table[cols].max(axis=1)
    return {"label": label, "cols": cols, "series": series}


def match_one_text(text):
    """Exact names + everyday-word concepts on ONE normalized text. Returns (features, leftover_text)."""
    features, exact_cols, used = [], [], set()
    masked = text

    # 1) exact dataset symptom names (longest first so "low back pain" wins)
    for norm in sorted(NORM_TO_COL, key=len, reverse=True):
        if norm and has_phrase(masked, norm):
            col = NORM_TO_COL[norm]
            features.append(feature_from_cols(col, [col]))
            exact_cols.append(col)
            used.add(col)
            if norm.isascii():
                masked = re.sub(r"(?<!\w)" + re.escape(norm) + r"(?!\w)", " ", masked)
            else:
                masked = masked.replace(norm, " ")

    # 2) everyday words / multilingual concepts (run on masked text)
    for row in CONCEPT_ROWS:
        if row["label"] in used:
            continue
        hit = [p for p in row["phrases"] if has_phrase(masked, p)]
        if hit:
            for p in hit:                      # remove used words so later steps only see the leftovers
                if p.isascii():
                    masked = re.sub(r"(?<!\w)" + re.escape(p) + r"(?!\w)", " ", masked)
                else:
                    masked = masked.replace(p, " ")
            if set(row["cols"]) <= set(exact_cols):
                continue
            features.append(feature_from_cols(row["label"], row["cols"]))
            used.add(row["label"])

    return features, re.sub(r"\s+", " ", masked).strip()


def match_tokens(text):
    """Order-free matching: 'pain in my arm' -> arm pain."""
    text_tokens = _tokens(text)
    if not text_tokens:
        return []
    found = []
    for col, toks in COL_TOKENS.items():
        if toks and all(_token_present(t, text_tokens) for t in toks):
            found.append(col)
    # drop columns fully contained in a longer matched column ("back pain" in "low back pain")
    keep = [c for c in found if not any(c != o and set(COL_TOKENS[c]) < set(COL_TOKENS[o]) for o in found)]
    return [feature_from_cols(c, [c]) for c in keep]


def match_fuzzy(text):
    """Spelling mistakes in single words: 'hedache', 'fevar', 'vomitting'."""
    features, used = [], set()
    single_cols = [n for n in NORM_TO_COL if " " not in n]
    single_phr = [p for p in ASCII_PHRASE_TO_CONCEPT if " " not in p]
    for word in set(text.split()):
        if not word.isascii() or len(word) < 4 or word in STOPWORDS:
            continue
        cutoff = 0.85 if len(word) == 4 else 0.8
        close = difflib.get_close_matches(word, single_cols, n=1, cutoff=cutoff)
        if close:
            col = NORM_TO_COL[close[0]]
            if col not in used:
                used.add(col)
                features.append(feature_from_cols(col, [col]))
            continue
        close = difflib.get_close_matches(word, single_phr, n=1, cutoff=cutoff)
        if close:
            row = CONCEPT_ROWS[ASCII_PHRASE_TO_CONCEPT[close[0]]]
            if row["label"] not in used:
                used.add(row["label"])
                features.append(feature_from_cols(row["label"], row["cols"]))
    return features


def match_closest(text, limit=2):
    """Last resort: symptom words that look similar to what was typed."""
    words = [w for w in text.split() if w.isascii() and len(w) >= 5 and w not in STOPWORDS]
    scored = {}
    for w in words:
        for col, toks in COL_TOKENS.items():
            for t in toks:
                if len(t) >= 5:
                    r = difflib.SequenceMatcher(None, w, t).ratio()
                    if r >= 0.78 and r > scored.get(col, 0):
                        scored[col] = r
    best = sorted(scored.items(), key=lambda kv: -kv[1])[:limit]
    return [feature_from_cols(col, [col]) for col, _ in best]


def merge_features(*lists):
    merged, seen = [], set()
    for lst in lists:
        for f in lst:
            if f["label"] not in seen:
                seen.add(f["label"])
                merged.append(f)
    return merged


DISEASE_ALIASES = [
    ("cold|common cold|jalubu|jalubhu|salai|sali|jukam|zukam|nazla|சளி|ஜலதோஷம்|जुकाम|ज़ुकाम", ["common cold"]),
    ("low sugar|sugar low|hypoglycaemia", ["hypoglycemia"]),
    ("uti|urine infection|urinary infection", ["urinary tract infection"]),
    ("piles|haemorrhoids", ["hemorrhoids"]),
    ("pink eye|madras eye|eye flu", ["conjunctivitis"]),
    ("hay fever|seasonal allergy|dust allergy", ["seasonal allergies (hay fever)"]),
    ("stomach flu|food poisoning|gastroenteritis", ["infectious gastroenteritis"]),
    ("heart problem|heart problems|heart issue|heart issues|heart trouble|heart disease|cardiac|weak heart|"
     "idhaya noi|idhaya prachanai|இதய நோய்|இதய பிரச்சனை|दिल की बीमारी|हृदय रोग",
     ["heart failure", "angina", "heart attack"]),
    ("high bp|hypertension|blood pressure|bp problem", ["hypertensive heart disease"]),
    ("tooth decay|cavity|cavities", ["dental caries"]),
    ("hepatitis|liver problem|jaundice", ["liver disease"]),
    ("ear infection", ["otitis media"]),
    ("sinus|sinus problem", ["acute sinusitis"]),
    ("throat infection|tonsillitis", ["strep throat"]),
    ("kidney infection|kidney problem", ["acute kidney injury"]),
    ("stone in gallbladder|gall stone", ["gallstone"]),
    ("urine burning", ["cystitis"]),
]

ALIAS_ROWS = []
for _phr, _ds in DISEASE_ALIASES:
    _real = [d for d in _ds if d in freq_table.index]
    if _real:
        ALIAS_ROWS.append(([normalize_text(p) for p in _phr.split("|")], _real))

NOT_IN_DATASET = {
    "diabetes", "sugar", "dengue", "malaria", "typhoid", "covid", "corona", "tuberculosis", "tb",
    "cancer", "thyroid", "cholera", "hiv", "chikungunya", "swine flu",
}


def find_named_diseases(texts):
    """If the user typed a condition name (e.g. 'asthma', 'cold', 'heart problem'), find it."""
    found = []
    for text in texts:
        for phrases, ds in ALIAS_ROWS:
            if any(has_phrase(text, p) for p in phrases):
                for d in ds:
                    if d not in found:
                        found.append(d)
    norm_names = {normalize_text(d): d for d in DISEASES}
    # also allow the abbreviation inside brackets, e.g. "(COPD)"
    for d in DISEASES:
        m = re.search(r"\(([^)]+)\)", d)
        if m:
            norm_names[normalize_text(m.group(1))] = d
            norm_names[normalize_text(d.replace(m.group(0), ""))] = d
    for text in texts:
        for name in sorted(norm_names, key=len, reverse=True):
            if len(name) >= 4 and has_phrase(text, name) and norm_names[name] not in found:
                found.append(norm_names[name])
        if not found:
            for gram in _ngrams(text, 4):
                if len(gram) >= 6 and gram.isascii():
                    close = difflib.get_close_matches(gram, list(norm_names), n=1, cutoff=0.88)
                    if close and norm_names[close[0]] not in found:
                        found.append(norm_names[close[0]])
    return found


def understand(text_raw, picked):
    """
    Returns (features, approx, english_text, named_diseases)
    Works for English, Tanglish, Hindi/Tamil script, spelling mistakes, and any
    other language (via translation when available).
    """
    text = normalize_text(text_raw)
    english = ""
    if HAS_TRANSLATOR and text_raw and has_non_ascii_letters(text_raw):
        english = normalize_text(to_english(text_raw))
    texts = [t for t in [text, english] if t]

    features = [feature_from_cols(c, [c]) for c in picked if c in freq_table.columns]

    def analyse(t):
        found, leftover = match_one_text(t)
        words = [w for w in leftover.split() if w not in STOPWORDS]
        if words:
            found = merge_features(found, match_tokens(leftover), match_fuzzy(" ".join(words)))
        return found

    for t in texts:
        features = merge_features(features, analyse(t))

    # any other language (Spanish, French, ...): try translation as a fallback
    if not features and not english and text:
        english = normalize_text(to_english(text_raw))
        if english and english != text:
            features = merge_features(features, analyse(english))
            texts.append(english)

    named = find_named_diseases(texts) if texts else []

    approx = False
    if not features and not named and texts:
        for t in texts:
            features = merge_features(features, match_closest(t))
        approx = bool(features)

    return features, approx, english, named


# ============================================================
# RANKING  (naive-Bayes style: how common the symptoms are for each disease,
#           weighted by how common the disease is in the dataset)
# ============================================================

EPS = 0.01


FEMALE_ONLY_WORDS = ["vagina", "vaginal", "menstru", "pregnan", "vulvo"]
MALE_ONLY_WORDS = ["prostat"]


def gender_allowed(disease, gender):
    d = disease.lower()
    if gender == "Female" and any(w in d for w in MALE_ONLY_WORDS):
        return False
    if gender == "Male" and any(w in d for w in FEMALE_ONLY_WORDS):
        return False
    return True


def get_results_ml(features, named, top_n, gender):
    """Real ML model: build the same 0/1 symptom vector it was trained on, and predict."""
    model, columns = ML_MODEL["model"], ML_MODEL["columns"]

    matched_cols = set()
    for f in features:
        matched_cols.update(f["cols"])
    vector = pd.DataFrame([[1 if c in matched_cols else 0 for c in columns]], columns=columns)

    proba = pd.Series(model.predict_proba(vector)[0], index=model.classes_)

    if gender and gender not in ("Other", "Prefer not to say"):
        named = [d for d in named if gender_allowed(d, gender)]
        proba = proba[[gender_allowed(d, gender) for d in proba.index]]

    order = list(proba.sort_values(ascending=False).index)
    ranked = list(named)
    for d in order:
        if d not in ranked:
            ranked.append(d)
        if len(ranked) >= top_n:
            break
    ranked = ranked[:top_n]

    results = []
    for d in ranked:
        matched = [f["label"] for f in features]  # every recognized symptom was fed to the model
        results.append({
            "disease": d,
            "coverage": len(features),
            "total": len(features),
            "share": round(float(proba.get(d, 0.0)) * 100, 1),
            "matched": matched,
            "is_named": d in named,
        })
    return results


def get_results(features, named, top_n=4, gender=None):
    results = []
    scores = None
    if features:
        log_scores = np.log(prior.values.astype(float))
        for f in features:
            log_scores = log_scores + np.log(f["series"].values.astype(float) + EPS)
        post = np.exp(log_scores - log_scores.max())
        post = post / post.sum()
        order = list(np.argsort(-post))
        scores = pd.Series(post, index=freq_table.index)
    else:
        order = []

    if gender and gender not in ("Other", "Prefer not to say"):
        named = [d for d in named if gender_allowed(d, gender)]
        order = [i for i in order if gender_allowed(freq_table.index[i], gender)]

    ranked = list(named)
    for i in order:
        d = freq_table.index[i]
        if d not in ranked:
            ranked.append(d)
        if len(ranked) >= top_n:
            break
    ranked = ranked[:top_n]

    total_share = float(scores[ranked].sum()) if scores is not None else 0.0

    for d in ranked:
        matched = [f["label"] for f in features if float(f["series"][d]) > 0]
        share = (float(scores[d]) / total_share * 100) if (scores is not None and total_share > 0) else None
        results.append({
            "disease": d,
            "coverage": len(matched),
            "total": len(features),
            "share": None if share is None else round(share, 1),
            "matched": matched,
            "is_named": d in named,
        })
    return results


# ============================================================
# TEXT SHOWN IN RESULTS (translatable)
# ============================================================

STR = {
    "done": "Symptom analysis completed.",
    "recognized": "Recognized Symptoms",
    "most_likely": "Most Likely Reason",
    "best_badge": "Best match for your symptoms",
    "ml_badge": "🤖 ML model prediction",
    "searched_badge": "Condition matching your words",
    "match": "Symptom match",
    "share": "Relative match",
    "ml_share": "Model confidence",
    "matched": "Matched symptoms",
    "about": "About this condition",
    "precautions": "Precautions",
    "medication": "Medication",
    "diet": "Diet",
    "workout": "Workout",
    "na": "Not available in the dataset.",
    "med_note": "Dataset information only. This is not a personal prescription. Please ask a doctor before taking any medicine.",
    "others": "Other Possible Conditions",
    "others_note": "These are less likely matches. Add more symptoms to narrow it down.",
    "alert": "Chest pain, breathing trouble or fainting can be an emergency. If it is severe or sudden, call 108 or 112, or go to the nearest hospital right away.",
    "approx": "I could not find an exact match, so I used the closest symptoms. Please check them below.",
    "warn_title": "Important health information",
    "warn_1": "The condition above is a dataset-based symptom match. The percentage is not a medical probability and does not confirm a diagnosis.",
    "warn_2": "Medication and lifestyle information comes from the dataset and should not be treated as a personal prescription.",
    "warn_3": "For persistent, severe, or concerning symptoms, consult a qualified healthcare professional.",
}


def list_html(items):
    if not items:
        return None
    lis = ""
    for item in items:
        if ": " in item and len(item.split(": ", 1)[0]) < 60:
            head, rest = item.split(": ", 1)
            lis += f"<li><b>{esc(head)}</b>: {esc(rest)}</li>"
        else:
            lis += f"<li>{esc(item)}</li>"
    return f"<ul>{lis}</ul>"


# ============================================================
# RESULT VIEW
# ============================================================

AGE_NOTES = {
    "Child (0-12)": "This is for a child. Please do not give any medicine without a pediatrician's advice.",
    "Teen (13-19)": "For teenagers, please confirm dosage with a doctor or pharmacist before taking any medicine.",
    "Adult (20-40)": "",
    "Middle-aged (41-60)": "If you have any existing condition (BP, sugar, etc.), please check with a doctor before starting any new medicine.",
    "Senior (60+)": "For senior citizens, medicine doses often need adjustment and interactions are more common. Please consult a doctor before taking anything.",
}


def show_results(query, other_count, out_choice, profile=None):
    text, picked = query["text"], query["picked"]
    profile = profile or {}
    gender = profile.get("gender")
    age_group = profile.get("age_group")

    with st.spinner("Analyzing symptoms..."):
        features, approx, english, named = understand(text, picked)
        use_ml = ML_MODEL is not None and bool(features)
        if use_ml:
            results = get_results_ml(features, named, top_n=1 + other_count, gender=gender)
        else:
            results = (
                get_results(features, named, top_n=1 + other_count, gender=gender)
                if (features or named) else []
            )

        # decide answer language
        if out_choice.startswith("Auto"):
            out_lang = detect_script_lang(text) or "en"
        else:
            out_lang = LANGS[out_choice][0]

        translated_ok = True
        if out_lang != "en" and results:
            infos = {r["disease"]: info_for(r["disease"]) for r in results}
            strings = list(STR.values())
            for r in results:
                inf = infos[r["disease"]]
                strings += [r["disease"], inf["description"]] + inf["precautions"]
                strings += inf["medication"] + inf["diet"] + inf["workout"]
            strings += [f["label"] for f in features]
            translated_ok = translate_many(strings, out_lang)
            if not translated_ok:
                out_lang = "en"

    def T(s):
        return tt(s, out_lang)

    if out_lang == "en" and not translated_ok:
        st.warning(
            "⚠️ Could not translate the answer (internet or `deep-translator` missing). Showing English."
        )

    if english and english != normalize_text(text):
        render(f'<div class="note-box">🌐 <b>Understood as:</b> {esc(english)}</div>')

    if not results:
        typed_norm = normalize_text(text)
        asked = [w for w in NOT_IN_DATASET if has_phrase(typed_norm, w)]
        if asked:
            st.warning(
                f"⚠️ '{asked[0]}' is not one of the 100 conditions in this dataset. "
                "Please describe the symptoms instead (for example: fever, tiredness, headache)."
            )
        else:
            st.warning("⚠️ I could not understand any symptom from your message.")
        st.info(
            "Try simple words like: fever, headache, cold, cough, chest pain, heart problem, stomach pain, "
            "dizziness, vomiting, skin rash. You can also choose symptoms from the list below the search box."
        )
        with st.expander("📖 See all symptoms I understand"):
            st.write(", ".join(sorted(SYMPTOM_COLS)))
        with st.expander("🏥 See all conditions in the dataset"):
            st.write(", ".join(sorted(d.title() for d in DISEASES)))
        return None

    name = profile.get("name")
    bits = [name, f'{profile["age"]} yrs' if profile.get("age") is not None else None, profile.get("gender")]
    if profile.get("blood_group") and profile["blood_group"] != "I don't know":
        bits.append(f'Blood group {profile["blood_group"]}')
    bits = [b for b in bits if b]
    if bits:
        render(f'<div class="patient-chip">🧾 {esc(" • ".join(bits))}</div>')

    hello = f'{esc(name)}, ' if name else ""
    render(f'<div class="success-box">✅ <b>{hello}{esc(T(STR["done"]))}</b></div>')

    if approx:
        render(f'<div class="note-box">🔍 {esc(T(STR["approx"]))}</div>')

    needs_alert = any(
        any(k in normalize_text(c) for k in EMERGENCY_KEYS) for f in features for c in f["cols"]
    )
    if needs_alert:
        render(f'<div class="alert-box">🚨 <b>{esc(T(STR["alert"]))}</b></div>')

    if features:
        render(f'<div class="section-heading">🩺 {esc(T(STR["recognized"]))}</div>')
        render("".join(f'<span class="symptom-pill">{esc(T(f["label"]))}</span>' for f in features))
        group_notes = [
            f'<b>{esc(f["label"])}</b> ➜ {esc(", ".join(f["cols"][:6]))}'
            for f in features if len(f["cols"]) > 1
        ]
        if group_notes:
            render('<div class="caption">' + "<br>".join(group_notes) + "</div>")

    best, others = results[0], results[1:]
    info = info_for(best["disease"])

    if use_ml and not best["is_named"]:
        badge = STR["ml_badge"]
    elif best["is_named"]:
        badge = STR["searched_badge"]
    else:
        badge = STR["best_badge"]
    share_label = STR["ml_share"] if use_ml else STR["share"]
    score_line = ""
    if best["total"]:
        share_txt = f' &nbsp;•&nbsp; {esc(T(share_label))} {best["share"]}%' if best["share"] is not None else ""
        score_line = f"""
        <div class="score-row">
            <span class="score-text">{esc(T(STR["match"]))}: {best["coverage"]}/{best["total"]}{share_txt}</span>
            <div class="bar"><div class="bar-fill" style="width:{min(best["share"] or 0, 100)}%"></div></div>
        </div>
        """
    matched_line = ""
    if best["matched"]:
        matched_line = (
            f'<div class="matched"><b>{esc(T(STR["matched"]))}:</b> '
            f'{esc(", ".join(T(m) for m in best["matched"]))}</div>'
        )

    render(f'<div class="section-heading">🎯 {esc(T(STR["most_likely"]))}</div>')
    render(f"""
    <div class="main-card">
        <span class="badge">{esc(T(badge))}</span>
        <div class="main-title">{esc(T(best["disease"]))}</div>
        {score_line}
        {matched_line}
        <div class="section-label">📋 {esc(T(STR["about"]))}</div>
        <div class="info-box">{esc(T(info["description"]))}</div>
    </div>
    """)

    tab_pre, tab_med, tab_diet, tab_work = st.tabs([
        f"🛡️ {T(STR['precautions'])}", f"💊 {T(STR['medication'])}",
        f"🥗 {T(STR['diet'])}", f"🏃 {T(STR['workout'])}",
    ])

    def box(items):
        body = list_html([T(i) for i in items])
        render(f'<div class="info-box">{body if body else esc(T(STR["na"]))}</div>')

    with tab_pre:
        box(info["precautions"])
    with tab_med:
        box(info["medication"])
        render(f'<div class="caption">{esc(T(STR["med_note"]))}</div>')
    with tab_diet:
        box(info["diet"])
    with tab_work:
        box(info["workout"])

    if others:
        render(f'<div class="section-heading">🔎 {esc(T(STR["others"]))}</div>')
        pills = ""
        for o in others:
            extra = ""
            if o["total"]:
                extra = f'<small>{o["coverage"]}/{o["total"]}' + (f' • {o["share"]}%' if o["share"] is not None else "") + "</small>"
            pills += f'<span class="other-pill">{esc(T(o["disease"]))}{extra}</span>'
        render(pills)
        render(f'<div class="caption">{esc(T(STR["others_note"]))}</div>')

    age_note = AGE_NOTES.get(age_group, "")
    age_line = f"<br><br>{esc(T(age_note))}" if age_note else ""

    render(f"""
    <div class="warning-box">
        <b>⚠️ {esc(T(STR["warn_title"]))}</b><br><br>
        {esc(T(STR["warn_1"]))}<br><br>
        {esc(T(STR["warn_2"]))}<br><br>
        {esc(T(STR["warn_3"]))}{age_line}
    </div>
    """)

    return best["disease"]


# ============================================================
# VOICE  ->  TEXT
# ============================================================

def transcribe(audio_bytes, sr_lang):
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio, language=sr_lang)


def hear(audio_bytes):
    """Voice bytes -> text, with a friendly message on failure."""
    try:
        with st.spinner("🎤 Listening..."):
            return transcribe(audio_bytes, LANGS[voice_choice][1]), None
    except sr.UnknownValueError:
        return None, "🎤 Sorry, I could not understand the voice. Please speak clearly and try again."
    except Exception as e:
        return None, f"🎤 Voice recognition needs internet. ({type(e).__name__})"


# ============================================================
# OPTIONAL FASTAPI APPOINTMENT INTEGRATION
# Keep the symptom-matching chatbot working when the API is offline.
# ============================================================
NALAM_API_BASE = "http://127.0.0.1:8000"


def nalam_api_request(path, method="GET", payload=None, timeout=10, token=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        NALAM_API_BASE + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw) if raw else {}, None
            except json.JSONDecodeError:
                return {"raw": raw}, None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail)
        except Exception:
            pass
        return None, f"API returned HTTP {e.code}: {detail}"
    except Exception as e:
        return None, f"Could not connect to Nalam API at {NALAM_API_BASE}. Check that FastAPI is running. ({e})"


def find_patient_id(obj):
    if isinstance(obj, dict):
        for key in ("patient_id", "PatientID", "patientId", "id", "ID"):
            value = obj.get(key)
            if value not in (None, ""):
                return str(value)
        for value in obj.values():
            found = find_patient_id(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_patient_id(value)
            if found:
                return found
    return None


# ============================================================
# CHAT FLOW
#   start -> choose_role -> (doctor)  -> doctor login portal
#                        -> (patient) -> await_symptom -> need_name -> need_age
#                                        -> need_gender -> need_blood -> ready
# Name / age / gender are collected once per conversation; after that the
# chat stays in "ready" and every new message is analyzed straight away,
# personalized with the saved profile.
# ============================================================

def bot(text):
    ss.messages.append({"role": "assistant", "kind": "text", "text": text})


def user_says(text):
    ss.messages.append({"role": "user", "kind": "text", "text": text})


def proceed_to_analysis():
    name = ss.profile["name"]
    bot(f"Thanks {name}! Here's what I found based on your symptoms:" if name
        else "Here's what I found based on your symptoms:")
    ss.messages.append({
        "role": "assistant", "kind": "analysis",
        "query": {"text": ss.pending_text, "picked": []},
        "profile": dict(ss.profile),
        "logged": False,
    })
    ss.stage = "ready"


GREETING_ONLY = {
    "hi", "hii", "hiii", "hey", "hello", "helo", "hlo", "hru", "vanakkam", "vannakam", "vanakam",
    "namaste", "namaskar", "namaskaram", "good morning", "good afternoon", "good evening", "gm",
    "hi there", "hello there", "how are you", "yo",
}


def is_greeting_only(text):
    return normalize_text(text) in GREETING_ONLY


def handle_symptom_text(txt):
    txt = txt.strip()
    if not txt:
        return
    user_says(txt)

    if is_greeting_only(txt):
        name = ss.profile["name"]
        bot(f"Hello {name}! 😊 Tell me what symptoms you're having." if name
            else "Hello! 😊 How can I help you today? Please tell me your symptoms.")
        st.rerun()
        return

    ss.pending_text = txt
    if not ss.profile["name"]:
        bot("Sure — before that, may I know your name?")
        ss.stage = "need_name"
    else:
        proceed_to_analysis()
    st.rerun()


def voice_widget():
    """Fallback mic (older Streamlit without a built-in mic inside chat_input)."""
    if not (HAS_AUDIO_INPUT and HAS_SR) or HAS_CHAT_MIC:
        return
    audio = st.audio_input("🎤 Or record your symptoms", key=f"voice_{ss.stage}_{len(ss.messages)}")
    if audio is None:
        return
    audio_bytes = audio.getvalue()
    audio_id = hashlib.md5(audio_bytes).hexdigest()
    if audio_id == ss.last_audio:
        return
    ss.last_audio = audio_id
    said, err = hear(audio_bytes)
    if said:
        handle_symptom_text(said)
    elif err:
        st.warning(err)


def chat_box(placeholder, key):
    """WhatsApp-style input: type OR tap the mic inside the SAME box and speak."""
    if HAS_CHAT_MIC and HAS_SR:
        val = st.chat_input(placeholder, accept_audio=True, audio_sample_rate=16000, key=key)
        if val is None:
            return None
        typed = (getattr(val, "text", "") or "").strip()
        audio = getattr(val, "audio", None)
        if audio is not None:
            audio_id = hashlib.md5(audio.getvalue()).hexdigest()
            if audio_id != ss.last_audio:
                ss.last_audio = audio_id
                said, err = hear(audio.getvalue())
                if said:
                    return (typed + " " + said).strip()
                if err:
                    st.warning(err)
        return typed or None
    return st.chat_input(placeholder, key=key)


if ss.stage == "start":
    bot("👋 Hi! I'm **Nalam Health AI**. Are you a **Doctor** or a **Patient**? Please choose below.")
    ss.stage = "choose_role"

# ---------- chat history ----------
for m in ss.messages:
    avatar = "🏥" if m["role"] == "assistant" else "🙂"
    with st.chat_message(m["role"], avatar=avatar):
        if m["kind"] == "text":
            st.markdown(m["text"])
        else:
            disease = show_results(m["query"], other_count, out_choice, m["profile"])
            if disease and not m.get("logged"):
                log_visit(m["profile"], disease, m["query"]["text"])
                m["logged"] = True

# ---------- next step ----------
if ss.stage == "choose_role":
    # user can say "hi" and the bot says hi back
    txt = st.chat_input("Say hi or choose Doctor / Patient below...", key="in_role")
    if txt and txt.strip():
        user_says(txt.strip())
        if is_greeting_only(txt):
            bot("Hello! 😊 Please choose below: are you a Doctor or a Patient?")
        else:
            bot("Please first choose whether you are a Doctor or a Patient 🙂")
        st.rerun()

    render("<div class='role-welcome'><h2>👋 Welcome to Nalam Health AI</h2><p>Please choose your role to continue. Doctors can sign in to manage appointment requests; patients can use the health chatbot and book a consultation.</p></div>")
    st.markdown("### Choose your role")
    doc_col, pat_col = st.columns(2)
    if doc_col.button("👨‍⚕️ Doctor", use_container_width=True, key="role_doctor"):
        user_says("Doctor")
        ss.role = "doctor"
        bot("Welcome, Doctor! 👨‍⚕️ Please log in below.")
        ss.stage = "doctor"
        st.rerun()
    if pat_col.button("🧑 Patient", use_container_width=True, key="role_patient"):
        user_says("Patient")
        ss.role = "patient"
        bot("Welcome! 😊 Please tell me your symptoms in any language.")
        ss.stage = "await_symptom"
        st.rerun()

elif ss.stage == "await_symptom":
    chips = [("🌡️", "Fever"), ("🤕", "Headache"), ("🤧", "Cold and cough"), ("🤢", "Stomach pain")]
    cols = st.columns(len(chips))
    for c, (icon, label) in zip(cols, chips):
        if c.button(f"{icon} {label}", use_container_width=True, key=f"chip_{label}"):
            handle_symptom_text(label)
    txt = chat_box("Type your symptoms or tap 🎤 and speak...", key="in_symptom")
    if txt:
        handle_symptom_text(txt)
    voice_widget()

elif ss.stage == "need_name":
    txt = st.chat_input("Your name...")
    if txt and txt.strip():
        user_says(txt)
        ss.profile["name"] = txt.strip().title()[:40]
        bot(f"Nice to meet you, {ss.profile['name']}! What is your age group?")
        ss.stage = "need_age"
        st.rerun()

elif ss.stage == "need_age":
    age_choice = st.selectbox(
        "Select your age (years)", list(range(1, 101)),
        index=None, placeholder="Choose your age", key="age_select",
    )
    if st.button("Continue ➜", use_container_width=True, key="age_continue", disabled=age_choice is None):
        user_says(f"{age_choice} years")
        ss.profile["age"] = age_choice
        ss.profile["age_group"] = age_group_for(age_choice)
        bot("Got it. And your gender?")
        ss.stage = "need_gender"
        st.rerun()

elif ss.stage == "need_gender":
    cols = st.columns(len(GENDERS))
    for c, label in zip(cols, GENDERS):
        if c.button(label, use_container_width=True, key=f"gender_{label}"):
            user_says(label)
            ss.profile["gender"] = "Other" if "Other" in label else label
            bot("Thanks. Do you know your blood group? This helps in an emergency.")
            ss.stage = "need_blood"
            st.rerun()

elif ss.stage == "need_blood":
    cols = st.columns(len(BLOOD_GROUPS))
    for c, label in zip(cols, BLOOD_GROUPS):
        if c.button(label, use_container_width=True, key=f"blood_{label}"):
            user_says(label)
            ss.profile["blood_group"] = label
            proceed_to_analysis()
            st.rerun()

elif ss.stage == "ready":
    txt = chat_box("Ask about more symptoms or tap 🎤 and speak...", key="in_ready")
    if txt:
        handle_symptom_text(txt)
    voice_widget()


# ============================================================
# DOCTOR CONSULTATION + REGISTRATION + APPOINTMENT BOOKING
# Shown after the existing symptom analysis; does not replace it.
# ============================================================
if ss.stage == "ready":
    st.markdown("---")
    st.subheader("👩‍⚕️ Would you like to consult a doctor?")
    st.caption("The chatbot provides general information, not a diagnosis. A qualified clinician makes medical decisions.")
    yes_col, no_col = st.columns(2)
    if yes_col.button("✅ Yes, consult a doctor", use_container_width=True, key="consult_yes"):
        ss.consultation_choice = "yes"
        ss.stage = "consult_registration"
        st.rerun()
    if no_col.button("No, continue chatting", use_container_width=True, key="consult_no"):
        ss.consultation_choice = "no"
        st.info("Okay. You can continue chatting about your symptoms above.")

elif ss.stage == "consult_registration":
    st.markdown("---")
    st.subheader("🧑 Patient registration for consultation")
    st.info("Please check the details before submitting. This sends the registration to your local Nalam API.")
    with st.form("nalam_chat_patient_registration"):
        reg_name = st.text_input("Patient name", value=str(ss.profile.get("name") or ""))
        reg_age = st.number_input("Age", min_value=1, max_value=120, value=int(ss.profile.get("age") or 25))
        gender_options = ["Male", "Female", "Other"]
        current_gender = ss.profile.get("gender")
        gender_index = gender_options.index(current_gender) if current_gender in gender_options else 0
        reg_gender = st.selectbox("Gender", gender_options, index=gender_index)
        reg_symptoms = st.text_area("Current symptoms", value=str(ss.pending_text or ""), height=90)
        reg_submit = st.form_submit_button("Register and continue ➜", use_container_width=True)
    if reg_submit:
        if not reg_name.strip() or not reg_symptoms.strip():
            st.error("Please enter patient name and symptoms.")
        else:
            payload = {
                "patient_name": reg_name.strip(),
                "age": int(reg_age),
                "gender": reg_gender,
                "symptoms": reg_symptoms.strip(),
            }
            with st.spinner("Registering patient..."):
                result, error = nalam_api_request("/patients", method="POST", payload=payload)
            if error:
                st.error(error)
                st.caption("No booking was submitted. Check the API docs for the required patient fields.")
            else:
                patient_id = find_patient_id(result)
                if not patient_id:
                    st.error("Registration API responded, but a patient ID could not be found in the response. See response below and confirm its field name.")
                    st.json(result)
                else:
                    ss.consult_patient_id = patient_id
                    ss.consult_patient_response = result
                    ss.profile["name"] = reg_name.strip()
                    ss.profile["age"] = int(reg_age)
                    ss.profile["age_group"] = age_group_for(int(reg_age))
                    ss.profile["gender"] = reg_gender
                    ss.pending_text = reg_symptoms.strip()
                    ss.stage = "appointment_booking"
                    st.rerun()
    if st.button("← Back", key="consult_back"):
        ss.stage = "ready"
        st.rerun()

elif ss.stage == "appointment_booking":
    st.markdown("---")
    st.subheader("📅 Choose a doctor and available appointment slot")
    st.success(f"Patient registered. Patient ID: {ss.consult_patient_id}")
    doctors, doctor_error = nalam_api_request("/doctors")
    if doctor_error:
        st.error(doctor_error)
    elif not isinstance(doctors, list) or not doctors:
        st.warning("No doctors were returned by the API.")
    else:
        active_doctors = [d for d in doctors if d.get("is_active", True)]
        specialties = sorted({str(d.get("specialty") or "General") for d in active_doctors})
        # Lightweight specialty suggestion based on symptom text; patient can change it.
        symptom_text = (ss.pending_text or "").lower()
        suggested = "Dermatologist" if any(w in symptom_text for w in ["rash", "skin", "itch", "pimple", "acne"]) else (
            "Pulmonologist" if any(w in symptom_text for w in ["breath", "wheez", "asthma", "lung"]) else "General Physician")
        default_specialty = suggested if suggested in specialties else (specialties[0] if specialties else "General")
        specialty = st.selectbox("Specialty", specialties, index=specialties.index(default_specialty) if default_specialty in specialties else 0, key="consult_specialty")
        specialty_doctors = [d for d in active_doctors if str(d.get("specialty") or "General") == specialty]
        if specialty_doctors:
            doctor_labels = {f"{d.get('doctor_name', 'Doctor')} — {d.get('specialty', '')} ({d.get('doctor_id', '')})": d for d in specialty_doctors}
            selected_label = st.selectbox("Doctor", list(doctor_labels.keys()), key="consult_doctor")
            selected_doctor = doctor_labels[selected_label]
            doctor_id = str(selected_doctor.get("doctor_id", ""))
            slots, slots_error = nalam_api_request(f"/doctors/{quote_plus(doctor_id)}/free-slots")
            if slots_error:
                st.error(slots_error)
            elif not isinstance(slots, list) or not slots:
                st.warning("This doctor has no slots listed by the API right now. Choose another doctor or check availability later.")
            else:
                usable_slots = [slot for slot in slots if slot.get("date") and slot.get("time")]
                slot_labels = [f"{slot['date']} at {slot['time']}" for slot in usable_slots]
                if slot_labels:
                    chosen_slot_label = st.selectbox("Available date and time", slot_labels, key=f"slot_{doctor_id}")
                    chosen_slot = usable_slots[slot_labels.index(chosen_slot_label)]
                    st.caption("The API slot list is checked before submission; the backend must still prevent double-booking at save time.")
                    if st.button("📌 Book appointment", type="primary", use_container_width=True, key="submit_nalam_appointment"):
                        booking_payload = {
                            "patient_id": ss.consult_patient_id,
                            "doctor_id": doctor_id,
                            "appointment_date": chosen_slot["date"],
                            "appointment_time": chosen_slot["time"],
                        }
                        with st.spinner("Submitting appointment..."):
                            booking_result, booking_error = nalam_api_request("/appointments", method="POST", payload=booking_payload)
                        if booking_error:
                            st.error(booking_error)
                        else:
                            ss.consult_booking_response = booking_result
                            st.success("Appointment request submitted successfully!")
                            if isinstance(booking_result, dict):
                                appointment_id = (booking_result.get("appointment_id") or
                                                  booking_result.get("AppointmentID") or
                                                  booking_result.get("id") or "Not returned")
                                status = booking_result.get("status") or booking_result.get("Status") or "Pending"
                                st.markdown("### 🗓️ Appointment details")
                                st.write(f"**Appointment ID:** {appointment_id}")
                                st.write(f"**Patient:** {ss.profile.get('name') or 'Registered patient'}")
                                st.write(f"**Doctor:** {selected_doctor.get('doctor_name', 'Selected doctor')}")
                                st.write(f"**Specialty:** {selected_doctor.get('specialty', specialty)}")
                                st.write(f"**Date:** {chosen_slot['date']}")
                                st.write(f"**Time:** {chosen_slot['time']}")
                                if status == "Pending":
                                    st.info("**Status: Pending** — doctor response-kaga wait pannunga.")
                                elif status == "Confirmed":
                                    st.success("**Status: Confirmed** — appointment confirm aagiduchu.")
                                elif status == "Rejected":
                                    st.warning("**Status: Rejected** — vera slot/doctor select pannunga.")
                                else:
                                    st.info(f"**Status:** {status}")
                                message = booking_result.get("message")
                                if message:
                                    st.caption(str(message))
                                notice = booking_result.get("notice")
                                if notice:
                                    st.warning(str(notice))
                                st.caption("Indha details-ai appointment status check pannumbodhu use pannalaam.")
                            else:
                                st.info(str(booking_result))
    if st.button("← Back", key="booking_back"):
        ss.stage = "ready"
        st.rerun()


# ============================================================
# DOCTOR LOGIN / ACCEPT-REJECT DASHBOARD
# Only shown when the user chose "Doctor" on the first screen.
# ============================================================
if ss.stage == "doctor":
    st.markdown("---")
    st.header("👨‍⚕️ Doctor Login & Appointment Decisions")
    st.caption("Demo portal. The API must enforce authentication and doctor ownership on protected endpoints.")
    ss.setdefault("doctor_portal_session", None)

    session = ss.doctor_portal_session
    if session is None:
        with st.form("nalam_doctor_login_form"):
            doctor_username = st.text_input("Doctor username", placeholder="Use the account configured in your API")
            doctor_password = st.text_input("Doctor password", type="password")
            doctor_login_submit = st.form_submit_button("🔐 Doctor Login", use_container_width=True)
        if doctor_login_submit:
            if not doctor_username.strip() or not doctor_password:
                st.warning("Username and password enter pannunga.")
            else:
                login_result, login_error = nalam_api_request(
                    "/doctor/login", method="POST",
                    payload={"username": doctor_username.strip(), "password": doctor_password}
                )
                if login_error:
                    st.error(login_error)
                elif isinstance(login_result, dict) and login_result.get("doctor_id") and login_result.get("access_token"):
                    ss.doctor_portal_session = {
                        "doctor_id": str(login_result["doctor_id"]),
                        "username": str(login_result.get("username", doctor_username.strip())),
                        "access_token": str(login_result["access_token"]),
                    }
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Login response-la doctor_id/access_token illai.")
                    st.json(login_result)
    else:
        st.success(f"Logged in: {session['username']} ({session['doctor_id']})")
        top_a, top_b = st.columns([3, 1])
        with top_a:
            refresh_doctor_list = st.button("🔄 Refresh appointments", key="doctor_refresh")
        with top_b:
            logout_doctor = st.button("🚪 Logout", key="doctor_logout")
        if logout_doctor:
            ss.doctor_portal_session = None
            st.rerun()
        doctor_id = session["doctor_id"]
        appointments, appointments_error = nalam_api_request(
            f"/doctors/{quote_plus(doctor_id)}/appointments",
            token=session["access_token"]
        )
        if appointments_error:
            st.error(appointments_error)
            st.info("If this endpoint returns 401/403, the backend token/auth configuration needs to be checked.")
        elif not isinstance(appointments, list) or not appointments:
            st.info("Indha doctor-ku appointment requests illai.")
        else:
            pending = [a for a in appointments if str(a.get("status", a.get("Status", ""))).lower() == "pending"]
            st.metric("Pending requests", len(pending))
            for i, apt in enumerate(appointments):
                apt_id = str(apt.get("appointment_id") or apt.get("AppointmentID") or apt.get("id") or "")
                apt_status = str(apt.get("status") or apt.get("Status") or "Unknown")
                with st.container(border=True):
                    st.subheader(f"Appointment {apt_id or '(ID not returned)'}")
                    st.write(f"**Patient ID:** {apt.get('patient_id', apt.get('PatientID', '—'))}")
                    st.write(f"**Date:** {apt.get('appointment_date', apt.get('AppointmentDate', '—'))}")
                    st.write(f"**Time:** {apt.get('appointment_time', apt.get('AppointmentTime', '—'))}")
                    st.write(f"**Status:** {apt_status}")
                    if apt_status.lower() == "pending" and apt_id:
                        accept_col, reject_col = st.columns(2)
                        if accept_col.button("✅ Accept appointment", key=f"accept_{apt_id}_{i}", use_container_width=True):
                            decision, decision_error = nalam_api_request(
                                f"/appointments/{quote_plus(apt_id)}/decision", method="PUT",
                                payload={"status": "Confirmed"}, token=session["access_token"]
                            )
                            if decision_error:
                                st.error(decision_error)
                            else:
                                st.success("Appointment accepted. Patient status can now show Confirmed.")
                                st.rerun()
                        if reject_col.button("❌ Reject appointment", key=f"reject_{apt_id}_{i}", use_container_width=True):
                            decision, decision_error = nalam_api_request(
                                f"/appointments/{quote_plus(apt_id)}/decision", method="PUT",
                                payload={"status": "Rejected"}, token=session["access_token"]
                            )
                            if decision_error:
                                st.error(decision_error)
                            else:
                                st.warning("Appointment rejected.")
                                st.rerun()
                    elif apt_status.lower() == "confirmed" and apt_id:
                        meeting_link = f"https://meet.jit.si/NalamHealth-{''.join(ch for ch in apt_id.upper() if ch.isalnum() or ch == '-') }"
                        st.link_button("🎥 Join patient video call", meeting_link, use_container_width=True)
                        st.caption("Same demo meeting URL is used by the patient status screen. It is not private/authenticated.")

    st.markdown("---")
    if st.button("← Back", key="doctor_back"):
        new_conversation()
        st.rerun()


# ============================================================
# PATIENT: CHECK APPOINTMENT STATUS
# Only shown when the user chose "Patient" on the first screen.
# ============================================================
if ss.role == "patient":
    st.markdown("---")
    st.markdown("### 🧑 Patient: Check appointment status / video link")
    with st.form("nalam_patient_status_form"):
        status_appointment_id = st.text_input("Appointment ID", placeholder="Paste the appointment ID from booking")
        status_submit = st.form_submit_button("🔎 Check status", use_container_width=True)
    if status_submit:
        if not status_appointment_id.strip():
            st.warning("Appointment ID enter pannunga.")
        else:
            status_result, status_error = nalam_api_request(f"/appointments/{quote_plus(status_appointment_id.strip())}")
            if status_error:
                st.error(status_error)
            elif isinstance(status_result, dict):
                current_status = str(status_result.get("status") or status_result.get("Status") or "Unknown")
                st.markdown("#### Appointment details")
                st.write(f"**Appointment ID:** {status_result.get('appointment_id', status_result.get('AppointmentID', status_appointment_id.strip()))}")
                st.write(f"**Date:** {status_result.get('appointment_date', status_result.get('AppointmentDate', '—'))}")
                st.write(f"**Time:** {status_result.get('appointment_time', status_result.get('AppointmentTime', '—'))}")
                if current_status.lower() == "confirmed":
                    apt_id_for_link = str(status_result.get("appointment_id") or status_result.get("AppointmentID") or status_appointment_id.strip())
                    meeting_link = f"https://meet.jit.si/NalamHealth-{''.join(ch for ch in apt_id_for_link.upper() if ch.isalnum() or ch == '-') }"
                    st.success("Appointment Confirmed")
                    st.link_button("🎥 Join video consultation", meeting_link, use_container_width=True)
                    st.warning("Demo Jitsi link only; anyone with the URL may be able to join. Do not use for real patient data without private-room access controls.")
                elif current_status.lower() == "pending":
                    st.info("Appointment Pending — doctor accept/reject pannum varaikkum wait pannunga.")
                elif current_status.lower() == "rejected":
                    st.error("Appointment Rejected — another slot/doctor choose pannunga.")
                else:
                    st.write(f"**Status:** {current_status}")
            else:
                st.json(status_result)


# ============================================================
# FOOTER
# ============================================================

render(f"""
<div class="footer">
    <b>NALAM HEALTH AI</b> &nbsp;—&nbsp; deployed for <b>{esc(HOSPITAL_NAME)}</b><br>
    Custom Healthcare Information Assistant<br>
    Dataset-based symptom matching • Safety-oriented information, not a diagnosis
</div>
""")