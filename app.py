import time
import math
import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForCausalLM
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from threading import Thread
import torch

# Page config
st.set_page_config(page_title="PersonaLM", page_icon="🧠", layout="wide")

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161a23;
    --border: #252a38;
    --accent: #7c6af7;
    --accent-soft: rgba(124,106,247,0.12);
    --text: #e8eaf0;
    --muted: #7a8099;
    --success: #4ecca3;
}

html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] { background-color: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: var(--text) !important; }

/* Centered pages (intro, quiz, generating) */
.center-wrap { max-width: 640px; margin: 0 auto; padding: 3rem 1rem; }

/* Wide results layout */
.block-container { padding: 2rem 2rem !important; }

/* Hero */
.hero-tag {
    display: inline-block; background: var(--accent-soft); color: var(--accent);
    border: 1px solid rgba(124,106,247,0.3); border-radius: 999px;
    padding: 4px 14px; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif; font-size: 3.2rem; line-height: 1.1;
    margin: 0 0 0.6rem 0;
    background: linear-gradient(135deg, #e8eaf0 30%, var(--accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: var(--muted); font-size: 1.05rem; line-height: 1.65; margin-bottom: 2rem; }

.stat-row { display: flex; gap: 1.5rem; margin-bottom: 2rem; }
.stat { text-align: center; flex: 1; background: var(--accent-soft); border-radius: 12px; padding: 0.9rem 0.5rem; }
.stat-num   { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
.stat-label { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

.info-box {
    background: rgba(78,204,163,0.08); border: 1px solid rgba(78,204,163,0.25);
    border-radius: 12px; padding: 1rem 1.2rem;
    color: var(--success); font-size: 0.9rem; margin: 1rem 0;
}

/* Quiz */
@keyframes slideIn {
    from { opacity: 0; transform: translateX(28px); }
    to { opacity: 1; transform: translateX(0); }
}
.q-card {
    animation: slideIn 0.38s cubic-bezier(0.22,1,0.36,1) both;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 20px; padding: 2rem 2rem 1.5rem 2rem; margin-bottom: 1.5rem;
}
.q-meta { color: var(--muted); font-size: 0.8rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 0.8rem; }
.q-text { font-family: 'DM Serif Display', serif; font-size: 1.55rem; line-height: 1.35; color: var(--text); margin: 0; }

/* Radio */
[data-testid="stRadio"] > label { display: none; }
[data-testid="stRadio"] div[role="radiogroup"] { gap: 10px; }
[data-testid="stRadio"] div[role="radiogroup"] > label {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; padding: 14px 18px !important;
    cursor: pointer; transition: all 0.18s ease; width: 100%;
}
[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    border-color: var(--accent) !important; background: var(--accent-soft) !important;
}

/* Progress */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--success)) !important;
    border-radius: 999px !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--border) !important; border-radius: 999px !important; height: 6px !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important; color: white !important;
    border: none !important; border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    padding: 0.7rem 1.8rem !important; transition: opacity 0.18s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stButton > button[kind="secondary"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--muted) !important;
}

/* Result box */
.result-box {
    background: var(--surface); border-left: 3px solid var(--accent);
    border-radius: 0 12px 12px 0; padding: 1.4rem 1.6rem;
    font-size: 0.95rem; line-height: 1.8; color: var(--text); margin: 0.8rem 0;
}

/* Right col sticky */
.right-sticky { position: sticky; top: 2rem; }

.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
audio { width: 100%; border-radius: 12px; margin-top: 0.5rem; }

/* Section fade-in */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.5s ease both; }
</style>
""", unsafe_allow_html=True)


# Constants
QUESTIONS = [
    {"text": "I have a vivid imagination and love to daydream.", "trait": "O", "reverse": False},
    {"text": "I enjoy exploring abstract ideas and theories.", "trait": "O", "reverse": False},
    {"text": "I prefer sticking to tried-and-true methods over experimenting.", "trait": "O", "reverse": True},
    {"text": "I rarely get lost in artistic or creative experiences.", "trait": "O", "reverse": True},
    {"text": "I plan ahead and always come prepared.", "trait": "C", "reverse": False},
    {"text": "I pay close attention to details in everything I do.", "trait": "C", "reverse": False},
    {"text": "I often leave tasks unfinished when something else catches my eye.", "trait": "C", "reverse": True},
    {"text": "I tend to be disorganized in my daily life.", "trait": "C", "reverse": True},
    {"text": "I feel energised by being around a lot of people.", "trait": "E", "reverse": False},
    {"text": "I enjoy being the center of attention.", "trait": "E", "reverse": False},
    {"text": "I prefer to stay in the background at social gatherings.", "trait": "E", "reverse": True},
    {"text": "I find small talk draining and avoid it when I can.", "trait": "E", "reverse": True},
    {"text": "I genuinely care about others' feelings and wellbeing.", "trait": "A", "reverse": False},
    {"text": "I find it easy to get along with almost anyone.", "trait": "A", "reverse": False},
    {"text": "I tend to be critical or skeptical of other people's motives.", "trait": "A", "reverse": True},
    {"text": "I sometimes say things that make others feel uncomfortable.", "trait": "A", "reverse": True},
    {"text": "I get stressed out or overwhelmed quite easily.", "trait": "N", "reverse": False},
    {"text": "My mood can shift quickly in response to events around me.", "trait": "N", "reverse": False},
    {"text": "I rarely feel anxious or worried about things.", "trait": "N", "reverse": True},
    {"text": "I stay calm and composed even under pressure.", "trait": "N", "reverse": True},
]

ANSWER_OPTIONS = [
    "Very Inaccurate", "Moderately Inaccurate",
    "Neither Accurate nor Inaccurate",
    "Moderately Accurate", "Very Accurate",
]

TRAIT_META = {
    "O": {"label": "Openness", "emoji": "🌌", "color": "#a78bfa"},
    "C": {"label": "Conscientiousness", "emoji": "📐", "color": "#60a5fa"},
    "E": {"label": "Extraversion", "emoji": "⚡", "color": "#fbbf24"},
    "A": {"label": "Agreeableness", "emoji": "🤝", "color": "#4ecca3"},
    "N": {"label": "Neuroticism", "emoji": "🌊", "color": "#f87171"},
}

ARCHETYPES = {
    "O": {"name": "The Visionary", "desc": "You see what others miss — patterns, possibilities, and worlds not yet built.", "icon": "✦", "grad": "linear-gradient(135deg,#7c3aed,#a78bfa,#4ecca3)"},
    "C": {"name": "The Architect", "desc": "You build with intention. Precision, reliability, and excellence define your path.", "icon": "◈", "grad": "linear-gradient(135deg,#1d4ed8,#60a5fa,#7c6af7)"},
    "E": {"name": "The Catalyst", "desc": "Rooms light up when you enter. You turn strangers into allies and ideas into movements.", "icon": "◉", "grad": "linear-gradient(135deg,#b45309,#fbbf24,#f97316)"},
    "A": {"name": "The Empath", "desc": "You read the room before anyone speaks. Connection and harmony are your superpowers.", "icon": "❋", "grad": "linear-gradient(135deg,#065f46,#4ecca3,#60a5fa)"},
    "N": {"name": "The Sentinel", "desc": "You feel everything deeply. Your emotional richness fuels unmatched creativity and insight.", "icon": "◬", "grad": "linear-gradient(135deg,#991b1b,#f87171,#fbbf24)"},
}

TRAIT_INSIGHTS = {
    "O": {"strength": ["Creative thinker", "Open to new ideas", "Intellectually curious"], "blindspot": ["Can seem impractical", "Prone to distraction", "Difficulty with routine"],       "growth": ["Build consistent habits", "Balance imagination with execution"]},
    "C": {"strength": ["Highly reliable", "Detail-oriented", "Self-disciplined"], "blindspot": ["Perfectionism", "Rigidity under pressure", "Overplanning"],                       "growth": ["Embrace flexibility", "Practice letting go of control"]},
    "E": {"strength": ["Natural leader", "Energises others", "Thrives socially"], "blindspot": ["Can dominate conversations", "Undervalues solitude", "Impulsive decisions"],      "growth": ["Cultivate deep listening", "Spend intentional time alone"]},
    "A": {"strength": ["Highly empathetic", "Collaborative", "Trusted by peers"], "blindspot": ["People-pleasing", "Avoids conflict", "Difficulty saying no"],                     "growth": ["Develop assertiveness", "Set healthy boundaries"]},
    "N": {"strength": ["Deep emotional awareness", "Empathetic to pain", "Highly self-reflective"], "blindspot": ["Prone to overthinking", "Mood-dependent decisions", "Catastrophising"],          "growth": ["Build emotional regulation habits", "Mindfulness practices"]},
}

TOTAL_Q = len(QUESTIONS)


# Cached resources
@st.cache_resource(show_spinner=False)
def load_model():
    tok = AutoTokenizer.from_pretrained("SanyaAhmed/llm-personality-model")
    mdl = AutoModelForCausalLM.from_pretrained("SanyaAhmed/llm-personality-model")
    mdl.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_kokoro():
    return KPipeline(lang_code='a')


# Helpers
def compute_traits(answers):
    buckets = {k: [] for k in TRAIT_META}
    for i, q in enumerate(QUESTIONS):
        raw = answers[i] + 1
        score = (6 - raw) if q["reverse"] else raw
        buckets[q["trait"]].append(score)
    return {t: round(np.mean(v) / 5, 2) for t, v in buckets.items()}

def dominant_trait(traits):
    return max(traits, key=traits.get)

def get_sections(traits):
    ranked = sorted(traits.items(), key=lambda x: x[1], reverse=True)
    top2 = [k for k, _ in ranked[:2]]
    bottom = ranked[-1][0]
    mid = [k for k, _ in ranked[2:4]]
    strengths = []
    blindspots = []
    growth = []
    for t in top2:
        strengths.extend(TRAIT_INSIGHTS[t]["strength"])
    blindspots.extend(TRAIT_INSIGHTS[bottom]["blindspot"])
    for t in mid:
        growth.extend(TRAIT_INSIGHTS[t]["growth"])
    return strengths[:4], blindspots[:3], growth[:3]

def generate_text(traits, tokenizer, model):
    trait_str = ", ".join(f"{TRAIT_META[k]['label']}: {v}" for k, v in traits.items())
    inputs = tokenizer(f"Input: {trait_str}", return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=300, temperature=0.85, top_p=0.92,
            top_k=50, repetition_penalty=1.3, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def generate_audio(text, koko):
    chunks = []
    for _, _, audio in koko(text, voice="af_heart", speed=1, split_pattern=r"\n+"):
        chunks.append(audio)
    if chunks:
        sf.write("output_audio.wav", np.concatenate(chunks), 24000)
        return "output_audio.wav"
    return None

def background_generate(traits, tokenizer, model, koko, result):
    try:
        text = generate_text(traits, tokenizer, model).replace("Output: ", "").strip()
        result["text"] = text
        result["audio"] = generate_audio(text, koko)
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["done"] = True

def simulate_typing(text, placeholder):
    words = text.split()
    displayed = ""
    for word in words:
        displayed += word + " "
        placeholder.markdown(
            f'<div class="result-box">{displayed}<span style="opacity:0.4;">▍</span></div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.045)
    placeholder.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)


# Radar chart
def render_radar(traits):
    keys = list(TRAIT_META.keys())
    colors = [TRAIT_META[k]["color"] for k in keys]
    scores = [traits[k] for k in keys]

    cx, cy, R = 275, 215, 115
    N = 5
    angles = [math.pi / 2 - 2 * math.pi * i / N for i in range(N)]

    def pt(r, a): return cx + r * math.cos(a), cy - r * math.sin(a)

    grid = "".join(
        '<path d="' + " ".join(
            f"{'M' if i==0 else 'L'}{pt(R*lv,a)[0]:.1f},{pt(R*lv,a)[1]:.1f}"
            for i, a in enumerate(angles)
        ) + 'Z" fill="none" stroke="#252a38" stroke-width="1"/>'
        for lv in [0.25, 0.5, 0.75, 1.0]
    )
    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{pt(R,a)[0]:.1f}" y2="{pt(R,a)[1]:.1f}" stroke="#252a38" stroke-width="1"/>'
        for a in angles
    )
    data_pts = [pt(R * scores[i], angles[i]) for i in range(N)]
    data_d = " ".join(f"{'M' if i==0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(data_pts)) + "Z"

    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[i]}" opacity="0">'
        f'<animate attributeName="opacity" from="0" to="1" dur="0.3s" begin="{0.6+i*0.08:.2f}s" fill="freeze"/>'
        f'</circle>'
        for i, (x, y) in enumerate(data_pts)
    )

    label_svg = ""
    for i, a in enumerate(angles):
        lx, ly = pt(R + 52, a)
        anchor = "middle"
        if lx < cx - 20: anchor = "end"
        elif lx > cx + 20: anchor = "start"
        emoji = TRAIT_META[keys[i]]["emoji"]
        lbl = TRAIT_META[keys[i]]["label"]
        col = colors[i]
        label_svg += (
            f'<text x="{lx:.1f}" y="{ly - 8:.1f}" text-anchor="{anchor}" dominant-baseline="auto" '
            f'font-size="15" font-family="Segoe UI Emoji,Apple Color Emoji,sans-serif">{emoji}</text>'
            f'<text x="{lx:.1f}" y="{ly + 10:.1f}" text-anchor="{anchor}" dominant-baseline="auto" '
            f'fill="{col}" font-size="11" font-weight="600" font-family="DM Sans,sans-serif">{lbl}</text>'
        )

    score_svg = "".join(
        f'<text x="{pt(R*scores[i]+18, angles[i])[0]:.1f}" y="{pt(R*scores[i]+18, angles[i])[1]:.1f}" '
        f'text-anchor="middle" dominant-baseline="middle" fill="{colors[i]}" '
        f'font-size="10" font-weight="700" font-family="DM Sans,sans-serif" opacity="0">'
        f'{int(scores[i]*100)}%'
        f'<animate attributeName="opacity" from="0" to="1" dur="0.3s" begin="{0.7+i*0.08:.2f}s" fill="freeze"/>'
        f'</text>'
        for i in range(N)
    )

    svg = f"""
    <svg viewBox="0 0 600 530" xmlns="http://www.w3.org/2000/svg"
         style="width:100%;display:block;margin:0 auto;">
      <defs>
        <linearGradient id="rf" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%"   stop-color="#7c6af7" stop-opacity="0.4"/>
          <stop offset="100%" stop-color="#4ecca3" stop-opacity="0.2"/>
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {grid}{axes}
      <path d="M{cx},{cy}" fill="url(#rf)" stroke="none">
        <animate attributeName="d" dur="0.7s" calcMode="spline" keySplines="0.22 1 0.36 1"
          from="M{cx},{cy} M{cx},{cy} M{cx},{cy} M{cx},{cy} M{cx},{cy}Z"
          to="{data_d}" fill="freeze"/>
      </path>
      <path d="M{cx},{cy}" fill="none" stroke="#7c6af7" stroke-width="2" filter="url(#glow)">
        <animate attributeName="d" dur="0.7s" calcMode="spline" keySplines="0.22 1 0.36 1"
          from="M{cx},{cy} M{cx},{cy} M{cx},{cy} M{cx},{cy} M{cx},{cy}Z"
          to="{data_d}" fill="freeze"/>
      </path>
      {dots}{label_svg}{score_svg}
    </svg>"""

    components.html(
        f'<style>body{{margin:0;background:transparent;overflow:hidden;}}</style>{svg}',
        height=530,
    )


# Archetype card
def render_archetype(traits):
    dt = dominant_trait(traits)
    arc = ARCHETYPES[dt]
    components.html(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;600&display=swap');
        *{{box-sizing:border-box;margin:0;padding:0;}}
        body{{background:transparent;font-family:'DM Sans',sans-serif;}}
        .card{{background:{arc['grad']};border-radius:20px;padding:1.8rem 2rem;position:relative;overflow:hidden;}}
        .card::before{{content:'';position:absolute;inset:0;background:rgba(13,15,20,0.42);border-radius:20px;}}
        .inner{{position:relative;z-index:1;}}
        .tag{{display:inline-block;background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.25);
              border-radius:999px;padding:3px 12px;font-size:0.72rem;font-weight:700;
              letter-spacing:0.1em;text-transform:uppercase;color:#fff;margin-bottom:0.9rem;}}
        .icon{{font-size:2.4rem;line-height:1;margin-bottom:0.5rem;display:block;}}
        .name{{font-family:'DM Serif Display',serif;font-size:1.8rem;color:#fff;margin-bottom:0.4rem;}}
        .desc{{color:rgba(255,255,255,0.82);font-size:0.92rem;line-height:1.6;}}
    </style>
    <div class="card"><div class="inner">
        <div class="tag">Your Archetype</div>
        <span class="icon">{arc['icon']}</span>
        <div class="name">{arc['name']}</div>
        <div class="desc">{arc['desc']}</div>
    </div></div>
    """, height=210)


# Insights
def render_insights(traits):
    strengths, blindspots, growth = get_sections(traits)
    def tags(items, bg, col):
        return "".join(f'<span class="tag" style="background:{bg};color:{col};">{item}</span>' for item in items)
    components.html(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');
        *{{box-sizing:border-box;margin:0;padding:0;}}
        body{{background:transparent;font-family:'DM Sans',sans-serif;}}
        .sec{{background:#161a23;border:1px solid #252a38;border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:0.7rem;}}
        .title{{font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.65rem;}}
        .tag{{display:inline-block;border-radius:999px;padding:4px 11px;font-size:0.8rem;font-weight:500;margin:3px;}}
    </style>
    <div class="sec"><div class="title" style="color:#4ecca3;">✦ Strengths</div>
        {tags(strengths,'rgba(78,204,163,0.12)','#4ecca3')}</div>
    <div class="sec"><div class="title" style="color:#f87171;">◈ Blind Spots</div>
        {tags(blindspots,'rgba(248,113,113,0.12)','#f87171')}</div>
    <div class="sec"><div class="title" style="color:#fbbf24;">◉ Growth Areas</div>
        {tags(growth,'rgba(251,191,36,0.12)','#fbbf24')}</div>
    """, height=290)


# Loader
def show_loader(placeholder):
    placeholder.empty()
    with placeholder.container():
        components.html("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600&display=swap');
            *{box-sizing:border-box;margin:0;padding:0;}
            body{background:transparent;font-family:'DM Sans',sans-serif;display:flex;justify-content:center;}
            .wrap{display:flex;flex-direction:column;align-items:center;gap:1.6rem;
                  padding:2.4rem 2rem;width:100%;max-width:480px;
                  background:#161a23;border:1px solid #252a38;border-radius:20px;}
            .orb{position:relative;width:72px;height:72px;}
            .orb::before,.orb::after{content:'';position:absolute;border-radius:50%;}
            .orb::before{inset:0;background:radial-gradient(circle at 35% 35%,#9d8fff,#7c6af7 50%,#4ecca3);
                         animation:pulse 2s ease-in-out infinite;filter:blur(1px);}
            .orb::after{inset:-6px;border:2px solid transparent;
                        border-top-color:#7c6af7;border-right-color:#4ecca3;
                        animation:spin 1.4s linear infinite;}
            @keyframes pulse{0%,100%{transform:scale(1);opacity:1;}50%{transform:scale(1.1);opacity:0.8;}}
            @keyframes spin{to{transform:rotate(360deg);}}
            .msgs{height:1.4rem;overflow:hidden;text-align:center;}
            .msg-list{display:flex;flex-direction:column;animation:scroll 8s steps(1) infinite;}
            .msg{height:1.4rem;line-height:1.4rem;font-size:0.95rem;font-weight:600;color:#e8eaf0;white-space:nowrap;}
            @keyframes scroll{
                0%{transform:translateY(0);}25%{transform:translateY(-1.4rem);}
                50%{transform:translateY(-2.8rem);}75%{transform:translateY(-4.2rem);}
                100%{transform:translateY(0);}}
            .dots{display:flex;gap:6px;}
            .dot{width:7px;height:7px;border-radius:50%;background:#7c6af7;animation:bounce 1.2s ease-in-out infinite;}
            .dot:nth-child(2){animation-delay:0.2s;background:#9d8fff;}
            .dot:nth-child(3){animation-delay:0.4s;background:#4ecca3;}
            @keyframes bounce{0%,80%,100%{transform:translateY(0);opacity:0.5;}40%{transform:translateY(-8px);opacity:1;}}
            .sub{color:#7a8099;font-size:0.78rem;letter-spacing:0.06em;text-transform:uppercase;}
        </style>
        <div class="wrap">
            <div class="orb"></div>
            <div class="msgs"><div class="msg-list">
                <div class="msg">Decoding your psyche…</div>
                <div class="msg">Mapping your inner universe…</div>
                <div class="msg">Consulting the Big Five…</div>
                <div class="msg">Weaving your narrative…</div>
            </div></div>
            <div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
            <div class="sub">Your story is being written — almost there</div>
        </div>
        """, height=260)


# Session state
for key, default in [
    ("page", "intro"),
    ("current_q", 0),
    ("answers", [2] * TOTAL_Q),
    ("traits", None),
    ("generated_text", None),
    ("audio_file", None),
    ("results_fresh", False),   # True only on first-ever visit to results
]:
    if key not in st.session_state:
        st.session_state[key] = default


# PAGE: INTRO
if st.session_state.page == "intro":
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="hero-tag">Big Five · AI-Powered</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">PersonaLM</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-sub">Answer 20 questions from the IPIP Big Five framework. '
            "Our fine-tuned model analyses your scores and generates a personalised "
            "narrative with your archetype, strengths, blind spots, and growth areas.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div class="stat-row">
            <div class="stat"><div class="stat-num">20</div><div class="stat-label">Questions</div></div>
            <div class="stat"><div class="stat-num">5</div><div class="stat-label">Dimensions</div></div>
            <div class="stat"><div class="stat-num">AI</div><div class="stat-label">Narrative</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">✦ &nbsp; No right or wrong answers — be honest for the most accurate profile.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Begin Assessment →", use_container_width=True):
            st.session_state.page = "quiz"
            st.rerun()


# PAGE: QUIZ
elif st.session_state.page == "quiz":
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        idx = st.session_state.current_q
        q = QUESTIONS[idx]
        meta = TRAIT_META[q["trait"]]

        st.progress(idx / TOTAL_Q)

        st.markdown(f"""
        <div class="q-card">
            <div class="q-meta">{meta['emoji']} {meta['label']} &nbsp;·&nbsp; {idx+1} of {TOTAL_Q}</div>
            <p class="q-text">{q['text']}</p>
        </div>
        """, unsafe_allow_html=True)

        selected = st.radio(
            "answer", options=list(range(5)),
            format_func=lambda x: ANSWER_OPTIONS[x],
            index=st.session_state.answers[idx],
            key=f"q_{idx}", label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if idx > 0:
                if st.button("← Back", use_container_width=True, type="secondary"):
                    st.session_state.answers[idx] = selected
                    st.session_state.current_q -= 1
                    st.rerun()
        with col2:
            if idx < TOTAL_Q - 1:
                if st.button("Next →", use_container_width=True):
                    st.session_state.answers[idx] = selected
                    st.session_state.current_q += 1
                    st.rerun()
            else:
                if st.button("See My Profile →", use_container_width=True):
                    st.session_state.answers[idx] = selected
                    st.session_state.page = "generating"
                    st.rerun()


# PAGE: GENERATION
elif st.session_state.page == "generating":

    st.markdown("""
    <style>
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        footer { display: none !important; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

    components.html("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            width: 100vw; height: 100vh;
            background: #0d0f14;
            display: flex; align-items: center; justify-content: center;
            font-family: 'DM Sans', sans-serif;
        }
        .wrap {
            display: flex; flex-direction: column; align-items: center;
            gap: 2rem; padding: 3rem 2.5rem; width: 100%; max-width: 460px;
            background: #161a23; border: 1px solid #252a38; border-radius: 24px;
        }
        .title {
            font-family: 'DM Serif Display', serif;
            font-size: 1.5rem; color: #e8eaf0; text-align: center;
        }
        .orb { position: relative; width: 88px; height: 88px; }
        .orb::before, .orb::after { content: ''; position: absolute; border-radius: 50%; }
        .orb::before {
            inset: 0;
            background: radial-gradient(circle at 35% 35%, #9d8fff, #7c6af7 50%, #4ecca3);
            animation: pulse 2s ease-in-out infinite; filter: blur(1px);
        }
        .orb::after {
            inset: -7px; border: 2px solid transparent;
            border-top-color: #7c6af7; border-right-color: #4ecca3;
            animation: spin 1.4s linear infinite;
        }
        @keyframes pulse { 0%,100%{transform:scale(1);opacity:1;} 50%{transform:scale(1.12);opacity:0.8;} }
        @keyframes spin  { to { transform: rotate(360deg); } }

        .msgs { height: 1.5rem; overflow: hidden; text-align: center; }
        .msg-list { display: flex; flex-direction: column; animation: scroll 8s steps(1) infinite; }
        .msg { height: 1.5rem; line-height: 1.5rem; font-size: 1rem; font-weight: 600; color: #e8eaf0; white-space: nowrap; }
        @keyframes scroll {
            0%  { transform: translateY(0); }       25% { transform: translateY(-1.5rem); }
            50% { transform: translateY(-3rem); }    75% { transform: translateY(-4.5rem); }
            100%{ transform: translateY(0); }
        }
        .dots { display: flex; gap: 8px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; animation: bounce 1.2s ease-in-out infinite; }
        .dot:nth-child(1) { background: #7c6af7; }
        .dot:nth-child(2) { background: #9d8fff; animation-delay: 0.2s; }
        .dot:nth-child(3) { background: #4ecca3; animation-delay: 0.4s; }
        @keyframes bounce { 0%,80%,100%{transform:translateY(0);opacity:0.5;} 40%{transform:translateY(-10px);opacity:1;} }

        .sub { color: #7a8099; font-size: 0.8rem; letter-spacing: 0.07em; text-transform: uppercase; text-align: center; }
    </style>

    <div class="wrap">
        <div class="title">Analysing your responses…</div>
        <div class="orb"></div>
        <div class="msgs">
            <div class="msg-list">
                <div class="msg">Decoding your psyche…</div>
                <div class="msg">Mapping your inner universe…</div>
                <div class="msg">Consulting the Big Five…</div>
                <div class="msg">Weaving your narrative…</div>
            </div>
        </div>
        <div class="dots">
            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        </div>
        <div class="sub">Your story is being written — almost there</div>
    </div>
    """, height=700)

    traits = compute_traits(st.session_state.answers)
    st.session_state.traits = traits

    tokenizer, model = load_model()
    koko = load_kokoro()

    result = {"text": None, "audio": None, "done": False, "error": None}
    thread = Thread(target=background_generate, args=(traits, tokenizer, model, koko, result))
    thread.start()

    while not result["done"]:
        time.sleep(0.4)
    thread.join()

    if result["error"]:
        st.error(f"Generation failed: {result['error']}")
        st.stop()

    st.session_state.generated_text = result["text"]
    st.session_state.audio_file = result["audio"]
    st.session_state.results_fresh = True
    st.session_state.page = "results"
    st.rerun()


# PAGE: RESULTS 
elif st.session_state.page == "results":
    traits = st.session_state.traits
    fresh = st.session_state.results_fresh

    # Header
    st.markdown('<h1 style="margin-bottom:0.1rem;">Your Profile</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--muted);margin-bottom:1.5rem;">Based on the Big Five personality model</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider" style="margin-top:0;">', unsafe_allow_html=True)

    left, gap, right = st.columns([5, 0.3, 4])

    # LEFT COLUMN
    with left:
        # Archetype
        arch_ph = st.empty()
        if fresh:
            time.sleep(0.15)
        with arch_ph.container():
            render_archetype(traits)

        st.markdown('<div style="margin:1rem 0;"></div>', unsafe_allow_html=True)

        # Radar
        st.markdown('<h3 style="margin-bottom:0.2rem;">Trait Dimensions</h3>', unsafe_allow_html=True)
        radar_ph = st.empty()
        if fresh:
            time.sleep(0.2)
        with radar_ph.container():
            render_radar(traits)

        # Insights
        st.markdown('<h3 style="margin-bottom:0.8rem;">Insights</h3>', unsafe_allow_html=True)
        ins_ph = st.empty()
        if fresh:
            time.sleep(0.2)
        with ins_ph.container():
            render_insights(traits)


    # RIGHT COLUMN
    with right:
        st.markdown('<h3 style="margin-bottom:0.5rem;">Personality Narrative</h3>', unsafe_allow_html=True)

        # Audio player
        if st.session_state.audio_file:
            with open(st.session_state.audio_file, "rb") as f:
                audio_bytes = f.read()
            if fresh:
                time.sleep(0.1)
                st.audio(audio_bytes, format="audio/wav", autoplay=True)
            else:
                st.audio(audio_bytes, format="audio/wav")

        # Narrative text
        text_ph = st.empty()
        if fresh:
            simulate_typing(st.session_state.generated_text, text_ph)
            st.session_state.results_fresh = False   # done revealing, mark as visited
        else:
            text_ph.markdown(
                f'<div class="result-box">{st.session_state.generated_text}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        if st.button("↩ Retake Assessment", use_container_width=True, type="secondary"):
            for key in ["page", "current_q", "answers", "traits",
                        "generated_text", "audio_file", "results_fresh"]:
                del st.session_state[key]
            st.rerun()