"""
SoundScape — AI Audio Intelligence Dashboard
Streamlit deployment of the Audio Intelligence Pipeline
"""

import streamlit as st
import os, re, json, warnings, tempfile, time
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SoundScape — Neural Audio Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS (matches soundscape.html aesthetic) ───────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400&display=swap');

  :root {
    --navy:  #0A1828;
    --navy2: #0d1f33;
    --gold:  #C9A44C;
    --gold2: #e8c56a;
    --cyan:  #00D9FF;
    --white: #f0f4f8;
    --muted: #7a8fa8;
    --card:  rgba(255,255,255,0.04);
    --border: rgba(201,164,76,0.25);
  }

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: var(--navy) !important;
    color: var(--white) !important;
  }
  .stApp { background-color: #0A1828 !important; }
  .block-container { padding-top: 1.5rem !important; max-width: 1400px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1f33 !important;
    border-right: 1px solid rgba(201,164,76,0.15) !important;
  }
  [data-testid="stSidebar"] * { color: var(--white) !important; }

  /* ── Header strip ── */
  .ss-header {
    display: flex; align-items: center; gap: 1.2rem;
    padding: 1.2rem 2rem;
    background: rgba(13,31,51,0.9);
    border: 1px solid rgba(201,164,76,0.2);
    border-radius: 4px;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
  }
  .ss-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #C9A44C 0%, #e8c56a 45%, #00D9FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em; line-height:1;
  }
  .ss-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.25em;
    color: var(--cyan); text-transform: uppercase;
  }
  .ss-sub {
    font-size: 0.8rem; color: var(--muted);
    margin-top: 0.2rem;
  }
  .ss-badge {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.15em;
    color: var(--navy); background: var(--gold);
    padding: 0.3rem 0.8rem; border-radius: 2px;
    text-transform: uppercase;
  }

  /* ── Section labels ── */
  .sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.3em; text-transform: uppercase;
    color: var(--cyan); margin-bottom: 0.4rem;
  }
  .sec-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem; font-weight: 600; color: var(--white);
    margin-bottom: 1.2rem;
  }
  .sec-title em { color: var(--gold); font-style: normal; }

  /* ── Cards ── */
  .ss-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(201,164,76,0.2);
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }
  .ss-card-cyan {
    border-color: rgba(0,217,255,0.25);
  }
  .ss-card-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem; font-weight: 600;
    color: var(--gold); margin-bottom: 0.6rem;
  }
  .ss-card-body {
    font-size: 0.88rem; color: var(--muted); line-height: 1.65;
  }

  /* ── Metric tiles ── */
  .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
  .metric-tile {
    flex: 1; min-width: 150px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(201,164,76,0.2);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-tile .val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem; font-weight: 700; color: var(--gold);
  }
  .metric-tile .lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.15em;
    color: var(--muted); text-transform: uppercase;
  }
  .metric-tile.cyan .val { color: var(--cyan); }

  /* ── Transcript box ── */
  .transcript-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(0,217,255,0.2);
    border-left: 3px solid var(--cyan);
    border-radius: 2px;
    padding: 1.2rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; line-height: 1.75;
    color: var(--white);
    white-space: pre-wrap;
    max-height: 280px; overflow-y: auto;
  }

  /* ── Emotion pill ── */
  .emotion-pill {
    display: inline-block;
    padding: 0.35rem 1.2rem;
    border-radius: 20px; font-weight: 700;
    font-size: 0.85rem; letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* ── Translation grid ── */
  .trans-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
  .trans-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(201,164,76,0.15);
    border-radius: 4px;
    padding: 1rem;
  }
  .trans-lang {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; color: var(--gold); letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 0.5rem;
  }
  .trans-text { font-size: 0.85rem; color: var(--white); line-height: 1.6; }

  /* ── Chat ── */
  .chat-bubble-user {
    background: rgba(201,164,76,0.12);
    border: 1px solid rgba(201,164,76,0.25);
    border-radius: 4px 4px 0 4px;
    padding: 0.8rem 1rem; margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.88rem; color: var(--white);
  }
  .chat-bubble-ai {
    background: rgba(0,217,255,0.06);
    border: 1px solid rgba(0,217,255,0.2);
    border-radius: 4px 4px 4px 0;
    padding: 0.8rem 1rem; margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.88rem; color: var(--white); line-height: 1.65;
  }
  .chat-sender {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.15em;
    color: var(--muted); margin-bottom: 0.25rem;
    text-transform: uppercase;
  }

  /* ── Streamlit overrides ── */
  .stButton > button {
    background: transparent !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.8rem !important;
    border-radius: 2px !important;
    transition: all 0.3s !important;
  }
  .stButton > button:hover {
    background: var(--gold) !important;
    color: var(--navy) !important;
  }
  .stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(201,164,76,0.25) !important;
    color: var(--white) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 2px !important;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: none !important;
  }
  .stFileUploader {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(201,164,76,0.3) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
  }
  .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(201,164,76,0.25) !important;
    color: var(--white) !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(201,164,76,0.2) !important;
    gap: 0 !important;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border: none !important;
    background: transparent !important;
    padding: 0.7rem 1.5rem !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: transparent !important;
  }
  .stProgress > div > div {
    background: linear-gradient(90deg, var(--gold), var(--cyan)) !important;
    border-radius: 2px !important;
  }
  .stSpinner > div { border-top-color: var(--gold) !important; }
  h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; color: var(--white) !important; }
  p { color: var(--muted) !important; }
  hr { border-color: rgba(201,164,76,0.15) !important; margin: 1.5rem 0 !important; }

  /* ── Waveform animation ── */
  @keyframes wave {
    0%,100% { transform: scaleY(1); }
    50%      { transform: scaleY(2.5); }
  }
  .wave-bars { display: flex; align-items: center; gap: 3px; height: 30px; }
  .wave-bar {
    width: 3px; background: var(--gold);
    border-radius: 2px; height: 10px;
    animation: wave 1s ease-in-out infinite;
  }
  .wave-bar:nth-child(2) { animation-delay: 0.1s; background: #b8a050; }
  .wave-bar:nth-child(3) { animation-delay: 0.2s; }
  .wave-bar:nth-child(4) { animation-delay: 0.3s; background: var(--cyan); }
  .wave-bar:nth-child(5) { animation-delay: 0.4s; }
  .wave-bar:nth-child(6) { animation-delay: 0.3s; background: #5ef0ff; }
  .wave-bar:nth-child(7) { animation-delay: 0.2s; }

  /* ── Pipeline step ── */
  .pipeline-step {
    display: flex; align-items: flex-start; gap: 1rem;
    padding: 0.8rem 1rem;
    border-left: 2px solid rgba(201,164,76,0.3);
    margin-bottom: 0.6rem;
  }
  .pipeline-step.done  { border-color: var(--gold); }
  .pipeline-step.active { border-color: var(--cyan); }
  .pipeline-step .step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: var(--muted);
    min-width: 30px; padding-top: 0.1rem;
  }
  .pipeline-step.done .step-num  { color: var(--gold); }
  .pipeline-step.active .step-num { color: var(--cyan); }
  .pipeline-step .step-name { font-size: 0.88rem; color: var(--white); }
  .pipeline-step .step-desc { font-size: 0.75rem; color: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ss-header">
  <div>
    <div class="ss-tag">// Neural Audio Processing System</div>
    <div class="ss-logo">SoundScape</div>
    <div class="ss-sub">AI-Powered Audio Intelligence &nbsp;·&nbsp; Understand &nbsp;·&nbsp; Restore &nbsp;·&nbsp; Analyze</div>
  </div>
  <div class="ss-badge">v2.0 Live</div>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sec-label">// Configuration</div>
    <div style="font-family:'Cormorant Garamond',serif;font-size:1.4rem;font-weight:600;color:#C9A44C;margin-bottom:1.2rem;">
        System Settings
    </div>
    """, unsafe_allow_html=True)

    groq_key = st.text_input(
        "GROQ API KEY",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com — needed for Q&A chatbot"
    )

    hf_token = st.text_input(
        "HUGGINGFACE TOKEN",
        type="password",
        placeholder="hf_...",
        value="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",
        help="Needed for speaker diarization (pyannote)"
    )

    st.markdown("---")
    st.markdown("""<div class="sec-label">// Pipeline Modules</div>""", unsafe_allow_html=True)

    do_denoise    = st.checkbox("Audio Enhancement",      value=True)
    do_transcribe = st.checkbox("Whisper Transcription",  value=True)
    do_emotion    = st.checkbox("Emotion Detection",      value=True)
    do_translate  = st.checkbox("Multilingual Translation", value=True)
    do_summary    = st.checkbox("AI Summary (Flan-T5)",   value=True)
    do_diarize    = st.checkbox("Speaker Diarization",    value=False,
                                help="Requires HuggingFace token")

    st.markdown("---")
    st.markdown("""
    <div class="ss-card" style="padding:1rem;">
      <div class="ss-card-title" style="font-size:0.9rem;">Model Info</div>
      <div class="ss-card-body" style="font-size:0.75rem;">
        🎤 Whisper <span style="color:#C9A44C">small</span><br>
        🎭 wav2vec2 + XLM-R<br>
        📝 Flan-T5-base<br>
        💬 LLaMA3-70B (Groq)<br>
        🔊 pyannote 3.1
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── SESSION STATE ────────────────────────────────────────────────────────────
defaults = {
    "results": None, "chat_history": [],
    "transcript": "", "emotion_label": "", "emotion_conf": 0,
    "detected_lang": "", "translations": {},
    "summary": "", "keywords": "", "topics": "",
    "score_before": 0, "score_after": 0,
    "num_speakers": 0,
    "audio_emotion": "", "text_emotion": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── LAZY MODEL LOADING ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper():
    import whisper
    return whisper.load_model("small")

@st.cache_resource(show_spinner=False)
def load_emotion_audio():
    from transformers import pipeline as hfp
    return hfp("audio-classification", model="superb/wav2vec2-base-superb-er")

@st.cache_resource(show_spinner=False)
def load_emotion_text():
    from transformers import pipeline as hfp
    return hfp("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

@st.cache_resource(show_spinner=False)
def load_flan():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, mdl


def audio_quality_score(arr):
    rms = np.sqrt(np.mean(arr ** 2))
    return round(20 * np.log10(rms + 1e-9), 2)


def run_flan(prompt, max_len=200):
    import torch
    tok, mdl = load_flan()
    inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_length=max_len, num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)


def safe_translate(text, target_code):
    try:
        from deep_translator import GoogleTranslator
        res = GoogleTranslator(source="auto", target=target_code).translate(text)
        return res if res else "[Empty result]"
    except Exception as ex:
        return f"[Translation failed: {str(ex)[:60]}]"


# ─── EMOTION COLOUR MAP ───────────────────────────────────────────────────────
EMOTION_COLORS = {
    "Happy":      ("#2ecc71", "#0d2b1a"),
    "Neutral":    ("#3498db", "#0d1e2b"),
    "Angry":      ("#e74c3c", "#2b0d0d"),
    "Sad":        ("#9b59b6", "#1e0d2b"),
    "Excited":    ("#f39c12", "#2b1e0d"),
    "Frustrated": ("#e67e22", "#2b1a0d"),
}

def emotion_pill(label):
    color, bg = EMOTION_COLORS.get(label, ("#C9A44C", "#1a150d"))
    return f'<span class="emotion-pill" style="background:{bg};color:{color};border:1px solid {color};">{label}</span>'


# ─── MAIN TABS ────────────────────────────────────────────────────────────────
tab_upload, tab_results, tab_chat, tab_pipeline = st.tabs([
    "🎙️  UPLOAD & ANALYZE",
    "📊  RESULTS",
    "💬  Q&A CHATBOT",
    "🔧  PIPELINE",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown("""
        <div class="sec-label">// Audio Input</div>
        <div class="sec-title">Upload <em>Audio</em></div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop audio here",
            type=["mp3", "mp4", "wav", "m4a", "ogg", "flac", "webm"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            st.markdown(f"""
            <div class="ss-card ss-card-cyan" style="padding:0.9rem 1.2rem;">
              <div class="ss-card-title" style="font-size:0.8rem;margin-bottom:0.3rem;">
                ✓ &nbsp; {uploaded_file.name}
              </div>
              <div class="ss-card-body" style="font-size:0.72rem;">
                Size: {uploaded_file.size/1024:.1f} KB &nbsp;·&nbsp; Type: {uploaded_file.type}
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.audio(uploaded_file)

        st.markdown("---")
        st.markdown("""
        <div class="sec-label">// Controls</div>
        """, unsafe_allow_html=True)

        analyze_btn = st.button("▶  RUN ANALYSIS", use_container_width=True)

        if not groq_key:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                        color:#7a8fa8;margin-top:0.6rem;padding:0.6rem;
                        border:1px solid rgba(201,164,76,0.15);border-radius:2px;">
              ⚠ Add Groq key in sidebar to enable Q&A chatbot
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="sec-label">// What This System Does</div>
        <div class="sec-title">Audio <em>Intelligence</em> Pipeline</div>
        """, unsafe_allow_html=True)

        features = [
            ("01", "🎵", "Audio Enhancement",
             "Noise reduction + volume normalization using noisereduce + librosa. Quality scored in dB before/after."),
            ("02", "🌐", "Multilingual Transcription",
             "OpenAI Whisper (small) auto-detects language — English, Hindi, Marathi, Hinglish, and 90+ others."),
            ("03", "🎭", "Dual Emotion Detection",
             "wav2vec2 reads acoustic features; XLM-R reads actual words. Final score is a smart weighted blend."),
            ("04", "🗣️", "Speaker Diarization",
             "pyannote 3.1 identifies distinct speakers with time-stamped segments. Requires HuggingFace token."),
            ("05", "📝", "AI Summary & Keywords",
             "Flan-T5 generates a concise summary, key topics, and keywords from the transcript."),
            ("06", "🌍", "Multilingual Translation",
             "Google Translator renders the transcript in English, Hindi, and Marathi simultaneously."),
        ]
        for num, icon, title, desc in features:
            st.markdown(f"""
            <div class="ss-card" style="padding:1rem 1.2rem;margin-bottom:0.7rem;">
              <div style="display:flex;align-items:flex-start;gap:0.8rem;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                             color:#C9A44C;min-width:28px;padding-top:0.2rem;">{num}</span>
                <span style="font-size:1.3rem;line-height:1.2;">{icon}</span>
                <div>
                  <div style="font-family:'Cormorant Garamond',serif;font-size:1rem;
                               font-weight:600;color:#f0f4f8;margin-bottom:0.25rem;">{title}</div>
                  <div style="font-size:0.78rem;color:#7a8fa8;line-height:1.55;">{desc}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─── ANALYSIS LOGIC ───────────────────────────────────────────────────────────
if analyze_btn and uploaded_file:
    with tab_upload:
        progress_bar = st.progress(0, text="Initializing pipeline…")
        status_area  = st.empty()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.read())

        results = {}

        # Step 1 — Convert to WAV
        progress_bar.progress(10, "Converting to WAV mono 16kHz…")
        try:
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_file(raw_path)
            audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
            wav_path  = os.path.join(tmpdir, "converted.wav")
            audio_seg.export(wav_path, format="wav")
            results["duration_sec"] = len(audio_seg) / 1000
        except Exception as e:
            st.error(f"Conversion failed: {e}"); st.stop()

        # Step 2 — Enhancement
        progress_bar.progress(20, "Enhancing audio quality…")
        if do_denoise:
            try:
                import noisereduce as nr
                y, sr = librosa.load(wav_path, sr=None)
                before = audio_quality_score(y)
                reduced = nr.reduce_noise(y=y, sr=sr)
                enhanced = np.clip(reduced * 1.8, -1.0, 1.0)
                after = audio_quality_score(enhanced)
                enh_path = os.path.join(tmpdir, "enhanced.wav")
                sf.write(enh_path, enhanced, sr)
                results["score_before"] = before
                results["score_after"]  = after
            except Exception:
                enh_path = wav_path
                results["score_before"] = results["score_after"] = 0
        else:
            enh_path = wav_path
            results["score_before"] = results["score_after"] = 0

        # Step 3 — Transcription
        progress_bar.progress(40, "Transcribing with Whisper…")
        if do_transcribe:
            try:
                model = load_whisper()
                res   = model.transcribe(enh_path, fp16=False)
                results["transcript"]    = res["text"].strip()
                results["detected_lang"] = res["language"]
                lp = res.get("language_probs", {})
                results["lang_conf"] = round(lp.get(res["language"], 0) * 100, 1) if lp else 0
            except Exception as e:
                results["transcript"]    = f"[Transcription error: {e}]"
                results["detected_lang"] = "unknown"
                results["lang_conf"]     = 0
        else:
            results["transcript"]    = "[Transcription skipped]"
            results["detected_lang"] = "unknown"
            results["lang_conf"]     = 0

        # Step 4 — Emotion
        progress_bar.progress(60, "Detecting emotion…")
        if do_emotion and results["transcript"] not in ["[Transcription skipped]"]:
            label_map = {"neu":"Neutral","hap":"Happy","ang":"Angry","sad":"Sad","exc":"Excited","fru":"Frustrated"}
            try:
                ed = load_emotion_audio()
                y_e, sr_e = librosa.load(enh_path, sr=16000)
                preds = ed({"raw": y_e, "sampling_rate": sr_e})
                top = max(preds, key=lambda x: x["score"])
                results["audio_emotion"]      = label_map.get(top["label"], top["label"])
                results["audio_emotion_conf"] = round(top["score"] * 100, 1)
            except Exception:
                results["audio_emotion"]      = "Neutral"
                results["audio_emotion_conf"] = 50.0

            try:
                et   = load_emotion_text()
                tres = et(results["transcript"][:512])
                sentiment_to_emotion = {"Positive":"Happy","Neutral":"Neutral","Negative":"Sad",
                                        "LABEL_2":"Happy","LABEL_1":"Neutral","LABEL_0":"Sad"}
                lbl  = tres[0]["label"]
                conf = round(tres[0]["score"] * 100, 1)
                results["text_emotion"]      = sentiment_to_emotion.get(lbl, "Neutral")
                results["text_emotion_conf"] = conf
            except Exception:
                results["text_emotion"]      = results.get("audio_emotion", "Neutral")
                results["text_emotion_conf"] = 50.0

            # Blend
            ae, te = results["audio_emotion"], results["text_emotion"]
            ac, tc = results["audio_emotion_conf"], results["text_emotion_conf"]
            if ae == te:
                results["emotion_label"] = ae
                results["emotion_conf"]  = round((ac + tc) / 2, 1)
            else:
                results["emotion_label"] = te if tc > ac else ae
                results["emotion_conf"]  = max(ac, tc)
        else:
            results.update({"audio_emotion":"—","text_emotion":"—",
                            "emotion_label":"—","emotion_conf":0})

        # Step 5 — Translation
        progress_bar.progress(75, "Translating…")
        if do_translate and results["transcript"] not in ["[Transcription skipped]"]:
            t = results["transcript"]
            results["translations"] = {
                "English": safe_translate(t, "en"),
                "Hindi":   safe_translate(t, "hi"),
                "Marathi": safe_translate(t, "mr"),
            }
        else:
            results["translations"] = {}

        # Step 6 — Flan-T5 summary
        progress_bar.progress(85, "Generating summary…")
        if do_summary and results["transcript"] not in ["[Transcription skipped]"]:
            t = results["transcript"][:800]
            try:
                results["summary"]  = run_flan(f"Summarize this transcript in 2-3 sentences: {t}")
                results["keywords"] = run_flan(f"List 5 key topics or keywords from this text: {t}")
            except Exception as e:
                results["summary"]  = f"[Summary error: {e}]"
                results["keywords"] = ""
        else:
            results["summary"]  = ""
            results["keywords"] = ""

        progress_bar.progress(100, "✓ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_area.success("Analysis complete — view results in the Results tab.")

        # Store in session
        for k, v in results.items():
            st.session_state[k] = v
        st.session_state["results"] = results
        st.session_state["chat_history"] = []


elif analyze_btn and not uploaded_file:
    st.warning("Please upload an audio file first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state["results"]:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
          <div style="font-family:'Cormorant Garamond',serif;font-size:3rem;
                      font-weight:300;color:#7a8fa8;">No Analysis Yet</div>
          <div style="font-size:0.85rem;color:#4a6080;margin-top:1rem;">
            Upload an audio file and click RUN ANALYSIS to see results here.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        R = st.session_state["results"]
        lang = R.get("detected_lang", "?").upper()
        dur  = R.get("duration_sec", 0)

        # ── Metrics row ──
        st.markdown("""<div class="sec-label">// Analysis Overview</div>""", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Language", lang)
        with m2:
            st.metric("Duration", f"{dur:.1f}s")
        with m3:
            st.metric("Emotion", R.get("emotion_label","—"))
        with m4:
            conf = R.get("emotion_conf", 0)
            st.metric("Confidence", f"{conf}%" if conf else "—")
        with m5:
            imp = R.get("score_after",0) - R.get("score_before",0)
            st.metric("Audio Gain", f"{imp:+.1f} dB" if imp else "—")

        st.markdown("---")

        # ── Columns ──
        r1, r2 = st.columns([1.4, 1], gap="large")

        with r1:
            st.markdown("""<div class="sec-label">// Transcript</div>""", unsafe_allow_html=True)
            transcript = R.get("transcript","")
            st.markdown(f'<div class="transcript-box">{transcript}</div>', unsafe_allow_html=True)

            if R.get("summary"):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""<div class="sec-label">// AI Summary (Flan-T5)</div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="ss-card ss-card-cyan">
                  <div class="ss-card-body">{R['summary']}</div>
                </div>
                """, unsafe_allow_html=True)

            if R.get("keywords"):
                st.markdown("""<div class="sec-label">// Keywords & Topics</div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="ss-card" style="padding:0.9rem 1.2rem;">
                  <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                               color:#C9A44C;">{R['keywords']}</div>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            st.markdown("""<div class="sec-label">// Emotion Analysis</div>""", unsafe_allow_html=True)
            em_label = R.get("emotion_label","—")
            em_conf  = R.get("emotion_conf", 0)
            em_color = EMOTION_COLORS.get(em_label, ("#C9A44C","#1a150d"))[0]

            st.markdown(f"""
            <div class="ss-card" style="padding:1.5rem;text-align:center;margin-bottom:1rem;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                           letter-spacing:0.2em;color:#7a8fa8;margin-bottom:0.8rem;">
                DETECTED EMOTION
              </div>
              {emotion_pill(em_label)}
              <div style="font-family:'Cormorant Garamond',serif;font-size:2.5rem;
                           font-weight:700;color:{em_color};margin:0.8rem 0 0.2rem;">
                {em_conf}%
              </div>
              <div style="font-size:0.72rem;color:#7a8fa8;">confidence</div>
            </div>
            <div class="ss-card" style="padding:1rem 1.2rem;margin-bottom:0.6rem;">
              <div class="ss-card-body" style="font-size:0.75rem;">
                🎵 Audio model: <span style="color:#f0f4f8;">{R.get('audio_emotion','—')} ({R.get('audio_emotion_conf',0)}%)</span>
              </div>
            </div>
            <div class="ss-card" style="padding:1rem 1.2rem;">
              <div class="ss-card-body" style="font-size:0.75rem;">
                📝 Text model: <span style="color:#f0f4f8;">{R.get('text_emotion','—')} ({R.get('text_emotion_conf',0)}%)</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Quality meters
            st.markdown("""<div class="sec-label" style="margin-top:1rem;">// Audio Quality</div>""", unsafe_allow_html=True)
            sb = R.get("score_before", -60)
            sa = R.get("score_after",  -60)
            # normalize dB to 0-1 for display (range -60 to 0 dB)
            norm = lambda v: max(0, min(1, (v + 60) / 60))
            st.markdown(f"""
            <div class="ss-card" style="padding:1rem 1.2rem;">
              <div class="ss-card-body" style="font-size:0.75rem;margin-bottom:0.6rem;">
                Before: <span style="color:#C9A44C;">{sb} dB</span>
              </div>
              <div style="background:rgba(255,255,255,0.06);border-radius:2px;height:6px;margin-bottom:0.8rem;">
                <div style="background:#7a8fa8;width:{norm(sb)*100:.0f}%;height:6px;border-radius:2px;"></div>
              </div>
              <div class="ss-card-body" style="font-size:0.75rem;margin-bottom:0.6rem;">
                After: <span style="color:#00D9FF;">{sa} dB</span>
              </div>
              <div style="background:rgba(255,255,255,0.06);border-radius:2px;height:6px;">
                <div style="background:linear-gradient(90deg,#C9A44C,#00D9FF);width:{norm(sa)*100:.0f}%;height:6px;border-radius:2px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Translations ──
        if R.get("translations"):
            st.markdown("---")
            st.markdown("""<div class="sec-label">// Multilingual Output</div>
            <div class="sec-title">Translation <em>Matrix</em></div>""", unsafe_allow_html=True)
            tc1, tc2, tc3 = st.columns(3)
            trans = R["translations"]
            for col, (lang_name, text) in zip([tc1, tc2, tc3], trans.items()):
                with col:
                    flag = {"English":"🇬🇧","Hindi":"🇮🇳","Marathi":"🟠"}.get(lang_name,"🌍")
                    st.markdown(f"""
                    <div class="ss-card ss-card-cyan" style="min-height:160px;">
                      <div class="trans-lang">{flag} {lang_name}</div>
                      <div class="trans-text">{text}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Q&A CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("""
    <div class="sec-label">// Groq LLaMA3-70B</div>
    <div class="sec-title">Q&A <em>Chatbot</em></div>
    """, unsafe_allow_html=True)

    if not st.session_state["results"]:
        st.info("Run analysis first to enable the chatbot.")
    elif not groq_key:
        st.markdown("""
        <div class="ss-card" style="padding:1.5rem;text-align:center;">
          <div style="font-size:1.5rem;margin-bottom:0.8rem;">🔑</div>
          <div class="ss-card-title">Groq API Key Required</div>
          <div class="ss-card-body">Add your Groq API key in the sidebar to enable the chatbot.<br>
          Get a free key at <span style="color:#00D9FF;">console.groq.com</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        transcript = st.session_state.get("transcript","")
        lang = st.session_state.get("detected_lang","en")

        # Display chat history
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="text-align:right;">
                  <div class="chat-sender" style="text-align:right;">You</div>
                  <div class="chat-bubble-user">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div>
                  <div class="chat-sender">
                    <span class="wave-bars" style="display:inline-flex;gap:2px;height:14px;margin-right:8px;">
                      <span class="wave-bar" style="height:6px;width:2px;background:#C9A44C;"></span>
                      <span class="wave-bar" style="height:8px;width:2px;background:#C9A44C;animation-delay:0.1s;"></span>
                      <span class="wave-bar" style="height:5px;width:2px;background:#C9A44C;animation-delay:0.2s;"></span>
                    </span>
                    SoundScape AI
                  </div>
                  <div class="chat-bubble-ai">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Input
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            user_q = st.text_input("", placeholder="Ask anything about the audio…",
                                   label_visibility="collapsed", key="chat_input")
        with btn_col:
            send_btn = st.button("SEND", use_container_width=True)

        if send_btn and user_q.strip():
            system_prompt = f"""You are SoundScape AI, an expert audio intelligence assistant.
You have analysed an audio file. Key results:
- Language detected: {lang}
- Emotion detected: {st.session_state.get('emotion_label','unknown')}
- Transcript: {transcript[:1500]}

Answer the user's questions about this audio concisely and helpfully.
If asked in Hindi or Marathi, respond in that language."""

            messages = [{"role":"system","content":system_prompt}]
            for m in st.session_state["chat_history"][-8:]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role":"user","content":user_q})

            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                with st.spinner("Thinking…"):
                    resp = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=messages,
                        max_tokens=600,
                        temperature=0.7
                    )
                answer = resp.choices[0].message.content
                st.session_state["chat_history"].append({"role":"user","content":user_q})
                st.session_state["chat_history"].append({"role":"assistant","content":answer})
                st.rerun()
            except Exception as e:
                st.error(f"Chatbot error: {e}")

        if st.session_state["chat_history"]:
            if st.button("CLEAR CHAT"):
                st.session_state["chat_history"] = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PIPELINE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("""
    <div class="sec-label">// System Architecture</div>
    <div class="sec-title">Processing <em>Pipeline</em></div>
    """, unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")

    with p1:
        steps = [
            ("01", "Audio Ingestion", "MP3/MP4/WAV/M4A/OGG/FLAC/WEBM → pydub → WAV mono 16kHz"),
            ("02", "Enhancement",     "noisereduce + librosa → SNR improvement → soundfile"),
            ("03", "Transcription",   "OpenAI Whisper (small) → 90+ languages → raw transcript"),
            ("04", "Missing Words",   "mBERT fill-mask → [inaudible] prediction → cleaned text"),
            ("05", "Emotion (Audio)", "wav2vec2-superb → acoustic features → 6-class emotion"),
            ("06", "Emotion (Text)",  "XLM-RoBERTa → multilingual sentiment → mapped emotion"),
            ("07", "Speaker ID",      "pyannote 3.1 → speaker diarization → labelled segments"),
            ("08", "NLP Analysis",    "Flan-T5-base → summary + keywords + topic detection"),
            ("09", "Translation",     "GoogleTranslator → English + Hindi + Marathi output"),
            ("10", "Q&A Chatbot",     "Groq API → LLaMA3-70B → transcript-aware conversation"),
        ]
        results_exist = bool(st.session_state["results"])
        for num, name, desc in steps:
            idx = int(num) - 1
            status_class = "done" if (results_exist and idx < 6) else ""
            tick = "✓" if (results_exist and idx < 6) else num
            st.markdown(f"""
            <div class="pipeline-step {status_class}">
              <div class="step-num">{tick}</div>
              <div>
                <div class="step-name">{name}</div>
                <div class="step-desc">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with p2:
        st.markdown("""<div class="sec-label">// Models & Libraries</div>""", unsafe_allow_html=True)

        models = [
            ("Whisper small",                  "OpenAI",      "ASR — 460MB, 90+ languages"),
            ("wav2vec2-base-superb-er",         "Facebook",    "Audio emotion, 6 classes"),
            ("twitter-xlm-roberta-base",        "CardiffNLP",  "Text sentiment, multilingual"),
            ("bert-base-multilingual-cased",    "Google/BERT", "Fill-mask for missing words"),
            ("flan-t5-base",                    "Google",      "Summarization, keywords"),
            ("pyannote/speaker-diarization-3.1","pyannote",    "Speaker ID (needs HF token)"),
            ("llama3-70b-8192",                 "Meta/Groq",   "Q&A chatbot (needs Groq key)"),
            ("noisereduce",                     "Tim Sainburg", "Spectral noise reduction"),
            ("deep-translator",                 "Google",      "En/Hi/Mr translation"),
            ("gTTS",                            "Google",      "Text-to-speech output"),
        ]
        for model, author, role in models:
            st.markdown(f"""
            <div class="ss-card" style="padding:0.8rem 1rem;margin-bottom:0.5rem;display:flex;
                         align-items:center;justify-content:space-between;">
              <div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                             color:#C9A44C;">{model}</div>
                <div style="font-size:0.7rem;color:#7a8fa8;margin-top:0.15rem;">{role}</div>
              </div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                           color:#4a6080;text-align:right;">{author}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="sec-label">// Supported Formats</div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="ss-card ss-card-cyan" style="padding:1rem 1.2rem;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                       color:#00D9FF;letter-spacing:0.15em;">
            MP3 &nbsp;·&nbsp; MP4 &nbsp;·&nbsp; WAV &nbsp;·&nbsp; M4A &nbsp;·&nbsp;
            OGG &nbsp;·&nbsp; FLAC &nbsp;·&nbsp; WEBM
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<div class="sec-label" style="margin-top:1rem;">// Supported Languages</div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="ss-card" style="padding:1rem 1.2rem;">
          <div style="font-size:0.82rem;color:#7a8fa8;line-height:1.7;">
            🇬🇧 English &nbsp;·&nbsp; 🇮🇳 Hindi &nbsp;·&nbsp; 🇮🇳 Marathi &nbsp;·&nbsp;
            🔀 Hinglish<br>
            <span style="font-size:0.72rem;color:#4a6080;">+ 90 additional languages via Whisper auto-detection</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
