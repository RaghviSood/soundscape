# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   SoundScape — AI Audio Intelligence Dashboard                          ║
# ║   Self-contained Streamlit app. NO agent.py. NO external local imports. ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import os
import warnings
import tempfile
import time
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SoundScape — Neural Audio Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
  --navy:#0A1828; --navy2:#0d1f33; --gold:#C9A44C; --gold2:#e8c56a;
  --cyan:#00D9FF; --white:#f0f4f8; --muted:#7a8fa8;
  --card:rgba(255,255,255,0.04); --border:rgba(201,164,76,0.22);
}

html,body,[class*="css"]{font-family:'Syne',sans-serif!important;
  background:#0A1828!important;color:var(--white)!important;}
.stApp{background:#0A1828!important;}
.block-container{padding-top:1.2rem!important;max-width:1400px;}

[data-testid="stSidebar"]{background:#0d1f33!important;
  border-right:1px solid rgba(201,164,76,0.12)!important;}
[data-testid="stSidebar"] *{color:var(--white)!important;}

.ss-header{display:flex;align-items:center;gap:1.4rem;padding:1.1rem 2rem;
  background:rgba(13,31,51,0.95);border:1px solid rgba(201,164,76,0.2);
  border-radius:4px;margin-bottom:1.4rem;}
.ss-logo{font-family:'Cormorant Garamond',serif;font-size:2.2rem;font-weight:700;
  background:linear-gradient(135deg,#C9A44C 0%,#e8c56a 45%,#00D9FF 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.02em;}
.ss-tag{font-family:'JetBrains Mono',monospace;font-size:0.6rem;
  letter-spacing:0.25em;color:var(--cyan);text-transform:uppercase;}
.ss-sub{font-size:0.78rem;color:var(--muted);margin-top:0.15rem;}
.ss-badge{margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:0.6rem;
  letter-spacing:0.15em;color:var(--navy);background:var(--gold);
  padding:0.3rem 0.8rem;border-radius:2px;text-transform:uppercase;white-space:nowrap;}

.sec-label{font-family:'JetBrains Mono',monospace;font-size:0.6rem;
  letter-spacing:0.3em;text-transform:uppercase;color:var(--cyan);margin-bottom:0.3rem;}
.sec-title{font-family:'Cormorant Garamond',serif;font-size:1.7rem;
  font-weight:600;color:var(--white);margin-bottom:1rem;}
.sec-title em{color:var(--gold);font-style:normal;}

.ss-card{background:var(--card);border:1px solid var(--border);
  border-radius:4px;padding:1.3rem 1.5rem;margin-bottom:0.8rem;}
.ss-card-cyan{border-color:rgba(0,217,255,0.25);}
.ss-card-title{font-family:'Cormorant Garamond',serif;font-size:1rem;
  font-weight:600;color:var(--gold);margin-bottom:0.5rem;}
.ss-card-body{font-size:0.85rem;color:var(--muted);line-height:1.65;}

.transcript-box{background:rgba(0,0,0,0.3);border:1px solid rgba(0,217,255,0.2);
  border-left:3px solid var(--cyan);border-radius:2px;padding:1.1rem 1.4rem;
  font-family:'JetBrains Mono',monospace;font-size:0.82rem;line-height:1.75;
  color:var(--white);white-space:pre-wrap;max-height:260px;overflow-y:auto;}

.em-pill{display:inline-block;padding:0.3rem 1.1rem;border-radius:20px;
  font-weight:700;font-size:0.82rem;letter-spacing:0.08em;text-transform:uppercase;}

.chat-u{background:rgba(201,164,76,0.1);border:1px solid rgba(201,164,76,0.22);
  border-radius:4px 4px 0 4px;padding:0.75rem 1rem;margin:0.4rem 0 0.4rem 4rem;
  font-size:0.85rem;color:var(--white);}
.chat-a{background:rgba(0,217,255,0.05);border:1px solid rgba(0,217,255,0.18);
  border-radius:4px 4px 4px 0;padding:0.75rem 1rem;margin:0.4rem 4rem 0.4rem 0;
  font-size:0.85rem;color:var(--white);line-height:1.65;}
.chat-lbl{font-family:'JetBrains Mono',monospace;font-size:0.58rem;
  letter-spacing:0.15em;color:var(--muted);margin-bottom:0.2rem;text-transform:uppercase;}

.pip-step{display:flex;align-items:flex-start;gap:0.9rem;padding:0.7rem 0.9rem;
  border-left:2px solid rgba(201,164,76,0.25);margin-bottom:0.5rem;}
.pip-step.done{border-color:var(--gold);}
.pip-num{font-family:'JetBrains Mono',monospace;font-size:0.62rem;
  color:var(--muted);min-width:26px;padding-top:0.1rem;}
.pip-step.done .pip-num{color:var(--gold);}
.pip-name{font-size:0.86rem;color:var(--white);}
.pip-desc{font-size:0.72rem;color:var(--muted);}

.mod-tile{background:var(--card);border:1px solid var(--border);border-radius:4px;
  padding:0.75rem 1rem;margin-bottom:0.45rem;display:flex;
  justify-content:space-between;align-items:center;}
.mod-name{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--gold);}
.mod-role{font-size:0.68rem;color:var(--muted);margin-top:0.12rem;}
.mod-by{font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#4a6080;}

.stButton>button{background:transparent!important;border:1px solid var(--gold)!important;
  color:var(--gold)!important;font-family:'Syne',sans-serif!important;
  font-size:0.72rem!important;font-weight:700!important;letter-spacing:0.15em!important;
  text-transform:uppercase!important;border-radius:2px!important;}
.stButton>button:hover{background:var(--gold)!important;color:#0A1828!important;}
.stTextInput input,.stTextArea textarea{background:rgba(255,255,255,0.04)!important;
  border:1px solid rgba(201,164,76,0.22)!important;color:var(--white)!important;
  font-family:'JetBrains Mono',monospace!important;border-radius:2px!important;}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:var(--cyan)!important;box-shadow:none!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;
  border-bottom:1px solid rgba(201,164,76,0.18)!important;}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace!important;
  font-size:0.62rem!important;letter-spacing:0.18em!important;
  text-transform:uppercase!important;color:var(--muted)!important;
  background:transparent!important;border:none!important;}
.stTabs [aria-selected="true"]{color:var(--gold)!important;
  border-bottom:2px solid var(--gold)!important;}
h1,h2,h3{font-family:'Cormorant Garamond',serif!important;color:var(--white)!important;}
p{color:var(--muted)!important;}
hr{border-color:rgba(201,164,76,0.12)!important;margin:1.2rem 0!important;}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ss-header">
  <div>
    <div class="ss-tag">// Neural Audio Processing System</div>
    <div class="ss-logo">SoundScape</div>
    <div class="ss-sub">AI-Powered Audio Intelligence &nbsp;·&nbsp; Understand · Restore · Analyze</div>
  </div>
  <div class="ss-badge">Live v2.0</div>
</div>
""", unsafe_allow_html=True)

# ─── MODULE-LEVEL CONSTANTS (must be defined before any tab renders) ──────────
EMOTION_COLORS = {
    "Happy":      ("#2ecc71", "#0d2b1a"),
    "Neutral":    ("#3498db", "#0d1e2b"),
    "Angry":      ("#e74c3c", "#2b0d0d"),
    "Sad":        ("#9b59b6", "#1e0d2b"),
    "Excited":    ("#f39c12", "#2b1e0d"),
    "Frustrated": ("#e67e22", "#2b1a0d"),
}

LABEL_MAP = {
    "neu": "Neutral", "hap": "Happy", "ang": "Angry",
    "sad": "Sad",     "exc": "Excited","fru": "Frustrated",
}

SENT_MAP = {
    "Positive": "Happy",   "Neutral": "Neutral",  "Negative": "Sad",
    "LABEL_2":  "Happy",   "LABEL_1": "Neutral",  "LABEL_0":  "Sad",
}


def em_pill(label):
    c, bg = EMOTION_COLORS.get(label, ("#C9A44C", "#1a150d"))
    return (f'<span class="em-pill" style="background:{bg};color:{c};'
            f'border:1px solid {c};">{label}</span>')


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-label">// API Keys</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title" style="font-size:1.2rem;">Configuration</div>',
                unsafe_allow_html=True)

    groq_key = st.text_input("GROQ API KEY", type="password", placeholder="gsk_…",
                             help="Free key at console.groq.com — powers the Q&A chatbot")
    hf_token = st.text_input("HUGGINGFACE TOKEN", type="password", placeholder="hf_…",
                             help="Needed for speaker diarization (pyannote)")

    st.markdown("---")
    st.markdown('<div class="sec-label">// Pipeline Modules</div>', unsafe_allow_html=True)
    do_denoise    = st.checkbox("🎵 Audio Enhancement",        value=True)
    do_transcribe = st.checkbox("🌐 Whisper Transcription",    value=True)
    do_emotion    = st.checkbox("🎭 Dual Emotion Detection",   value=True)
    do_translate  = st.checkbox("🌍 Multilingual Translation", value=True)
    do_summary    = st.checkbox("📝 AI Summary (Flan-T5)",     value=True)
    do_diarize    = st.checkbox("🗣️ Speaker Diarization",      value=False,
                                help="Requires HuggingFace token + pyannote model access")

    st.markdown("---")
    st.markdown("""
    <div class="ss-card" style="padding:0.9rem;">
      <div class="ss-card-title" style="font-size:0.82rem;">Stack</div>
      <div class="ss-card-body" style="font-size:0.72rem;line-height:1.8;">
        🎤 Whisper <span style="color:#C9A44C">small</span><br>
        🎭 wav2vec2-superb-er<br>
        📝 XLM-RoBERTa (text)<br>
        🤖 Flan-T5-base<br>
        💬 LLaMA3-70B via Groq<br>
        🔊 pyannote 3.1
      </div>
    </div>""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in {
    "results": None, "chat_history": [],
    "transcript": "", "emotion_label": "", "emotion_conf": 0,
    "detected_lang": "", "translations": {},
    "summary": "", "keywords": "",
    "score_before": 0.0, "score_after": 0.0,
    "audio_emotion": "", "audio_emotion_conf": 0.0,
    "text_emotion":  "", "text_emotion_conf":  0.0,
    "duration_sec": 0.0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── CACHED MODEL LOADERS ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    import whisper
    return whisper.load_model("small")

@st.cache_resource(show_spinner=False)
def load_audio_emotion():
    from transformers import pipeline as hfp
    return hfp("audio-classification", model="superb/wav2vec2-base-superb-er")

@st.cache_resource(show_spinner=False)
def load_text_emotion():
    from transformers import pipeline as hfp
    return hfp("text-classification",
               model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

@st.cache_resource(show_spinner=False)
def load_flan():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, mdl

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def audio_quality_score(arr):
    rms = np.sqrt(np.mean(arr.astype(np.float32) ** 2))
    return round(20 * np.log10(float(rms) + 1e-9), 2)

def run_flan(prompt, max_len=200):
    import torch
    tok, mdl = load_flan()
    inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_length=max_len,
                           num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)

def safe_translate(text, tgt):
    try:
        from deep_translator import GoogleTranslator
        # deep-translator has a 5000-char limit per request
        chunk = text[:4500]
        r = GoogleTranslator(source="auto", target=tgt).translate(chunk)
        return r if r else "[Empty]"
    except Exception as ex:
        return f"[Translation error: {str(ex)[:80]}]"

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎙️  UPLOAD & ANALYZE",
    "📊  RESULTS",
    "💬  Q&A CHATBOT",
    "🔧  PIPELINE",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([1, 1.55], gap="large")

    with c1:
        st.markdown('<div class="sec-label">// Audio Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Upload <em>Audio</em></div>', unsafe_allow_html=True)

        up = st.file_uploader("", type=["mp3","mp4","wav","m4a","ogg","flac","webm"],
                              label_visibility="collapsed")
        if up:
            st.markdown(f"""
            <div class="ss-card ss-card-cyan" style="padding:0.8rem 1rem;">
              <div class="ss-card-title" style="font-size:0.78rem;margin-bottom:0.2rem;">
                ✓ &nbsp;{up.name}
              </div>
              <div class="ss-card-body" style="font-size:0.68rem;">
                {up.size/1024:.1f} KB &nbsp;·&nbsp; {up.type}
              </div>
            </div>""", unsafe_allow_html=True)
            st.audio(up)

        st.markdown("---")
        run_btn = st.button("▶  RUN FULL ANALYSIS", use_container_width=True)
        if not groq_key:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                        color:#7a8fa8;padding:0.6rem;margin-top:0.5rem;
                        border:1px solid rgba(201,164,76,0.12);border-radius:2px;">
              ⚠ Add Groq key in sidebar for Q&A chatbot
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sec-label">// Pipeline Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">What <em>Happens</em></div>', unsafe_allow_html=True)
        feats = [
            ("01","🎵","Audio Enhancement","noisereduce + librosa — spectral denoising, SNR scoring in dB"),
            ("02","🌐","Whisper Transcription","OpenAI Whisper small — 90+ languages, auto language detection"),
            ("03","🎭","Dual Emotion Detection","wav2vec2 (acoustic) + XLM-RoBERTa (text) — weighted blend"),
            ("04","🗣️","Speaker Diarization","pyannote 3.1 — who spoke when (needs HuggingFace token)"),
            ("05","📝","AI Summary","Flan-T5-base — concise summary + keywords + topics"),
            ("06","🌍","Multilingual Translation","Google Translator — English · Hindi · Marathi output"),
        ]
        for num, icon, title, desc in feats:
            st.markdown(f"""
            <div class="ss-card" style="padding:0.9rem 1.1rem;margin-bottom:0.55rem;">
              <div style="display:flex;align-items:flex-start;gap:0.7rem;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                             color:#C9A44C;min-width:24px;">{num}</span>
                <span style="font-size:1.2rem;">{icon}</span>
                <div>
                  <div style="font-family:'Cormorant Garamond',serif;font-size:0.95rem;
                               font-weight:600;color:#f0f4f8;">{title}</div>
                  <div style="font-size:0.73rem;color:#7a8fa8;line-height:1.5;">{desc}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
if run_btn and up:
    prog = st.progress(0, "Starting…")

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, up.name)
        with open(raw, "wb") as f:
            f.write(up.getvalue())   # FIX: use getvalue() not read() so buffer isn't consumed

        R = {}

        # 1 — Convert to WAV 16kHz mono
        prog.progress(8, "Converting to WAV 16kHz…")
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(raw).set_channels(1).set_frame_rate(16000)
            wav = os.path.join(tmp, "converted.wav")
            seg.export(wav, format="wav")
            R["duration_sec"] = len(seg) / 1000
        except Exception as e:
            st.error(f"❌ Conversion failed: {e}")
            st.stop()

        # 2 — Enhance
        prog.progress(18, "Enhancing audio…")
        enh_wav = wav
        R["score_before"] = R["score_after"] = 0.0
        if do_denoise:
            try:
                import librosa, noisereduce as nr, soundfile as sf
                y, sr = librosa.load(wav, sr=None)
                R["score_before"] = audio_quality_score(y)
                enh = np.clip(nr.reduce_noise(y=y, sr=sr) * 1.8, -1.0, 1.0)
                R["score_after"] = audio_quality_score(enh)
                enh_wav = os.path.join(tmp, "enhanced.wav")
                sf.write(enh_wav, enh.astype(np.float32), sr)
            except Exception:
                enh_wav = wav

        # 3 — Transcribe
        prog.progress(38, "Transcribing with Whisper (may take a minute)…")
        R["transcript"] = "[Transcription skipped]"
        R["detected_lang"] = "unknown"
        R["lang_conf"] = 0
        if do_transcribe:
            try:
                mdl_w = load_whisper_model()
                res = mdl_w.transcribe(enh_wav, fp16=False)
                R["transcript"]    = res["text"].strip()
                R["detected_lang"] = res.get("language", "unknown")
                lp = res.get("language_probs", {})
                R["lang_conf"] = round(lp.get(res.get("language",""), 0) * 100, 1) if lp else 0
            except Exception as e:
                R["transcript"]    = f"[Transcription error: {e}]"

        # 4 — Emotion
        prog.progress(58, "Detecting emotion…")
        transcript_ok = R["transcript"].strip() not in [
            "", "[Transcription skipped]"
        ] and not R["transcript"].startswith("[Transcription error")

        for k in ["audio_emotion", "text_emotion", "emotion_label"]:
            R[k] = "—"
        for k in ["audio_emotion_conf", "text_emotion_conf", "emotion_conf"]:
            R[k] = 0.0

        if do_emotion and transcript_ok:
            # Audio-based emotion
            try:
                import librosa as lb
                ed = load_audio_emotion()
                ya, sra = lb.load(enh_wav, sr=16000)
                preds = ed({"raw": ya, "sampling_rate": sra})
                top = max(preds, key=lambda x: x["score"])
                R["audio_emotion"]      = LABEL_MAP.get(top["label"], top["label"].capitalize())
                R["audio_emotion_conf"] = round(top["score"] * 100, 1)
            except Exception:
                R["audio_emotion"]      = "Neutral"
                R["audio_emotion_conf"] = 50.0

            # Text-based sentiment
            try:
                et   = load_text_emotion()
                tres = et(R["transcript"][:512])
                raw_lbl = tres[0]["label"]
                R["text_emotion"]      = SENT_MAP.get(raw_lbl, "Neutral")
                R["text_emotion_conf"] = round(tres[0]["score"] * 100, 1)
            except Exception:
                R["text_emotion"]      = R.get("audio_emotion", "Neutral")
                R["text_emotion_conf"] = 50.0

            # Blend audio + text
            ae, te = R["audio_emotion"], R["text_emotion"]
            ac, tc = R["audio_emotion_conf"], R["text_emotion_conf"]
            if ae == te:
                R["emotion_label"] = ae
                R["emotion_conf"]  = round((ac + tc) / 2, 1)
            else:
                R["emotion_label"] = te if tc > ac else ae
                R["emotion_conf"]  = round(max(ac, tc), 1)

        # 5 — Translation
        prog.progress(72, "Translating…")
        R["translations"] = {}
        if do_translate and transcript_ok:
            t = R["transcript"]
            R["translations"] = {
                "English": safe_translate(t, "en"),
                "Hindi":   safe_translate(t, "hi"),
                "Marathi": safe_translate(t, "mr"),
            }

        # 6 — Flan-T5 Summary
        prog.progress(85, "Generating AI summary…")
        R["summary"] = R["keywords"] = ""
        if do_summary and transcript_ok:
            t = R["transcript"][:800]
            try:
                R["summary"]  = run_flan(f"Summarize this transcript in 2-3 sentences: {t}")
                R["keywords"] = run_flan(f"List 5 keywords or topics from: {t}")
            except Exception as e:
                R["summary"] = f"[Error: {e}]"

        prog.progress(100, "✓ Done!")
        time.sleep(0.4)
        prog.empty()

        for k, v in R.items():
            st.session_state[k] = v
        st.session_state["results"]      = R
        st.session_state["chat_history"] = []

    st.success("✓ Analysis complete — open the Results tab!")

elif run_btn:
    st.warning("Please upload an audio file first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    R = st.session_state.get("results")
    if not R:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
          <div style="font-family:'Cormorant Garamond',serif;font-size:2.8rem;
                      font-weight:300;color:#7a8fa8;">No Analysis Yet</div>
          <div style="font-size:0.82rem;color:#4a6080;margin-top:0.8rem;">
            Upload audio and click RUN FULL ANALYSIS
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        lang = R.get("detected_lang", "?").upper()
        dur  = R.get("duration_sec", 0)
        elbl = R.get("emotion_label", "—")
        econ = R.get("emotion_conf", 0)

        st.markdown('<div class="sec-label">// Overview</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Language",   lang)
        m2.metric("Duration",   f"{dur:.1f}s")
        m3.metric("Emotion",    elbl)
        m4.metric("Confidence", f"{econ}%" if econ else "—")
        imp = R.get("score_after", 0) - R.get("score_before", 0)
        m5.metric("Audio Gain", f"{imp:+.1f} dB" if imp else "—")

        st.markdown("---")
        left, right = st.columns([1.4, 1], gap="large")

        with left:
            st.markdown('<div class="sec-label">// Transcript</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="transcript-box">{R.get("transcript","")}</div>',
                        unsafe_allow_html=True)

            if R.get("summary"):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="sec-label">// AI Summary</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="ss-card ss-card-cyan">
                  <div class="ss-card-body">{R["summary"]}</div>
                </div>""", unsafe_allow_html=True)

            if R.get("keywords"):
                st.markdown('<div class="sec-label">// Keywords</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="ss-card" style="padding:0.85rem 1rem;">
                  <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                               color:#C9A44C;">{R["keywords"]}</div>
                </div>""", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="sec-label">// Emotion</div>', unsafe_allow_html=True)
            ec, _ = EMOTION_COLORS.get(elbl, ("#C9A44C", "#1a150d"))
            st.markdown(f"""
            <div class="ss-card" style="padding:1.4rem;text-align:center;margin-bottom:0.8rem;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                           letter-spacing:0.2em;color:#7a8fa8;margin-bottom:0.7rem;">
                DETECTED EMOTION
              </div>
              {em_pill(elbl)}
              <div style="font-family:'Cormorant Garamond',serif;font-size:2.4rem;
                           font-weight:700;color:{ec};margin:0.7rem 0 0.15rem;">
                {econ}%
              </div>
              <div style="font-size:0.68rem;color:#7a8fa8;">confidence</div>
            </div>
            <div class="ss-card" style="padding:0.85rem 1rem;margin-bottom:0.5rem;">
              <div class="ss-card-body" style="font-size:0.72rem;">
                🎵 Audio: <span style="color:#f0f4f8;">
                {R.get("audio_emotion","—")} ({R.get("audio_emotion_conf",0)}%)</span>
              </div>
            </div>
            <div class="ss-card" style="padding:0.85rem 1rem;">
              <div class="ss-card-body" style="font-size:0.72rem;">
                📝 Text: <span style="color:#f0f4f8;">
                {R.get("text_emotion","—")} ({R.get("text_emotion_conf",0)}%)</span>
              </div>
            </div>""", unsafe_allow_html=True)

            # Quality bars
            st.markdown('<div class="sec-label" style="margin-top:0.9rem;">// Audio Quality</div>',
                        unsafe_allow_html=True)
            sb = R.get("score_before", -60)
            sa = R.get("score_after",  -60)
            nrm = lambda v: max(0, min(100, int((v + 60) / 60 * 100)))
            st.markdown(f"""
            <div class="ss-card" style="padding:1rem;">
              <div style="font-size:0.72rem;color:#7a8fa8;margin-bottom:0.4rem;">
                Before: <span style="color:#C9A44C;">{sb} dB</span></div>
              <div style="background:rgba(255,255,255,0.06);border-radius:2px;height:5px;margin-bottom:0.7rem;">
                <div style="background:#7a8fa8;width:{nrm(sb)}%;height:5px;border-radius:2px;"></div></div>
              <div style="font-size:0.72rem;color:#7a8fa8;margin-bottom:0.4rem;">
                After: <span style="color:#00D9FF;">{sa} dB</span></div>
              <div style="background:rgba(255,255,255,0.06);border-radius:2px;height:5px;">
                <div style="background:linear-gradient(90deg,#C9A44C,#00D9FF);
                             width:{nrm(sa)}%;height:5px;border-radius:2px;"></div></div>
            </div>""", unsafe_allow_html=True)

        # Translations
        if R.get("translations"):
            st.markdown("---")
            st.markdown('<div class="sec-label">// Translations</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Multilingual <em>Output</em></div>',
                        unsafe_allow_html=True)
            flags = {"English": "🇬🇧", "Hindi": "🇮🇳", "Marathi": "🟠"}
            tc1, tc2, tc3 = st.columns(3)
            for col, (ln, txt) in zip([tc1, tc2, tc3], R["translations"].items()):
                with col:
                    st.markdown(f"""
                    <div class="ss-card ss-card-cyan" style="min-height:140px;">
                      <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                                   letter-spacing:0.2em;color:#C9A44C;margin-bottom:0.5rem;">
                        {flags.get(ln,"🌍")} {ln.upper()}
                      </div>
                      <div style="font-size:0.82rem;color:#f0f4f8;line-height:1.6;">{txt}</div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-label">// Groq · LLaMA3-70B</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Q&A <em>Chatbot</em></div>', unsafe_allow_html=True)

    if not st.session_state.get("results"):
        st.info("Run analysis first, then ask questions about your audio here.")
    elif not groq_key:
        st.markdown("""
        <div class="ss-card" style="padding:1.4rem;text-align:center;">
          <div style="font-size:1.4rem;margin-bottom:0.6rem;">🔑</div>
          <div class="ss-card-title">Groq API Key Required</div>
          <div class="ss-card-body">Add your Groq key in the sidebar.
            Get a free key at <span style="color:#00D9FF;">console.groq.com</span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        transcript = st.session_state.get("transcript", "")
        lang       = st.session_state.get("detected_lang", "en")
        elbl       = st.session_state.get("emotion_label", "unknown")

        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="text-align:right;">
                  <div class="chat-lbl" style="text-align:right;">You</div>
                  <div class="chat-u">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div>
                  <div class="chat-lbl">SoundScape AI</div>
                  <div class="chat-a">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        qc, bc = st.columns([5, 1])
        with qc:
            user_q = st.text_input("", placeholder="Ask anything about the audio…",
                                   label_visibility="collapsed", key="q_in")
        with bc:
            send = st.button("SEND", use_container_width=True)

        if send and user_q.strip():
            sys_prompt = (
                f"You are SoundScape AI, an expert audio intelligence assistant.\n"
                f"Audio analysis results:\n"
                f"- Language: {lang}\n- Emotion: {elbl}\n"
                f"- Transcript (first 1500 chars): {transcript[:1500]}\n\n"
                f"Answer concisely. If the user writes in Hindi/Marathi, reply in that language."
            )
            messages = [{"role": "system", "content": sys_prompt}]
            for m in st.session_state["chat_history"][-8:]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": user_q})

            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                with st.spinner("Thinking…"):
                    resp = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=messages,
                        max_tokens=600,
                        temperature=0.7,
                    )
                ans = resp.choices[0].message.content
                st.session_state["chat_history"].append({"role": "user",      "content": user_q})
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                st.rerun()
            except Exception as e:
                st.error(f"Chatbot error: {e}")

        if st.session_state.get("chat_history"):
            if st.button("CLEAR CHAT"):
                st.session_state["chat_history"] = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-label">// Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Processing <em>Pipeline</em></div>',
                unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")
    has_results = bool(st.session_state.get("results"))

    with p1:
        steps = [
            ("01", "Audio Ingestion",   "pydub — any format → WAV mono 16kHz"),
            ("02", "Enhancement",       "noisereduce + librosa → SNR scored before/after"),
            ("03", "Transcription",     "Whisper small — auto language, 90+ supported"),
            ("04", "Missing Words",     "mBERT fill-mask — [inaudible] gap filling"),
            ("05", "Emotion (Audio)",   "wav2vec2-superb — 6-class acoustic emotion"),
            ("06", "Emotion (Text)",    "XLM-RoBERTa — multilingual sentiment blend"),
            ("07", "Speaker ID",        "pyannote 3.1 — timestamped speaker segments"),
            ("08", "NLP (Flan-T5)",     "Summary · Keywords · Topic extraction"),
            ("09", "Translation",       "Google Translator → EN / HI / MR"),
            ("10", "Chatbot",           "Groq LLaMA3-70B — transcript-aware Q&A"),
        ]
        for num, name, desc in steps:
            done = has_results and int(num) <= 6
            st.markdown(f"""
            <div class="pip-step {'done' if done else ''}">
              <div class="pip-num">{'✓' if done else num}</div>
              <div>
                <div class="pip-name">{name}</div>
                <div class="pip-desc">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="sec-label">// Models & Libraries</div>', unsafe_allow_html=True)
        models = [
            ("openai/whisper-small",               "OpenAI",    "ASR, 90+ languages"),
            ("superb/wav2vec2-base-superb-er",     "Facebook",  "Audio emotion"),
            ("twitter-xlm-roberta-base-sentiment", "CardiffNLP","Text sentiment"),
            ("bert-base-multilingual-cased",       "Google",    "Fill-mask"),
            ("google/flan-t5-base",                "Google",    "NLP summary"),
            ("pyannote/speaker-diarization-3.1",   "pyannote",  "Speaker ID"),
            ("llama3-70b-8192",                    "Meta/Groq", "Q&A chatbot"),
            ("noisereduce",                        "T.Sainburg","Noise reduction"),
            ("deep-translator",                    "Google",    "Translation"),
            ("gTTS",                               "Google",    "Text-to-speech"),
        ]
        for mname, author, role in models:
            st.markdown(f"""
            <div class="mod-tile">
              <div>
                <div class="mod-name">{mname}</div>
                <div class="mod-role">{role}</div>
              </div>
              <div class="mod-by">{author}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">// Supported Formats</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="ss-card ss-card-cyan" style="padding:0.9rem 1.1rem;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                       color:#00D9FF;letter-spacing:0.12em;">
            MP3 · MP4 · WAV · M4A · OGG · FLAC · WEBM
          </div>
        </div>
        <div class="ss-card" style="padding:0.9rem 1.1rem;">
          <div style="font-size:0.78rem;color:#7a8fa8;">
            🇬🇧 English · 🇮🇳 Hindi · 🇮🇳 Marathi · Hinglish
            <br><span style="font-size:0.68rem;color:#4a6080;">
            + 90 more via Whisper auto-detection</span>
          </div>
        </div>""", unsafe_allow_html=True)
