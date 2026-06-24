"""
═══════════════════════════════════════════════════════════════════════
  MESIN FLOW READER v1.0  —  Per-Saham Deep Dive
  Input manual: Stockbit / Neo BDM (broker summary + bandar + running)
  Output: Flow Score (rule-based) + Narasi AI + Chart visual
═══════════════════════════════════════════════════════════════════════
  Run:  streamlit run flow_reader.py
  Dep:  pip install streamlit pandas plotly requests pillow
        (pillow buat infografis PNG/PDF — gak perlu wkhtml/Chrome/kaleido)
  Config: simpan key & telegram di .streamlit/secrets.toml (ada template-nya)
  AI :  Gemini (set GEMINI_API_KEY / paste di app) ATAU Ollama lokal (gratis).
        Model: GEMINI_MODEL / OLLAMA_MODEL.
═══════════════════════════════════════════════════════════════════════
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── AI BACKEND: Gemini (API) + Ollama (lokal, gratis) ───────────────
import requests

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "openai/gpt-4o")
KEY_FILE = "gemini_key.txt"  # opsional: simpan key di file lokal (sekali aja)


def _secret(key, default=""):
    """Baca dari .streamlit/secrets.toml dengan aman (gak error kalau gak ada).
    Nilai template (PASTE_...) dianggap kosong."""
    try:
        v = str(st.secrets.get(key, default)).strip()
        if v.startswith("PASTE_"):  # template belum diisi
            return default
        return v
    except Exception:
        return default


def get_gemini_key():
    """Ambil API key dari: input box → secrets.toml → file lokal → env var."""
    k = st.session_state.get("gemini_key", "").strip()
    if k:
        return k
    k = _secret("GEMINI_API_KEY")
    if k:
        return k
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE) as f:
                k = f.read().strip()
            if k:
                return k
        except Exception:
            pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


def call_gemini(prompt, max_tokens=8192, max_retries=5):
    """Panggil Gemini generateContent (REST), auto-retry agresif kalau server sibuk."""
    import time
    api_key = get_gemini_key()
    if not api_key:
        return None
    # model utama + fallback (kalau model utama overload terus, coba yang lain)
    models = [GEMINI_MODEL]
    for fb in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"):
        if fb not in models:
            models.append(fb)
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    RETRYABLE = {429, 500, 502, 503, 504}
    last_err = ""
    # loop model: kalau 1 model sibuk terus, pindah ke model fallback
    for mi, model in enumerate(models):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent")
        for attempt in range(max_retries):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=90)
                if r.status_code == 200:
                    data = r.json()
                    cands = data.get("candidates", [])
                    if not cands:
                        fbk = data.get("promptFeedback", {})
                        return f"⚠️ Gemini gak balikin output (mungkin kena filter). {fbk}"
                    cand = cands[0]
                    parts = cand.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()
                    finish = cand.get("finishReason", "")
                    if not text:
                        return (f"⚠️ Gemini balikin kosong (finishReason: {finish}). "
                                f"Coba lagi atau pakai Ollama lokal.")
                    if finish == "MAX_TOKENS":
                        text += "\n\n_(...output dipotong limit token, inti analisa udah di atas)_"
                    return text
                if r.status_code in RETRYABLE and attempt < max_retries - 1:
                    wait = min(2 ** attempt, 8)  # 1,2,4,8,8 detik (cap 8)
                    last_err = f"{r.status_code} (server sibuk)"
                    time.sleep(wait)
                    continue
                if r.status_code in RETRYABLE:
                    # model ini nyerah, coba model fallback berikutnya
                    last_err = f"{r.status_code} (server sibuk) di model {model}"
                    break
                return f"⚠️ Gemini API error {r.status_code}: {r.text[:300]}"
            except requests.exceptions.Timeout:
                last_err = "timeout"
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
            except Exception as e:
                return f"⚠️ Gemini error: {e}"
    return ("⚠️ GEMINI_BUSY")  # penanda khusus → UI fallback ke summary lokal


def call_ollama(prompt, max_tokens=8192):
    """
    Panggil Ollama lokal (gratis, tanpa API key). Perlu Ollama jalan di
    localhost:11434 + model udah di-pull (cth: `ollama pull llama3.1`).
    Return teks / '⚠️ ...'.
    """
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": max_tokens},
            },
            timeout=300,  # model lokal bisa lambat, kasih napas
        )
        if r.status_code != 200:
            return (f"⚠️ Ollama error {r.status_code}: {r.text[:200]}. "
                    f"Pastikan Ollama jalan & model '{OLLAMA_MODEL}' udah di-pull.")
        return r.json().get("response", "").strip() or "⚠️ Ollama balikin kosong."
    except requests.exceptions.ConnectionError:
        return ("⚠️ Gak bisa konek ke Ollama. Pastikan Ollama udah jalan "
                "(buka app Ollama atau jalankan `ollama serve`), dan model "
                f"'{OLLAMA_MODEL}' udah di-pull (`ollama pull {OLLAMA_MODEL}`).")
    except Exception as e:
        return f"⚠️ Ollama error: {e}"


def get_github_token():
    """Ambil GitHub PAT dari: input box → secrets.toml → env var."""
    k = st.session_state.get("github_token", "").strip()
    if k:
        return k
    k = _secret("GITHUB_TOKEN")
    if k:
        return k
    return os.environ.get("GITHUB_TOKEN", "").strip()


def call_github_models(prompt, max_tokens=8192, max_retries=4):
    """
    Panggil GitHub Models (GPT-4o, gratis pakai GitHub PAT). OpenAI-compatible.
    Cocok buat deploy ke cloud (gak butuh PC kuat kayak Ollama lokal).
    Return teks / None / '⚠️ ...'.
    """
    import time
    token = get_github_token()
    if not token:
        return None
    url = "https://models.github.ai/inference/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "model": GITHUB_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": min(max_tokens, 16384),
    }
    RETRYABLE = {429, 500, 502, 503, 504}
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=90)
            if r.status_code == 200:
                data = r.json()
                ch = data.get("choices", [])
                if not ch:
                    return "⚠️ GitHub Models gak balikin output."
                return ch[0].get("message", {}).get("content", "").strip() or \
                    "⚠️ GitHub Models balikin kosong."
            if r.status_code == 401:
                return ("⚠️ GitHub Token salah/expired. Pastikan PAT punya permission "
                        "`models: read`. Bikin ulang di github.com/settings/tokens.")
            if r.status_code == 429:
                # rate limit — GitHub Models free tier ketat (cth GPT-4o 50/hari)
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                return ("⚠️ GitHub Models kena rate limit (free tier terbatas: "
                        "GPT-4o ~50 request/hari). Coba lagi nanti atau pakai Gemini.")
            if r.status_code in RETRYABLE and attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 8))
                continue
            return f"⚠️ GitHub Models error {r.status_code}: {r.text[:250]}"
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 8))
                continue
            return "⚠️ GitHub Models timeout. Coba lagi atau pakai Gemini."
        except Exception as e:
            return f"⚠️ GitHub Models error: {e}"
    return "⚠️ GITHUB_BUSY"


def call_ai(prompt, max_tokens=8192):
    """
    Router: pilih backend AI sesuai pilihan user di app (session_state).
    Return teks / None / '⚠️ ...'.
    """
    backend = st.session_state.get("ai_backend", "Gemini (API)")
    if backend.startswith("Ollama"):
        return call_ollama(prompt, max_tokens)
    if backend.startswith("GitHub"):
        return call_github_models(prompt, max_tokens)
    return call_gemini(prompt, max_tokens)


def ai_ready():
    """Cek apakah backend AI yang dipilih siap dipakai."""
    backend = st.session_state.get("ai_backend", "Gemini (API)")
    if backend.startswith("Ollama"):
        return True  # asumsi user udah setup; error ditangani saat call
    if backend.startswith("GitHub"):
        return bool(get_github_token())
    return bool(get_gemini_key())


# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG + STYLE (Bloomberg dark)
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Flow Reader", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background:#0a0e14; color:#d4d4d4; }
    .block-container { padding-top:1.2rem; max-width:1400px; }
    h1,h2,h3 { color:#e8b339 !important; font-family:'Consolas',monospace; }
    .metric-box {
        background:#11161f; border:1px solid #1e2530; border-radius:6px;
        padding:14px 16px; text-align:center;
    }
    .metric-box .lbl { color:#6b7280; font-size:11px; letter-spacing:1px; text-transform:uppercase; }
    .metric-box .val { font-size:26px; font-weight:700; font-family:'Consolas',monospace; }
    .verdict {
        border-radius:8px; padding:18px 22px; margin:8px 0;
        font-family:'Consolas',monospace; font-size:15px; line-height:1.7;
    }
    .stDataFrame { border:1px solid #1e2530; }
    div[data-testid="stMetricValue"] { font-family:'Consolas',monospace; }
    .tag { display:inline-block; padding:3px 10px; border-radius:4px; font-size:11px; margin-right:6px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# HELPERS — parsing input manual
# ════════════════════════════════════════════════════════════════════
def parse_broker_paste(text: str) -> pd.DataFrame:
    """
    Parse paste broker summary dari Stockbit.
    Format fleksibel: Broker | NBLot | NBVal | NBAvg  ||  Broker | NSLot | NSVal | NSAvg
    Atau baris per broker dipisah tab/koma/spasi.
    Auto-detect kolom yang ada.
    """
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # pisah by tab dulu, fallback ke koma, fallback ke multi-space
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t")]
        elif "," in line:
            parts = [p.strip() for p in line.split(",")]
        else:
            parts = line.split()
        rows.append(parts)
    return rows


def to_num(x):
    """Konversi '1,234' / '1.2K' / '3.4M' / '12,3' ke float."""
    if x is None:
        return 0.0
    s = str(x).strip().replace(" ", "")
    if s in ("", "-", "—"):
        return 0.0
    mult = 1.0
    if s[-1:].upper() == "K":
        mult, s = 1e3, s[:-1]
    elif s[-1:].upper() == "M":
        mult, s = 1e6, s[:-1]
    elif s[-1:].upper() == "B":
        mult, s = 1e9, s[:-1]
    s = s.replace(",", "")
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


# ════════════════════════════════════════════════════════════════════
# SCORING ENGINE (rule-based) — inti baca flow
# ════════════════════════════════════════════════════════════════════
def compute_flow_score(buy_df, sell_df, last_price, running, bandar):
    """
    Hitung Flow Score 0–100 dari komponen:
      1. Net concentration  — barang pindah ke sedikit broker besar? (akumulasi)
      2. Avg price posisi    — bandar avg di bawah harga (nyaman) vs di atas (nyangkut)
      3. Big vs Retail       — strong hands net buy?
      4. Running pressure    — lifting offer dominan vs hitting bid
    Return: dict skor + breakdown
    """
    breakdown = {}
    score = 50.0  # netral

    # ---- 1. KONSENTRASI BUYER (akumulasi = barang ke sedikit tangan) ----
    # Hanya valid kalau ada cukup broker dalam list. 1-2 broker = data terlalu
    # tipis untuk menyimpulkan konsentrasi (konsentrasi 100% dari 1 broker itu
    # artefak data, bukan sinyal akumulasi).
    if buy_df is not None and len(buy_df) >= 3:
        total_buy = buy_df["val"].sum()
        top3_buy = buy_df.nlargest(3, "val")["val"].sum()
        conc = (top3_buy / total_buy) if total_buy > 0 else 0
        # konsentrasi tinggi (>60%) = bandar fokus akumulasi
        c_score = (conc - 0.4) * 100  # 0.4->0, 0.9->50
        c_score = max(-15, min(20, c_score))
        score += c_score
        breakdown["Konsentrasi Buyer"] = (conc, c_score)
    else:
        # data tipis → tidak berkontribusi (netral)
        conc = (buy_df.nlargest(3, "val")["val"].sum() / buy_df["val"].sum()) \
            if (buy_df is not None and buy_df["val"].sum() > 0) else 0
        breakdown["Konsentrasi Buyer"] = (conc, 0)

    # ---- 2. AVG PRICE TOP BUYER vs HARGA ----
    if buy_df is not None and "avg" in buy_df.columns and last_price > 0:
        top_buyers = buy_df.nlargest(3, "val")
        valid_avg = top_buyers[top_buyers["avg"] > 0]
        if len(valid_avg) > 0:
            w_avg = (valid_avg["avg"] * valid_avg["val"]).sum() / valid_avg["val"].sum()
            # avg buyer di bawah harga = posisi profit/nyaman = bullish flow
            disc = (last_price - w_avg) / last_price  # +ve = buyer profit
            a_score = disc * 120
            a_score = max(-15, min(15, a_score))
            score += a_score
            breakdown["Avg Buyer vs Harga"] = (w_avg, a_score)
        else:
            breakdown["Avg Buyer vs Harga"] = (0, 0)
    else:
        breakdown["Avg Buyer vs Harga"] = (0, 0)

    # ---- 3. BIG vs RETAIL (strong hands) ----
    big_net = bandar.get("big_net", 0)
    retail_net = bandar.get("retail_net", 0)
    if big_net != 0 or retail_net != 0:
        # ideal: big net buy + retail net sell (smart money akumulasi dari ritel)
        if big_net > 0 and retail_net < 0:
            b_score = 15
        elif big_net > 0:
            b_score = 8
        elif big_net < 0 and retail_net > 0:
            b_score = -15  # distribusi ke ritel
        else:
            b_score = -5
        score += b_score
        breakdown["Big vs Retail"] = (big_net, b_score)
    else:
        breakdown["Big vs Retail"] = (0, 0)

    # ---- 4. RUNNING PRESSURE ----
    lift = running.get("lifting", 0)
    hit = running.get("hitting", 0)
    if lift + hit > 0:
        press = (lift - hit) / (lift + hit)  # -1..+1
        r_score = press * 15
        score += r_score
        breakdown["Running Pressure"] = (press, r_score)
    else:
        breakdown["Running Pressure"] = (0, 0)

    score = max(0, min(100, score))

    # ---- CONFIDENCE: berapa komponen yang punya data valid ----
    valid_components = 0
    if buy_df is not None and len(buy_df) >= 3:
        valid_components += 1
    if breakdown.get("Avg Buyer vs Harga", (0, 0))[1] != 0:
        valid_components += 1
    if bandar.get("big_net", 0) != 0 or bandar.get("retail_net", 0) != 0:
        valid_components += 1
    if running.get("lifting", 0) + running.get("hitting", 0) > 0:
        valid_components += 1

    if valid_components >= 3:
        confidence = "TINGGI"
    elif valid_components == 2:
        confidence = "SEDANG"
    else:
        confidence = "RENDAH (data minim)"

    # verdict
    if score >= 70:
        verdict, color, tag = "AKUMULASI KUAT", "#16c784", "STRONG BUY FLOW"
    elif score >= 58:
        verdict, color, tag = "AKUMULASI", "#56d364", "ACCUM"
    elif score >= 43:
        verdict, color, tag = "NETRAL / RANGING", "#e8b339", "NEUTRAL"
    elif score >= 30:
        verdict, color, tag = "DISTRIBUSI", "#f0883e", "DISTRIB"
    else:
        verdict, color, tag = "DISTRIBUSI KUAT", "#f85149", "STRONG SELL FLOW"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "color": color,
        "tag": tag,
        "confidence": confidence,
        "valid_components": valid_components,
        "breakdown": breakdown,
    }


def build_flow_summary(result, buy_df, sell_df, last_price, bandar, ticker):
    """
    Summary spesifik (rule-based, tanpa API): siapa net buyer/seller terkuat,
    posisi avg-nya, dan arah kecenderungan. Return list of (label, text, color).
    """
    lines = []
    # --- broker terkuat tiap sisi ---
    top_buyer = None
    top_seller = None
    if buy_df is not None and len(buy_df) > 0:
        tb = buy_df.nlargest(1, "val").iloc[0]
        top_buyer = (tb["broker"], tb["val"], tb.get("avg", 0))
    if sell_df is not None and len(sell_df) > 0:
        ts = sell_df.nlargest(1, "val").iloc[0]
        top_seller = (ts["broker"], ts["val"], ts.get("avg", 0))

    def _fmt(v):
        # B = miliar (billion), jt = juta — sesuai konvensi Stockbit/IDX
        if v >= 1e9:
            return f"{v/1e9:.1f}B"
        if v >= 1e6:
            return f"{v/1e6:.0f}jt"
        return f"{v:,.0f}"

    if top_buyer:
        b_name, b_val, b_avg = top_buyer
        avg_note = ""
        if b_avg > 0 and last_price > 0:
            diff_pct = (b_avg - last_price) / last_price
            if diff_pct < -0.005:
                avg_note = f", avg {b_avg:,.0f} (di bawah harga — posisi nyaman/profit)"
            elif diff_pct > 0.005:
                avg_note = f", avg {b_avg:,.0f} (di atas harga — masih nyangkut, perlu jaga harga)"
            else:
                avg_note = f", avg {b_avg:,.0f} (setara harga sekarang — baru masuk)"
        lines.append(("🟢 NET BUYER TERKUAT",
                      f"**{b_name}** borong {_fmt(b_val)}{avg_note}", "#16c784"))

    if top_seller:
        s_name, s_val, s_avg = top_seller
        lines.append(("🔴 NET SELLER TERKUAT",
                      f"**{s_name}** lepas {_fmt(s_val)}", "#f85149"))

    # --- battle: net buyer vs net seller total ---
    total_buy = buy_df["val"].sum() if buy_df is not None else 0
    total_sell = sell_df["val"].sum() if sell_df is not None else 0
    net = total_buy - total_sell
    if total_buy + total_sell > 0:
        if net > 0:
            lines.append(("⚖️ NET FLOW",
                          f"Sisi beli unggul **+{_fmt(abs(net))}** — demand > supply hari ini",
                          "#16c784"))
        elif net < 0:
            lines.append(("⚖️ NET FLOW",
                          f"Sisi jual unggul **−{_fmt(abs(net))}** — supply > demand hari ini",
                          "#f85149"))

    # --- ARAH (kesimpulan) ---
    score = result["score"]
    conc = result["breakdown"].get("Konsentrasi Buyer", (0, 0))[0]
    if score >= 70:
        if top_buyer:
            arah = (f"Barang lagi diserap ke **{top_buyer[0]}** (konsentrasi {conc:.0%}). "
                    f"Selama {top_buyer[0]} belum balik jadi net seller, bias **NAIK** / lanjut markup. "
                    f"Pantau: kalau dia mulai distribusi di harga atas, itu sinyal exit.")
        else:
            arah = "Tekanan beli dominan, bias naik."
        arah_color = "#16c784"
    elif score >= 58:
        arah = (f"Akumulasi berlangsung tapi belum agresif. Bias **NAIK terbatas** — "
                f"butuh konfirmasi volume / breakout buat lanjut. Cocok buat akumulasi bertahap.")
        arah_color = "#56d364"
    elif score >= 43:
        arah = (f"Beli & jual imbang, belum ada pihak yang dominan kuat. Kemungkinan **RANGING/sideways**. "
                f"Tunggu salah satu sisi menang dulu sebelum entry — jangan tebak arah.")
        arah_color = "#e8b339"
    elif score >= 30:
        seller_name = top_seller[0] if top_seller else "broker besar"
        arah = (f"**{seller_name}** lagi distribusi. Bias **TURUN / koreksi** selama tekanan jual lanjut. "
                f"Hati-hati buy — tunggu seller habis dulu (net sell mengecil) baru pertimbangkan.")
        arah_color = "#f0883e"
    else:
        seller_name = top_seller[0] if top_seller else "broker besar"
        arah = (f"Distribusi kuat oleh **{seller_name}**. Bias **TURUN**. "
                f"Hindari averaging down — strong hands lagi buang barang. Lebih baik wait & see.")
        arah_color = "#f85149"

    lines.append(("🎯 ARAH KECENDERUNGAN", arah, arah_color))
    return lines


# ════════════════════════════════════════════════════════════════════
# TICK CARD — status warna (GREEN / RED / GREY / GOLDEN bagger)
# ════════════════════════════════════════════════════════════════════
def get_tick_card(result):
    """
    Tentukan tick card berdasarkan flow score + confidence.
      🏆 GOLDEN  : score >= 85 DAN confidence TINGGI (potensi bagger)
      🟢 GREEN   : akumulasi (score >= 58)
      🔴 RED     : distribusi (score < 43)
      ⚪ GREY    : netral / ragu (43-58) atau confidence rendah
    Return dict: tier, label, emoji, color, bg, desc.
    """
    score = result["score"]
    conf = result.get("confidence", "")
    is_high_conf = conf.startswith("TINGGI")

    # GOLDEN: bagger candidate — butuh skor tinggi + confidence tinggi
    if score >= 85 and is_high_conf:
        return {
            "tier": "GOLDEN", "label": "GOLDEN — POTENSI BAGGER", "emoji": "🏆",
            "color": "#ffd700", "bg": "linear-gradient(135deg,#3d2f00,#1a1400)",
            "desc": "Akumulasi super kuat + confidence tinggi. Kandidat multi-bagger — "
                    "watchlist prioritas, tapi tetap validasi & atur risk.",
        }
    if score >= 70:
        return {
            "tier": "GREEN", "label": "GREEN — AKUMULASI KUAT", "emoji": "🟢",
            "color": "#16c784", "bg": "#0d2a1f",
            "desc": "Flow beli dominan & terkonsentrasi. Bias naik.",
        }
    if score >= 58:
        return {
            "tier": "GREEN", "label": "GREEN — AKUMULASI", "emoji": "🟢",
            "color": "#56d364", "bg": "#0d2419",
            "desc": "Akumulasi berlangsung, belum agresif. Bias naik terbatas.",
        }
    if score >= 43:
        return {
            "tier": "GREY", "label": "GREY — NETRAL / RAGU", "emoji": "⚪",
            "color": "#9ca3af", "bg": "#1a1d23",
            "desc": "Beli & jual imbang. Belum jelas arah — tunggu konfirmasi.",
        }
    if score >= 30:
        return {
            "tier": "RED", "label": "RED — DISTRIBUSI", "emoji": "🔴",
            "color": "#f0883e", "bg": "#2a1a0d",
            "desc": "Tekanan jual dominan. Bias turun / koreksi.",
        }
    return {
        "tier": "RED", "label": "RED — DISTRIBUSI KUAT", "emoji": "🔴",
        "color": "#f85149", "bg": "#2a0f0d",
        "desc": "Distribusi kuat. Bias turun. Hindari averaging down.",
    }


# ════════════════════════════════════════════════════════════════════
# TELEGRAM — kirim summary + chart + data broker
# ════════════════════════════════════════════════════════════════════
def get_telegram_config():
    """Ambil bot token + chat id dari: input app → secrets.toml → env var."""
    token = (st.session_state.get("tg_token", "").strip()
             or _secret("TELEGRAM_TOKEN")
             or os.environ.get("TELEGRAM_TOKEN", "").strip())
    chat_id = (st.session_state.get("tg_chat", "").strip()
               or _secret("TELEGRAM_CHAT_ID")
               or os.environ.get("TELEGRAM_CHAT_ID", "").strip())
    return token, chat_id


def _tg_post(url, **kwargs):
    """POST ke Telegram, return (ok, detail). Cek field 'ok' di JSON, bukan cuma HTTP status."""
    try:
        r = requests.post(url, timeout=60, **kwargs)
        try:
            j = r.json()
        except Exception:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        if j.get("ok"):
            return True, "ok"
        # Telegram kasih pesan error spesifik di description
        desc = j.get("description", "unknown error")
        code = j.get("error_code", r.status_code)
        return False, f"[{code}] {desc}"
    except Exception as e:
        return False, str(e)


def telegram_test():
    """Cek bot token + chat ID valid. Return (ok, pesan jelas)."""
    token, chat_id = get_telegram_config()
    if not token:
        return False, "Bot Token belum diisi."
    if not chat_id:
        return False, "Chat ID belum diisi."
    base = f"https://api.telegram.org/bot{token}"
    # 1. cek token valid via getMe
    ok, detail = _tg_post(f"{base}/getMe")
    if not ok:
        return False, (f"Bot Token SALAH atau gak valid ({detail}). "
                       "Cek lagi token dari @BotFather, pastikan ke-copy penuh.")
    # 2. cek chat ID valid: kirim pesan test beneran
    ok, detail = _tg_post(f"{base}/sendMessage",
                          data={"chat_id": chat_id,
                                "text": "✅ Tes koneksi Flow Reader berhasil bro! Bot siap kirim analisa."})
    if not ok:
        # error chat id paling umum
        d = detail.lower()
        if "chat not found" in d:
            hint = ("Chat ID SALAH, atau lo BELUM chat bot-nya duluan. "
                    "Buka Telegram → cari bot lo → klik START / kirim 'halo' → coba lagi.")
        elif "bot was blocked" in d:
            hint = "Lo nge-block bot-nya. Unblock dulu di Telegram."
        elif "not enough rights" in d or "not a member" in d:
            hint = "Bot belum di-add ke grup, atau gak punya izin kirim. Add bot ke grup dulu."
        else:
            hint = "Cek Chat ID (pribadi: angka biasa; grup: diawali -100) & pastikan udah chat bot duluan."
        return False, f"Gagal kirim ke chat ({detail}). {hint}"
    return True, "✅ Koneksi OK! Pesan tes udah masuk ke Telegram lo. Cek HP."


def send_telegram(text, photo_bytes=None, caption_short=None):
    """
    Kirim ke Telegram secara bulletproof (tanpa markdown — sumber error utama).
      - Kalau ada photo_bytes: kirim foto + caption pendek, lalu badan teks terpisah.
      - Semua dikirim plain text (parse_mode dihindari biar gak ditolak Telegram).
    Cek field 'ok' tiap request. Return (ok: bool, msg: str).
    """
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        return False, "Bot token / chat ID belum diisi."

    base = f"https://api.telegram.org/bot{token}"
    # buang karakter markdown yang sering bikin Telegram nolak (* _ ` #)
    plain = text.replace("**", "").replace("*", "").replace("`", "")

    try:
        # 1. kirim foto dengan caption pendek (judul + score aja, plain)
        if photo_bytes is not None:
            cap = (caption_short or plain[:200])[:1024]
            ok, detail = _tg_post(
                f"{base}/sendPhoto",
                data={"chat_id": chat_id, "caption": cap},  # NO parse_mode
                files={"photo": ("chart.png", photo_bytes, "image/png")},
            )
            if not ok:
                return False, f"Gagal kirim foto: {_tg_hint(detail)}"

        # 2. kirim badan analisa sebagai teks (potong per 4000 char, plain)
        buf = plain
        sent_any = photo_bytes is not None
        while buf:
            chunk, buf = buf[:4000], buf[4000:]
            ok, detail = _tg_post(f"{base}/sendMessage",
                                  data={"chat_id": chat_id, "text": chunk})
            if not ok:
                if sent_any:
                    return False, f"Sebagian terkirim, lalu gagal: {_tg_hint(detail)}"
                return False, _tg_hint(detail)
            sent_any = True

        return True, "Terkirim ke Telegram ✓"
    except Exception as e:
        return False, f"Telegram error: {e}"


def send_telegram_document(file_bytes, filename, caption="", as_photo=False):
    """
    Kirim file ke Telegram. as_photo=True → sendPhoto (PNG keliatan langsung di chat).
    Else → sendDocument (PDF/file, bisa di-download). Return (ok, msg).
    """
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        return False, "Bot token / chat ID belum diisi."
    base = f"https://api.telegram.org/bot{token}"
    cap = (caption or "").replace("**", "").replace("*", "").replace("`", "")[:1024]
    try:
        if as_photo:
            ok, detail = _tg_post(
                f"{base}/sendPhoto",
                data={"chat_id": chat_id, "caption": cap},
                files={"photo": (filename, file_bytes, "image/png")},
            )
        else:
            mime = "application/pdf" if filename.lower().endswith(".pdf") else "image/png"
            ok, detail = _tg_post(
                f"{base}/sendDocument",
                data={"chat_id": chat_id, "caption": cap},
                files={"document": (filename, file_bytes, mime)},
            )
        if not ok:
            return False, _tg_hint(detail)
        return True, "Infografis terkirim ke Telegram ✓"
    except Exception as e:
        return False, f"Telegram error: {e}"


def _tg_hint(detail):
    """Ubah error Telegram jadi pesan yang jelas + solusi."""
    d = detail.lower()
    if "chat not found" in d:
        return ("Chat ID salah / lo belum chat bot duluan. "
                "Buka Telegram → START bot → coba lagi. " + detail)
    if "bot was blocked" in d:
        return "Lo block bot-nya. Unblock dulu. " + detail
    if "not enough rights" in d or "not a member" in d:
        return "Bot belum di grup / gak ada izin kirim. Add bot ke grup. " + detail
    if "unauthorized" in d:
        return "Bot Token salah/invalid. Cek dari @BotFather. " + detail
    return detail


def build_telegram_message(ticker, tf, result, card, summary, buy_df, sell_df,
                           last_price, narrative):
    """Rakit pesan Telegram: tick card + summary + data broker + narasi AI."""
    L = []
    L.append(f"{card['emoji']} *{ticker}* · {tf}")
    L.append(f"{card['label']}")
    L.append(f"Flow Score: *{result['score']}*/100 · Confidence: {result.get('confidence','-')}")
    L.append(f"Harga: {last_price:,.0f}")
    L.append("")
    # summary spesifik
    L.append("📌 *SUMMARY FLOW*")
    for label, text, _ in summary:
        clean = text.replace("**", "*")
        L.append(f"{label}: {clean}")
    L.append("")
    # data broker mentah (top 5)
    L.append("📊 *DATA BROKER*")
    if buy_df is not None:
        L.append("🟢 Net Buyer:")
        for _, r in buy_df.nlargest(5, "val").iterrows():
            v = f"{r['val']/1e9:.1f}B" if r['val'] >= 1e9 else f"{r['val']/1e6:.0f}jt"
            avg = f" @{r['avg']:,.0f}" if r.get('avg', 0) > 0 else ""
            L.append(f"  {r['broker']}: {v}{avg}")
    if sell_df is not None:
        L.append("🔴 Net Seller:")
        for _, r in sell_df.nlargest(5, "val").iterrows():
            v = f"{r['val']/1e9:.1f}B" if r['val'] >= 1e9 else f"{r['val']/1e6:.0f}jt"
            avg = f" @{r['avg']:,.0f}" if r.get('avg', 0) > 0 else ""
            L.append(f"  {r['broker']}: {v}{avg}")
    L.append("")
    # narasi AI (kalau ada)
    if narrative and not narrative.startswith("⚠️"):
        L.append("🤖 *ANALISA AI*")
        L.append(narrative)
    L.append("")
    L.append("_Flow Reader · alat bantu, bukan ajakan beli/jual_")
    return "\n".join(L)


# ════════════════════════════════════════════════════════════════════
# INFOGRAFIS — HTML 1 halaman → PNG / PDF
# ════════════════════════════════════════════════════════════════════
def _donut_svg(buy_df, size=240):
    """Bikin donut SVG share net buyer (top 8). Return string SVG."""
    if buy_df is None or len(buy_df) == 0:
        return ""
    d = buy_df.nlargest(8, "val").copy()
    total = d["val"].sum()
    if total <= 0:
        return ""
    palette = ["#16c784", "#58a6ff", "#e8b339", "#bc8cff", "#f0883e",
               "#f85149", "#39d3bb", "#db61a2"]
    cx, cy, r, rin = size/2, size/2, size/2 - 6, size/2 - 44
    import math
    a0 = -90  # mulai dari atas
    paths = []
    legend = []
    for i, (_, row) in enumerate(d.iterrows()):
        frac = row["val"] / total
        a1 = a0 + frac * 360
        large = 1 if (a1 - a0) > 180 else 0
        x0 = cx + r * math.cos(math.radians(a0)); y0 = cy + r * math.sin(math.radians(a0))
        x1 = cx + r * math.cos(math.radians(a1)); y1 = cy + r * math.sin(math.radians(a1))
        xi0 = cx + rin * math.cos(math.radians(a1)); yi0 = cy + rin * math.sin(math.radians(a1))
        xi1 = cx + rin * math.cos(math.radians(a0)); yi1 = cy + rin * math.sin(math.radians(a0))
        col = palette[i % len(palette)]
        paths.append(
            f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 {large} 1 {x1:.1f} {y1:.1f} '
            f'L {xi0:.1f} {yi0:.1f} A {rin} {rin} 0 {large} 0 {xi1:.1f} {yi1:.1f} Z" '
            f'fill="{col}"/>')
        legend.append((row["broker"], frac * 100, col, row["val"]))
        a0 = a1
    svg = (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
           + "".join(paths) + "</svg>")
    return svg, legend


def build_infographic_html(ticker, tf, result, card, summary, buy_df, sell_df,
                           last_price, narrative):
    """Rakit infografis HTML 1 halaman (dark Bloomberg style)."""
    donut_out = _donut_svg(buy_df)
    donut_svg, legend = ("", []) if not donut_out else donut_out

    # legend broker
    legend_html = ""
    for name, pct, col, val in legend:
        v = f"{val/1e9:.1f}B" if val >= 1e9 else f"{val/1e6:.0f}jt"
        legend_html += (
            f'<div class="lg"><span class="dot" style="background:{col}"></span>'
            f'<b>{name}</b><span class="lgv">{v} · {pct:.1f}%</span></div>')

    # summary rows
    summ_html = ""
    for label, text, color in summary:
        clean = text.replace("**", "").replace("*", "")
        summ_html += (
            f'<div class="srow" style="border-left:3px solid {color}">'
            f'<div class="slbl" style="color:{color}">{label}</div>'
            f'<div class="stxt">{clean}</div></div>')

    # broker tables
    def _btable(df, color, title):
        if df is None or len(df) == 0:
            return ""
        rows = ""
        for _, r in df.nlargest(6, "val").iterrows():
            v = f"{r['val']/1e9:.1f}B" if r['val'] >= 1e9 else f"{r['val']/1e6:.0f}jt"
            avg = f"{r['avg']:,.0f}" if r.get('avg', 0) > 0 else "-"
            rows += (f'<tr><td class="bk" style="color:{color}">{r["broker"]}</td>'
                     f'<td>{v}</td><td>{avg}</td></tr>')
        return (f'<div class="btbl"><div class="bttl" style="color:{color}">{title}</div>'
                f'<table><tr class="bh"><td>Broker</td><td>Value</td><td>Avg</td></tr>'
                f'{rows}</table></div>')

    buy_table = _btable(buy_df, "#16c784", "🟢 NET BUYER")
    sell_table = _btable(sell_df, "#f85149", "🔴 NET SELLER")

    # narasi AI (ringkas, ambil ~700 char pertama biar muat)
    narr_html = ""
    if narrative and not narrative.startswith("⚠️"):
        clean_narr = narrative.replace("##", "").replace("**", "").replace("*", "")
        # konversi heading sederhana
        narr_html = (f'<div class="narr"><div class="ntitle">🤖 ANALISA AI</div>'
                     f'<div class="ntxt">{clean_narr}</div></div>')

    glow = ("box-shadow:0 0 40px rgba(255,215,0,0.4);"
            if card["tier"] == "GOLDEN" else "")
    score_color = card["color"]

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0e14; color:#e6edf3; font-family:'Segoe UI',Arial,sans-serif;
       width:1000px; padding:32px; }}
.head {{ display:flex; justify-content:space-between; align-items:flex-start;
         border-bottom:2px solid #1e2530; padding-bottom:16px; margin-bottom:18px; }}
.brand {{ font-size:13px; color:#e8b339; letter-spacing:3px; font-weight:700; }}
.tk {{ font-size:48px; font-weight:800; letter-spacing:1px; }}
.tkmeta {{ color:#8b949e; font-size:15px; margin-top:2px; }}
.price {{ text-align:right; }}
.price .p {{ font-size:38px; font-weight:800; font-family:'Consolas',monospace; }}
.price .l {{ color:#8b949e; font-size:13px; letter-spacing:1px; }}
.card {{ background:{card['bg']}; border:2px solid {card['color']}; border-radius:14px;
         padding:20px 24px; margin-bottom:18px; {glow} }}
.card .cl {{ font-size:30px; font-weight:800; color:{card['color']};
             font-family:'Consolas',monospace; letter-spacing:1px; }}
.card .cd {{ color:#cbd5e1; font-size:14px; margin-top:6px; }}
.grid {{ display:flex; gap:18px; margin-bottom:18px; }}
.col {{ flex:1; }}
.panel {{ background:#0e141c; border:1px solid #1e2530; border-radius:12px;
          padding:16px 18px; margin-bottom:18px; }}
.ptitle {{ color:#e8b339; font-size:14px; font-weight:700; letter-spacing:1px;
           margin-bottom:12px; text-transform:uppercase; }}
.scorebox {{ text-align:center; }}
.scorebox .sv {{ font-size:64px; font-weight:800; color:{score_color};
                 font-family:'Consolas',monospace; line-height:1; }}
.scorebox .ss {{ color:#8b949e; font-size:14px; }}
.metrics {{ display:flex; gap:10px; margin-top:14px; }}
.metric {{ flex:1; background:#11161f; border-radius:8px; padding:10px; text-align:center; }}
.metric .mv {{ font-size:20px; font-weight:700; font-family:'Consolas',monospace; }}
.metric .ml {{ color:#6b7280; font-size:10px; text-transform:uppercase; margin-top:3px; }}
.donutwrap {{ display:flex; align-items:center; gap:18px; }}
.lg {{ display:flex; align-items:center; gap:8px; font-size:13px; margin:5px 0; }}
.dot {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
.lgv {{ color:#8b949e; margin-left:auto; font-family:'Consolas',monospace; font-size:12px; }}
.srow {{ background:#11161f; border-radius:6px; padding:9px 12px; margin:7px 0; }}
.slbl {{ font-size:11px; font-weight:700; letter-spacing:0.5px; }}
.stxt {{ color:#c9d1d9; font-size:13px; margin-top:3px; line-height:1.5; }}
.btbl {{ margin-bottom:14px; }}
.bttl {{ font-size:13px; font-weight:700; margin-bottom:6px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
td {{ padding:5px 8px; border-bottom:1px solid #161b22; }}
.bh td {{ color:#6b7280; font-size:11px; text-transform:uppercase; }}
.bk {{ font-weight:700; font-family:'Consolas',monospace; }}
table td:nth-child(2),table td:nth-child(3) {{ text-align:right;
       font-family:'Consolas',monospace; }}
.narr {{ background:#0e141c; border:1px solid #1e2530; border-radius:12px;
         padding:16px 18px; margin-bottom:14px; }}
.ntitle {{ color:#e8b339; font-size:14px; font-weight:700; letter-spacing:1px;
           margin-bottom:10px; }}
.ntxt {{ color:#c9d1d9; font-size:13px; line-height:1.6; white-space:pre-wrap; }}
.foot {{ text-align:center; color:#6b7280; font-size:12px; margin-top:8px;
         border-top:1px solid #1e2530; padding-top:12px; }}
.foot b {{ color:#e8b339; }}
</style></head><body>
  <div class="head">
    <div>
      <div class="brand">⚡ FLOW READER</div>
      <div class="tk" style="color:{card['color']}">{card['emoji']} {ticker}</div>
      <div class="tkmeta">{tf}</div>
    </div>
    <div class="price">
      <div class="l">HARGA</div>
      <div class="p">{last_price:,.0f}</div>
    </div>
  </div>

  <div class="card">
    <div class="cl">{card['label']}</div>
    <div class="cd">{card['desc']}</div>
  </div>

  <div class="grid">
    <div class="col">
      <div class="panel scorebox">
        <div class="ptitle">Flow Score</div>
        <div class="sv">{result['score']}<span style="font-size:24px;color:#6b7280">/100</span></div>
        <div class="ss">Confidence: {result.get('confidence','-')} · {result.get('valid_components','-')}/4 komponen</div>
        <div class="metrics">
          {_metrics_html(result)}
        </div>
      </div>
      <div class="panel">
        <div class="ptitle">Konsentrasi Net Buyer</div>
        <div class="donutwrap">
          {donut_svg}
          <div style="flex:1">{legend_html}</div>
        </div>
      </div>
    </div>
    <div class="col">
      <div class="panel">
        <div class="ptitle">📌 Summary Flow</div>
        {summ_html}
      </div>
    </div>
  </div>

  <div class="grid">
    <div class="col"><div class="panel">{buy_table}</div></div>
    <div class="col"><div class="panel">{sell_table}</div></div>
  </div>

  {narr_html}

  <div class="foot">Dibuat oleh <b>Flow Reader</b> · alat bantu baca flow, BUKAN ajakan beli/jual ·
  validasi sendiri & atur risk management</div>
</body></html>"""
    return html


def _metrics_html(result):
    """4 metric box buat infografis."""
    defs = [("Konsentrasi", "Konsentrasi Buyer", "{:.0%}"),
            ("Avg vs Hrg", "Avg Buyer vs Harga", "{:,.0f}"),
            ("Big/Retail", "Big vs Retail", "{:+,.0f}"),
            ("Orderbook", "Running Pressure", "{:+.0%}")]
    out = ""
    for short, key, fmt in defs:
        raw, pts = result["breakdown"].get(key, (0, 0))
        col = "#16c784" if pts > 0 else "#f85149" if pts < 0 else "#8b949e"
        try:
            disp = fmt.format(raw)
        except Exception:
            disp = str(raw)
        out += (f'<div class="metric"><div class="mv" style="color:{col}">{disp}</div>'
                f'<div class="ml">{short}</div></div>')
    return out


def _load_font(size, bold=False, mono=False):
    """Load font dengan fallback aman lintas OS."""
    from PIL import ImageFont
    # daftar kandidat font per kategori (Linux, Windows, Mac)
    if mono:
        cands = ["DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf",
                 "consolab.ttf" if bold else "consola.ttf",
                 "/System/Library/Fonts/Menlo.ttc"]
    elif bold:
        cands = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"]
    else:
        cands = ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"]
    # cari di path umum
    search_dirs = ["", "/usr/share/fonts/truetype/dejavu/",
                   "C:/Windows/Fonts/", "/Library/Fonts/", "/System/Library/Fonts/"]
    for name in cands:
        for d in search_dirs:
            try:
                return ImageFont.truetype(d + name, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def _hex(c):
    """Hex '#rrggbb' → (r,g,b) tuple."""
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))


def _wrap_text(draw, text, font, max_w):
    """Pecah teks jadi list baris yang muat di max_w pixel."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_infographic_pil(ticker, tf, result, card, summary, buy_df, sell_df,
                           last_price, narrative, scale=2):
    """
    Render infografis pakai Pillow MURNI (gak butuh wkhtml/Chrome/kaleido).
    Teknik: render di koordinat logical 1x, lalu canvas di-supersample S kali
    via ImageDraw yang otomatis scale semua koordinat & font → output tajam HD.
    Return PNG bytes.
    """
    from PIL import Image, ImageDraw
    import io as _io
    import math

    S = scale  # 2 = HD (~2160px), 3 = 4K-ish

    BG = _hex("#0a0e14")
    PANEL = _hex("#11171f")        # lebih terang → kontras lebih baik
    BORDER = _hex("#2a3441")       # border lebih jelas
    GOLD = _hex("#f0c040")         # gold lebih terang
    MUTE = _hex("#aeb9c4")         # mute lebih kebaca
    TXT = _hex("#f2f5f8")          # teks utama lebih putih
    ccol = _hex(card["color"])

    # ── Draw wrapper: semua koordinat & ukuran otomatis dikali S ──
    class SDraw:
        """Proxy ImageDraw yang scale semua koordinat numerik dengan S."""
        def __init__(self, draw, s):
            self._d = draw; self._s = s
        def _sc(self, v):
            if isinstance(v, (int, float)): return v * self._s
            if isinstance(v, (list, tuple)):
                return type(v)(self._sc(x) for x in v)
            return v
        def text(self, xy, *a, **k):
            self._d.text(self._sc(xy), *a, **k)
        def line(self, xy, **k):
            if "width" in k: k["width"] = int(k["width"] * self._s) or 1
            self._d.line(self._sc(xy), **k)
        def rectangle(self, xy, **k):
            self._d.rectangle(self._sc(xy), **k)
        def rounded_rectangle(self, xy, radius=0, **k):
            if "width" in k: k["width"] = int(k["width"] * self._s) or 1
            self._d.rounded_rectangle(self._sc(xy), radius=self._sc(radius), **k)
        def ellipse(self, xy, **k):
            self._d.ellipse(self._sc(xy), **k)
        def pieslice(self, xy, start, end, **k):
            self._d.pieslice(self._sc(xy), start, end, **k)
        def textlength(self, text, font=None):
            # balikin dalam koordinat LOGICAL (font di-scale, jadi bagi S)
            return self._d.textlength(text, font=font) / self._s

    def F(sz, **kw):  # font di-scale ke resolusi nyata
        return _load_font(int(sz * S), **kw)

    f_brand = F(18, bold=True); f_tk = F(52, bold=True); f_meta = F(18)
    f_price = F(40, bold=True, mono=True); f_card = F(34, bold=True, mono=True)
    f_cardd = F(17); f_h = F(18, bold=True); f_score = F(72, bold=True, mono=True)
    f_metric = F(22, bold=True, mono=True); f_small = F(13); f_body = F(15)
    f_bodyb = F(15, bold=True); f_mono = F(15, mono=True)

    W = 1080  # koordinat LOGICAL (real canvas = W*S)
    H = 5200
    img = Image.new("RGB", (W * S, H * S), BG)
    d = SDraw(ImageDraw.Draw(img), S)
    M = 32  # margin (logical)
    y = M

    def panel(x, yy, w, h, fill=PANEL, border=BORDER, bw=1, radius=12):
        d.rounded_rectangle([x, yy, x + w, yy + h], radius=radius,
                            fill=fill, outline=border, width=bw)

    def clean_emoji(s):
        """Buang emoji (DejaVu gak render emoji) — sisakan teks bersih."""
        import re
        return re.sub(r'[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]', '', s).strip()

    # ── HEADER ──
    d.text((M, y), "FLOW READER", font=f_brand, fill=GOLD)
    d.text((M, y + 26), ticker, font=f_tk, fill=ccol)
    d.text((M, y + 88), tf, font=f_meta, fill=MUTE)
    # harga kanan
    d.text((W - M, y + 4), "HARGA", font=f_small, fill=MUTE, anchor="ra")
    d.text((W - M, y + 22), f"{last_price:,.0f}", font=f_price, fill=TXT, anchor="ra")
    y += 120
    d.line([(M, y), (W - M, y)], fill=BORDER, width=2)
    y += 20

    # ── TICK CARD ──
    card_h = 96
    cbg = _hex("#3d2f00") if card["tier"] == "GOLDEN" else _hex(
        card["bg"] if card["bg"].startswith("#") else "#11161f")
    panel(M, y, W - 2 * M, card_h, fill=cbg, border=ccol, bw=2, radius=14)
    d.text((M + 22, y + 18), clean_emoji(card["label"]), font=f_card, fill=ccol)
    for i, ln in enumerate(_wrap_text(d, card["desc"], f_cardd, W - 2 * M - 44)[:2]):
        d.text((M + 22, y + 56 + i * 20), ln, font=f_cardd, fill=_hex("#cbd5e1"))
    y += card_h + 20

    # ── 2 KOLOM: kiri (score+donut), kanan (summary) ──
    col_w = (W - 2 * M - 18) // 2
    left_x, right_x = M, M + col_w + 18
    y_top = y

    # KIRI: Score panel
    sp_h = 200
    panel(left_x, y, col_w, sp_h)
    d.text((left_x + col_w // 2, y + 16), "FLOW SCORE", font=f_h, fill=GOLD, anchor="ma")
    d.text((left_x + col_w // 2, y + 38), f"{result['score']}", font=f_score,
           fill=ccol, anchor="ma")
    d.text((left_x + col_w // 2, y + 124),
           f"Confidence: {result.get('confidence','-')} · {result.get('valid_components','-')}/4",
           font=f_small, fill=MUTE, anchor="ma")
    # 4 metric box
    mdefs = [("KONSEN", "Konsentrasi Buyer", "{:.0%}"),
             ("AVG/HRG", "Avg Buyer vs Harga", "{:,.0f}"),
             ("BIG/RTL", "Big vs Retail", "{:+,.0f}"),
             ("ORDERBOOK", "Running Pressure", "{:+.0%}")]
    mw = (col_w - 20 - 3 * 8) // 4
    mx = left_x + 10
    my = y + 150
    for short, key, fmt in mdefs:
        raw, pts = result["breakdown"].get(key, (0, 0))
        mc = _hex("#16c784") if pts > 0 else _hex("#f85149") if pts < 0 else MUTE
        panel(mx, my, mw, 40, fill=_hex("#11161f"), border=_hex("#11161f"), radius=8)
        try:
            disp = fmt.format(raw)
        except Exception:
            disp = str(raw)
        d.text((mx + mw // 2, my + 6), disp, font=f_metric, fill=mc, anchor="ma")
        d.text((mx + mw // 2, my + 28), short, font=_load_font(9), fill=MUTE, anchor="ma")
        mx += mw + 8
    y_left = y + sp_h + 18

    # KIRI: Donut panel
    if buy_df is not None and len(buy_df) > 0:
        d_total = buy_df["val"].sum()
        if d_total > 0:
            dn = buy_df.nlargest(8, "val")
            dp_h = 280
            panel(left_x, y_left, col_w, dp_h)
            d.text((left_x + 18, y_left + 14), "KONSENTRASI NET BUYER", font=f_h, fill=GOLD)
            # donut
            cx, cy, rad, rin = left_x + 90, y_left + 160, 70, 42
            palette = ["#16c784", "#58a6ff", "#e8b339", "#bc8cff", "#f0883e",
                       "#f85149", "#39d3bb", "#db61a2"]
            a0 = -90
            for i, (_, row) in enumerate(dn.iterrows()):
                frac = row["val"] / d_total
                a1 = a0 + frac * 360
                d.pieslice([cx - rad, cy - rad, cx + rad, cy + rad], a0, a1,
                          fill=_hex(palette[i % len(palette)]))
                a0 = a1
            d.ellipse([cx - rin, cy - rin, cx + rin, cy + rin], fill=PANEL)
            # legend
            lx, ly = left_x + 180, y_left + 70
            for i, (_, row) in enumerate(dn.iterrows()):
                col = _hex(palette[i % len(palette)])
                pct = row["val"] / d_total * 100
                v = f"{row['val']/1e9:.1f}B" if row['val'] >= 1e9 else f"{row['val']/1e6:.0f}jt"
                d.rounded_rectangle([lx, ly + 2, lx + 11, ly + 13], radius=3, fill=col)
                d.text((lx + 18, ly), f"{row['broker']}", font=f_bodyb, fill=TXT)
                d.text((left_x + col_w - 14, ly), f"{v} · {pct:.1f}%", font=f_mono,
                       fill=MUTE, anchor="ra")
                ly += 22
            y_left += dp_h + 18

    # KANAN: Summary panel
    sum_h = max(y_left - y_top - 18, 200)
    panel(right_x, y_top, col_w, sum_h)
    d.text((right_x + 18, y_top + 14), "SUMMARY FLOW", font=f_h, fill=GOLD)
    sy = y_top + 44
    for label, text, color in summary:
        col = _hex(color)
        clean = text.replace("**", "").replace("*", "")
        d.rectangle([right_x + 14, sy, right_x + 17, sy + 18], fill=col)
        d.text((right_x + 26, sy), clean_emoji(label), font=_load_font(12, bold=True), fill=col)
        wrapped = _wrap_text(d, clean, f_body, col_w - 50)
        for j, ln in enumerate(wrapped[:4]):
            d.text((right_x + 26, sy + 20 + j * 19), ln, font=f_body, fill=_hex("#c9d1d9"))
        sy += 26 + min(len(wrapped), 4) * 19 + 8

    y = max(y_left, y_top + sum_h + 18)

    # ── TABEL BROKER (2 kolom) ──
    def broker_table(x, yy, df, color, title):
        if df is None or len(df) == 0:
            return yy
        rows = df.nlargest(6, "val")
        th = 50 + len(rows) * 26 + 16
        panel(x, yy, col_w, th)
        d.text((x + 18, yy + 12), title, font=f_h, fill=_hex(color))
        d.text((x + 18, yy + 40), "BROKER", font=_load_font(11), fill=MUTE)
        d.text((x + col_w - 130, yy + 40), "VALUE", font=_load_font(11), fill=MUTE)
        d.text((x + col_w - 18, yy + 40), "AVG", font=_load_font(11), fill=MUTE, anchor="ra")
        ry = yy + 62
        for _, r in rows.iterrows():
            v = f"{r['val']/1e9:.1f}B" if r['val'] >= 1e9 else f"{r['val']/1e6:.0f}jt"
            avg = f"{r['avg']:,.0f}" if r.get('avg', 0) > 0 else "-"
            d.text((x + 18, ry), r["broker"], font=f_mono, fill=_hex(color))
            d.text((x + col_w - 130, ry), v, font=f_mono, fill=TXT)
            d.text((x + col_w - 18, ry), avg, font=f_mono, fill=TXT, anchor="ra")
            ry += 26
        return yy + th

    yb1 = broker_table(left_x, y, buy_df, "#16c784", "NET BUYER")
    yb2 = broker_table(right_x, y, sell_df, "#f85149", "NET SELLER")
    y = max(yb1, yb2) + 18

    # ── NARASI AI ──
    if narrative and not narrative.startswith("⚠️"):
        clean = narrative.replace("##", "").replace("**", "").replace("*", "").replace("`", "")
        lines = []
        for para in clean.split("\n"):
            para = para.strip()
            if not para:
                lines.append("")
                continue
            lines.extend(_wrap_text(d, para, f_body, W - 2 * M - 36))
        nh = 50 + len(lines) * 19 + 16
        panel(M, y, W - 2 * M, nh)
        d.text((M + 18, y + 14), "ANALISA AI", font=f_h, fill=GOLD)
        ny = y + 44
        for ln in lines:
            d.text((M + 18, ny), ln, font=f_body, fill=_hex("#c9d1d9"))
            ny += 19
        y += nh + 18

    # ── FOOTER ──
    d.line([(M, y), (W - M, y)], fill=BORDER, width=1)
    y += 12
    d.text((W // 2, y), "Dibuat oleh Flow Reader · alat bantu baca flow, BUKAN ajakan beli/jual",
           font=f_small, fill=MUTE, anchor="ma")
    y += 30

    # crop ke tinggi konten + simpan (koordinat real = logical * S)
    real_h = min(int(y * S), H * S)
    img = img.crop((0, 0, W * S, real_h))
    buf = _io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def png_to_pdf(png_bytes):
    """Konversi PNG bytes → PDF bytes pakai Pillow (gak butuh tool eksternal)."""
    from PIL import Image
    import io as _io
    try:
        img = Image.open(_io.BytesIO(png_bytes)).convert("RGB")
        buf = _io.BytesIO()
        img.save(buf, "PDF", resolution=100)
        return buf.getvalue()
    except Exception:
        return None


def render_infographic(html, fmt="png"):
    """Legacy wkhtml render — disimpan buat fallback, tapi default pakai Pillow."""
    import subprocess, tempfile, shutil
    tool = "wkhtmltoimage" if fmt == "png" else "wkhtmltopdf"
    if shutil.which(tool) is None:
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            hp = os.path.join(td, "in.html")
            op = os.path.join(td, f"out.{fmt}")
            with open(hp, "w", encoding="utf-8") as f:
                f.write(html)
            if fmt == "png":
                cmd = [tool, "--quality", "92", "--width", "1064",
                       "--enable-local-file-access", hp, op]
            else:
                cmd = [tool, "--enable-local-file-access",
                       "--page-width", "280mm", "--page-height", "400mm",
                       "--margin-top", "0", "--margin-bottom", "0",
                       "--margin-left", "0", "--margin-right", "0", hp, op]
            subprocess.run(cmd, capture_output=True, timeout=60)
            if not os.path.exists(op):
                return None
            with open(op, "rb") as f:
                return f.read()
    except Exception:
        return None


def chart_to_png(fig):
    """Konversi plotly figure ke PNG bytes (butuh kaleido). Return bytes / None."""
    try:
        return fig.to_image(format="png", width=1000, height=500, scale=2)
    except Exception:
        return None


def build_chart(buy_df, sell_df, last_price, ticker):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("TOP NET BUYER", "TOP NET SELLER"),
        horizontal_spacing=0.12,
    )

    if buy_df is not None and len(buy_df) > 0:
        b = buy_df.nlargest(8, "val").sort_values("val")
        fig.add_trace(go.Bar(
            y=b["broker"], x=b["val"], orientation="h",
            marker_color="#16c784", name="Buy",
            text=[f"{v/1e9:.1f}B" if v >= 1e9 else f"{v/1e6:.0f}jt" for v in b["val"]],
            textposition="outside", textfont=dict(color="#16c784"),
        ), row=1, col=1)

    if sell_df is not None and len(sell_df) > 0:
        s = sell_df.nlargest(8, "val").sort_values("val")
        fig.add_trace(go.Bar(
            y=s["broker"], x=s["val"], orientation="h",
            marker_color="#f85149", name="Sell",
            text=[f"{v/1e9:.1f}B" if v >= 1e9 else f"{v/1e6:.0f}jt" for v in s["val"]],
            textposition="outside", textfont=dict(color="#f85149"),
        ), row=1, col=2)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14",
        height=380, showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Consolas, monospace", size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e2530")
    return fig


def build_avg_chart(buy_df, last_price):
    """Scatter avg price top buyer vs harga sekarang."""
    if buy_df is None or "avg" not in buy_df.columns:
        return None
    d = buy_df.nlargest(6, "val")
    d = d[d["avg"] > 0]
    if len(d) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["broker"], y=d["avg"],
        marker_color=["#16c784" if a < last_price else "#f85149" for a in d["avg"]],
        text=[f"{a:,.0f}" for a in d["avg"]], textposition="outside",
        name="Avg Buyer",
    ))
    fig.add_hline(y=last_price, line_dash="dash", line_color="#e8b339",
                  annotation_text=f"Harga {last_price:,.0f}", annotation_position="right")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14",
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        title="AVG PRICE TOP BUYER vs HARGA (hijau=profit/nyaman, merah=nyangkut)",
        font=dict(family="Consolas, monospace", size=11),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1e2530")
    return fig


# ════════════════════════════════════════════════════════════════════
# NARASI AI (Gemini / Ollama) — prompt super lengkap
# ════════════════════════════════════════════════════════════════════
def _enrich_broker_data(buy_df, sell_df, last_price):
    """Siapkan ringkasan data yang kaya buat prompt AI."""
    lines = []
    if buy_df is not None and len(buy_df) > 0:
        tot_b = buy_df["val"].sum()
        for _, r in buy_df.nlargest(7, "val").iterrows():
            pct = r["val"] / tot_b * 100 if tot_b else 0
            avg = r.get("avg", 0)
            pos = ""
            if avg > 0 and last_price > 0:
                d = (avg - last_price) / last_price * 100
                pos = f", avg {avg:,.0f} ({d:+.1f}% vs harga)"
            lines.append(f"    BUY  {r['broker']}: {r['val']/1e9:.1f}B ({pct:.0f}% sisi beli), "
                         f"{r['lot']/1e3:.0f}rb lot{pos}")
    if sell_df is not None and len(sell_df) > 0:
        tot_s = sell_df["val"].sum()
        for _, r in sell_df.nlargest(7, "val").iterrows():
            pct = r["val"] / tot_s * 100 if tot_s else 0
            avg = r.get("avg", 0)
            pos = ""
            if avg > 0 and last_price > 0:
                d = (avg - last_price) / last_price * 100
                pos = f", avg {avg:,.0f} ({d:+.1f}% vs harga)"
            lines.append(f"    SELL {r['broker']}: {r['val']/1e9:.1f}B ({pct:.0f}% sisi jual), "
                         f"{r['lot']/1e3:.0f}rb lot{pos}")
    return "\n".join(lines)


def generate_ai_narrative(ticker, tf, result, buy_df, sell_df, last_price, bandar, running, notes):
    broker_detail = _enrich_broker_data(buy_df, sell_df, last_price)
    tot_buy = buy_df["val"].sum() if buy_df is not None else 0
    tot_sell = sell_df["val"].sum() if sell_df is not None else 0
    net = tot_buy - tot_sell

    prompt = f"""Lo analis flow saham IDX (Indonesia) SENIOR, spesialis metode Wyckoff + bandarmologi (baca pergerakan bandar lewat broker summary). Gaya lo tajam, jujur, gak basa-basi, gak promosi. Pakai Bahasa Indonesia santai gaya 'bro' tapi tetap berbobot.

═══════════════════════════════
DATA SAHAM: {ticker} · Timeframe: {tf}
═══════════════════════════════
Harga sekarang: {last_price:,.0f}
Flow Score (rule-based): {result['score']}/100 → {result['verdict']}
Confidence: {result.get('confidence','-')}

BROKER SUMMARY (sisi beli vs jual, % share, avg vs harga):
{broker_detail}

AGREGAT:
- Total nilai BELI: {tot_buy/1e9:.1f}B
- Total nilai JUAL: {tot_sell/1e9:.1f}B
- Net flow: {net/1e9:+.1f}B ({'beli unggul' if net>0 else 'jual unggul' if net<0 else 'imbang'})
- Big money net: {bandar.get('big_net')} lot | Retail net: {bandar.get('retail_net')} lot
- Running trade: lifting offer {running.get('lifting')} vs hitting bid {running.get('hitting')}
- Catatan trader: {notes or '-'}

═══════════════════════════════
TUGAS LO — tulis analisa SUPER LENGKAP & MENDALAM dengan struktur ini:

## 1. BACA FLOW (siapa main)
Bedah broker per broker. Siapa net buyer dominan & seberapa serius akumulasinya (lihat % share + lot). Siapa yang lepas barang. Apakah barang pindah dari ritel/banyak broker ke sedikit broker besar (tanda akumulasi) atau sebaliknya. Sebut KODE BROKER konkret.

## 2. POSISI BANDAR & AVG
Analisa avg price tiap broker besar vs harga sekarang. Siapa yang avg-nya di bawah (nyaman/profit, kuat nahan) vs di atas (nyangkut, butuh effort jaga harga). Apa artinya buat niat bandar — mau markup, lagi serok, atau jebakan?

## 3. FASE WYCKOFF
Tentukan fase Wyckoff yang paling mungkin (Accumulation A-E / Markup / Distribution / Markdown / Re-accumulation). Jelasin INDIKASI dari data yang dukung tebakan fase ini. Kalau di fase accumulation, sebut sub-fase-nya (PS, SC, AR, ST, Spring, Test, SOS, LPS) kalau datanya cukup buat nebak.

## 4. SKENARIO & LEVEL
Kasih 2 skenario (bullish & bearish): apa trigger-nya, ke mana arah harga, berapa lama biasanya main di fase ini. Sebut level/zona kunci yang harus dipantau (pakai angka harga & avg broker sebagai acuan).

## 5. ACTION PLAN
Rekomendasi sikap konkret: WAIT / akumulasi bertahap / hindari / siap entry. Plus kondisi spesifik yang bikin lo ganti pandangan.

## 6. RISK NOTE
Apa yang bikin tesis ini GUGUR. Sinyal bahaya yang harus diwaspadai (mis. broker akumulator tiba-tiba net sell, harga jebol level X).

PENTING:
- Jujur kalau data ambigu atau confidence rendah — jangan maksa kasih sinyal kuat dari data tipis.
- Ini timeframe {tf}, sesuaikan horizon analisa (panjang = bulanan, jangan kasih level intraday).
- Tutup dengan reminder bahwa ini alat bantu baca flow, keputusan & risk management di tangan trader."""

    return call_ai(prompt, max_tokens=8192)



# ════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════
st.markdown("# 📊 MESIN FLOW READER")
st.caption("Per-saham deep dive · input Stockbit/Neo BDM · scoring + tick card + narasi AI + Telegram")

with st.sidebar:
    st.markdown("### ⚙️ INPUT SAHAM")
    ticker = st.text_input("Kode Saham", value="", placeholder="cth: DSSA",
                           help="Wajib diisi. Otomatis huruf besar.").strip().upper()
    if not ticker:
        st.warning("⬆️ Isi kode saham dulu bro")
        ticker = "—"
    last_price = st.number_input("Harga Sekarang", min_value=0.0, value=0.0, step=5.0,
                                 help="Harga terakhir saham ini")
    tf = st.selectbox("Timeframe", ["Pendek (intraday/daily)", "Sedang (mingguan)", "Panjang (bulanan)"])

    st.markdown("---")
    st.markdown("### 🏦 BANDAR (opsional)")
    st.caption("Skip aja kalau males — dari broker summary udah ketauan bandarnya. "
               "Isi cuma kalau lo punya data Bandar Detector (Stockbit) buat nambah presisi.")
    big_net = st.number_input("Big Money Net (lot, +/−)", value=0.0, step=100.0,
                              help="Dari Bandar Detector Stockbit: Net Volume sisi 'Big Acc'. "
                                   "Positif = big player akumulasi. Kosongin = skip.")
    retail_net = st.number_input("Retail Net (lot, +/−)", value=0.0, step=100.0,
                                 help="Net volume ritel. Positif = ritel beli (sering kontrarian "
                                      "vs bandar). Kosongin = skip.")

    st.markdown("---")
    st.markdown("### 📊 ORDERBOOK (opsional)")
    st.caption("Tekanan agresif dari orderbook/running. Lifting = beli ngejar offer, "
               "Hitting = jual nabrak bid. Skip kalau gak ada datanya.")
    lifting = st.number_input("Lifting Offer (lot, agresif beli)", min_value=0.0, value=0.0, step=10.0,
                              help="Total lot yang lifting offer (beli agresif angkat harga).")
    hitting = st.number_input("Hitting Bid (lot, agresif jual)", min_value=0.0, value=0.0, step=10.0,
                              help="Total lot yang hitting bid (jual agresif tekan harga).")

    st.markdown("---")
    notes = st.text_area("📝 Catatan / konteks", placeholder="cth: lagi di support kuat, abis breakout, dll")

    # ── PILIH AI BACKEND ──
    st.markdown("---")
    st.markdown("### 🤖 MESIN AI (analisa)")

    # diagnostik secrets — bantu debug kenapa masih minta manual
    _gk = _secret("GEMINI_API_KEY")
    _tk = _secret("TELEGRAM_TOKEN")
    _ci = _secret("TELEGRAM_CHAT_ID")
    _placeholder = lambda v: (not v) or v.startswith("PASTE_") or v == ""
    if _gk and not _placeholder(_gk):
        pass  # secrets bener, status detail di bawah
    elif _gk and _placeholder(_gk):
        st.warning("⚠️ `secrets.toml` kebaca tapi GEMINI_API_KEY masih template "
                   "(`PASTE_...`). Edit file & isi key beneran, lalu restart app.")

    ai_backend = st.selectbox(
        "Pilih AI",
        ["Gemini (API)", "GitHub Models (GPT-4o)", "Ollama (lokal, gratis)"],
        key="ai_backend",
        help="Gemini: cepat, key gratis. GitHub Models: GPT-4o gratis pakai GitHub token, "
             "cocok buat cloud. Ollama: offline tapi perlu PC kuat.")

    if ai_backend.startswith("Gemini"):
        _key_from_secret = bool(_secret("GEMINI_API_KEY"))
        if _key_from_secret and not st.session_state.get("gemini_key", "").strip():
            st.caption(f"🔐 Key dari secrets.toml · model: `{GEMINI_MODEL}`")
            with st.expander("Ganti key sementara (opsional)"):
                st.text_input("🔑 Gemini API key", type="password",
                              key="gemini_key", placeholder="kosongin = pakai secrets")
        else:
            st.text_input("🔑 Gemini API key", type="password",
                          key="gemini_key", placeholder="AIzaSy...",
                          help="Gratis dari aistudio.google.com. Atau simpan di secrets.toml.")
            if get_gemini_key():
                st.caption(f"✅ Key kebaca · model: `{GEMINI_MODEL}`")
            else:
                st.caption("⚪ Belum ada key — narasi AI nonaktif (skor & chart tetap jalan)")

    elif ai_backend.startswith("GitHub"):
        _gh_from_secret = bool(_secret("GITHUB_TOKEN"))
        if _gh_from_secret and not st.session_state.get("github_token", "").strip():
            st.caption(f"🔐 GitHub token dari secrets.toml · model: `{GITHUB_MODEL}`")
            with st.expander("Ganti token sementara (opsional)"):
                st.text_input("🔑 GitHub Token (PAT)", type="password",
                              key="github_token", placeholder="kosongin = pakai secrets")
        else:
            st.text_input("🔑 GitHub Token (PAT)", type="password",
                          key="github_token", placeholder="github_pat_... atau ghp_...",
                          help="PAT dari github.com dengan permission models:read. Gratis.")
            if get_github_token():
                st.caption(f"✅ Token kebaca · model: `{GITHUB_MODEL}`")
            else:
                st.caption("⚪ Belum ada token — narasi AI nonaktif")
        with st.expander("📖 Cara dapat GitHub Token (gratis, buat GPT-4o)"):
            st.markdown(
                "1. Buka **github.com/settings/personal-access-tokens**\n"
                "2. Klik **Generate new token** → **Fine-grained token**\n"
                "3. Kasih nama, set **Expiration** (cth 90 hari)\n"
                "4. Di **Permissions** → **Account permissions** → cari **Models** → "
                "set ke **Read-only**\n"
                "5. **Generate token** → copy (mulai `github_pat_...`)\n"
                "6. Paste di atas atau simpan di secrets.toml\n\n"
                "✅ GPT-4o gratis (limit ~50 request/hari). Cocok buat deploy cloud "
                "karena gak butuh PC kuat kayak Ollama.")

    else:  # Ollama
        st.caption(f"🖥️ Pakai Ollama lokal · model: `{OLLAMA_MODEL}`")
        with st.expander("📖 Cara setup Ollama (sekali aja)"):
            st.markdown(
                "1. Download & install **Ollama** dari ollama.com\n"
                "2. Buka terminal, jalankan: `ollama pull llama3.1`\n"
                "3. Pastikan Ollama jalan (biasanya auto-start)\n"
                "4. Pilih Ollama di dropdown ini — langsung jalan, gratis & offline\n\n"
                "⚠️ Ollama jalan LOKAL — gak bisa dipakai kalau app di-deploy ke cloud "
                "(Streamlit Cloud gak punya akses ke PC lo). Buat cloud, pakai Gemini "
                "atau GitHub Models.")

    # ── TELEGRAM ──
    st.markdown("---")
    st.markdown("### 📤 TELEGRAM")
    _tg_from_secret = bool(_secret("TELEGRAM_TOKEN")) and bool(_secret("TELEGRAM_CHAT_ID"))
    if _tg_from_secret and not st.session_state.get("tg_token", "").strip():
        st.caption("🔐 Token + Chat ID dari secrets.toml")
        with st.expander("Ganti sementara (opsional)"):
            st.text_input("Bot Token", type="password", key="tg_token",
                          placeholder="kosongin = pakai secrets")
            st.text_input("Chat ID", key="tg_chat", placeholder="kosongin = pakai secrets")
    else:
        st.text_input("Bot Token", type="password", key="tg_token", placeholder="123456:ABC-...",
                      help="Bikin bot via @BotFather. Atau simpan di secrets.toml.")
        st.text_input("Chat ID", key="tg_chat", placeholder="-100123... atau 123456",
                      help="ID grup/channel/personal. Cari via @getidsbot.")

    _tgt, _tgc = get_telegram_config()
    if _tgt and _tgc:
        st.caption("✅ Telegram siap")
        if st.button("🔌 Test koneksi Telegram", use_container_width=True):
            with st.spinner("Ngetes kirim ke Telegram..."):
                ok, info = telegram_test()
            if ok:
                st.success(info)
            else:
                st.error(info)
    else:
        st.caption("⚪ Isi token + chat ID buat ngaktifin kirim")
    with st.expander("📖 Cara setup Telegram (PENTING — baca kalau gagal)"):
        st.markdown(
            "**Langkah wajib (urutan penting):**\n\n"
            "1. Chat **@BotFather** → ketik `/newbot` → ikutin → dapet **Bot Token** "
            "(bentuknya `123456789:ABCdef...`)\n\n"
            "2. ⚠️ **CHAT BOT LO DULUAN!** Cari nama bot yang barusan lo bikin di Telegram, "
            "klik **START** atau kirim 'halo'. **Tanpa ini, bot GAK BISA kirim ke lo** "
            "(ini penyebab #1 'sukses tapi gak nyampe').\n\n"
            "3. Buat **Chat ID**: chat **@getidsbot** → dia kasih ID lo (angka, cth `123456789`)\n\n"
            "4. Kalau mau kirim ke **grup**: add bot ke grup dulu, ID grup diawali `-100`\n\n"
            "5. Paste Token + Chat ID di atas → klik **🔌 Test koneksi**. "
            "Kalau pesan tes masuk = beres.\n\n"
            "💡 **Biar gak input ulang tiap buka app**: simpan di file "
            "`.streamlit/secrets.toml` (ada template-nya di folder app).")


def build_df_from_table(df_edit):
    """Dari data editor (kolom: Broker, Val, Lot, Freq, Avg) → df standar."""
    if df_edit is None or len(df_edit) == 0:
        return None
    recs = []
    for _, r in df_edit.iterrows():
        broker = str(r.get("Broker", "")).strip().upper()
        if not broker or broker.lower() == "nan":
            continue
        val = to_num(r.get("Val", 0))
        lot = to_num(r.get("Lot", 0))
        avg = to_num(r.get("Avg", 0))
        freq = to_num(r.get("Freq", 0))
        if val == 0 and lot == 0:
            continue
        recs.append({"broker": broker, "lot": lot, "val": val, "avg": avg, "freq": freq})
    if not recs:
        return None
    return pd.DataFrame(recs)


def parse_stockbit_paste(text, side="buy"):
    """
    Parse paste langsung dari Stockbit broker summary.
    Format kolom Stockbit: KODE  Val  Lot  Freq  Avg
    (Val pakai B/M, Lot pakai K/M). Return df buat ngisi tabel.
    """
    recs = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        broker = parts[0].strip().upper()
        # skip header
        if broker in ("BUY", "SELL", "B.VAL", "S.VAL", "BROKER"):
            continue
        val = to_num(parts[1]) if len(parts) > 1 else 0
        lot = to_num(parts[2]) if len(parts) > 2 else 0
        freq = to_num(parts[3]) if len(parts) > 3 else 0
        avg = to_num(parts[4]) if len(parts) > 4 else 0
        recs.append({"Broker": broker, "Val": val, "Lot": lot, "Freq": freq, "Avg": avg})
    return pd.DataFrame(recs) if recs else None


# kolom default kosong buat data editor
EMPTY_TABLE = pd.DataFrame([{"Broker": "", "Val": 0.0, "Lot": 0.0, "Freq": 0.0, "Avg": 0.0}
                            for _ in range(8)])

COL_CONFIG = {
    "Broker": st.column_config.TextColumn("Broker", width="small", help="Kode broker, cth: LG"),
    "Val": st.column_config.NumberColumn("Val (Rp)", help="Buy/Sell Value. Boleh ketik 56.2B / 249.8B"),
    "Lot": st.column_config.NumberColumn("Lot", help="Buy/Sell Lot"),
    "Freq": st.column_config.NumberColumn("Freq", help="Frekuensi (opsional)"),
    "Avg": st.column_config.NumberColumn("Avg", help="Harga rata-rata"),
}


_main = st.container()

# ════════════════════════════════════════════════════════════════════
# HALAMAN UTAMA — ANALISA PER SAHAM
# ════════════════════════════════════════════════════════════════════
with _main:
    st.markdown(f"### 📋 {ticker} — BROKER SUMMARY")
    st.caption("Input via tabel (kolom samain Stockbit: Val · Lot · Freq · Avg). "
               "Val boleh ketik `249.8B` / `56.2M`. Atau pakai paste-helper di bawah buat isi cepat.")

    # ── paste helper (opsional) ──
    with st.expander("📋 Paste cepat dari Stockbit (auto-isi tabel)", expanded=False):
        st.caption("Copy baris broker dari Stockbit (KODE Val Lot Freq Avg), paste di sini, "
                   "lalu klik tombol. Tabel di bawah otomatis keisi.")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            paste_buy = st.text_area("Paste BUY side", height=120, key="paste_buy",
                                     placeholder="SS\t249.8B\t3M\t5846\t847\nRF\t33.7B\t404.5K\t741\t847")
            if st.button("➡️ Isi tabel BUY", key="fill_buy"):
                pb = parse_stockbit_paste(paste_buy, "buy")
                if pb is not None:
                    st.session_state["buy_table"] = pb
                    st.rerun()
        with pcol2:
            paste_sell = st.text_area("Paste SELL side", height=120, key="paste_sell",
                                      placeholder="AK\t67.8B\t805.7K\t8006\t846\nBK\t43B\t505.2K\t2017\t851")
            if st.button("➡️ Isi tabel SELL", key="fill_sell"):
                ps = parse_stockbit_paste(paste_sell, "sell")
                if ps is not None:
                    st.session_state["sell_table"] = ps
                    st.rerun()

    # ── tabel input ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🟢 NET BUYER**")
        buy_init = st.session_state.get("buy_table", EMPTY_TABLE.copy())
        buy_edit = st.data_editor(buy_init, num_rows="dynamic", use_container_width=True,
                                  column_config=COL_CONFIG, hide_index=True, key="ed_buy")
    with c2:
        st.markdown("**🔴 NET SELLER**")
        sell_init = st.session_state.get("sell_table", EMPTY_TABLE.copy())
        sell_edit = st.data_editor(sell_init, num_rows="dynamic", use_container_width=True,
                                   column_config=COL_CONFIG, hide_index=True, key="ed_sell")

    run = st.button("🔍 ANALISA FLOW", type="primary", use_container_width=True)

    if run:
        buy_df = build_df_from_table(buy_edit)
        sell_df = build_df_from_table(sell_edit)
        if buy_df is None and sell_df is None:
            st.error("Minimal isi salah satu: Net Buyer atau Net Seller.")
            st.stop()
        bandar = {"big_net": big_net, "retail_net": retail_net}
        running = {"lifting": lifting, "hitting": hitting}
        result = compute_flow_score(buy_df, sell_df, last_price, running, bandar)
        # SIMPAN semua hasil ke session_state biar gak hilang pas rerun
        # (rerun terjadi tiap pencet toggle/tombol — ini kunci fix bug kirim & toggle)
        st.session_state["analysis"] = {
            "ticker": ticker, "tf": tf, "last_price": last_price,
            "buy_df": buy_df, "sell_df": sell_df, "bandar": bandar,
            "running": running, "notes": notes, "result": result,
        }
        # reset narasi lama kalau analisa baru
        st.session_state.pop("narr_text", None)
        st.session_state.pop("narr_key", None)

    # ── RENDER HASIL DARI SESSION_STATE (persist walau rerun) ──
    A = st.session_state.get("analysis")
    if A:
        ticker = A["ticker"]; tf = A["tf"]; last_price = A["last_price"]
        buy_df = A["buy_df"]; sell_df = A["sell_df"]
        bandar = A["bandar"]; running = A["running"]; notes = A["notes"]
        result = A["result"]
        card = get_tick_card(result)

        # ── TICK CARD (status warna) ──
        st.markdown(f"## {ticker} · {tf}")
        glow = "box-shadow:0 0 24px #ffd70066;" if card["tier"] == "GOLDEN" else ""
        st.markdown(
            f"<div style='background:{card['bg']};border:2px solid {card['color']};"
            f"border-radius:12px;padding:18px 22px;margin:6px 0;{glow}'>"
            f"<div style='font-size:34px;font-weight:800;color:{card['color']};"
            f"font-family:Consolas,monospace;letter-spacing:1px;'>"
            f"{card['emoji']} {card['label']}</div>"
            f"<div style='color:#cbd5e1;font-size:14px;margin-top:6px;'>{card['desc']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── HEADER VERDICT ──
        conf_color = {"TINGGI": "#16c784", "SEDANG": "#e8b339"}.get(result["confidence"], "#f0883e")
        st.markdown(
            f"<div class='verdict' style='background:{result['color']}22;border:1px solid {result['color']};'>"
            f"<span class='tag' style='background:{result['color']};color:#0a0e14;'>{result['tag']}</span>"
            f"<b style='color:{result['color']};font-size:22px;'> {result['verdict']}</b><br>"
            f"<span style='color:#9ca3af;'>Flow Score: </span>"
            f"<b style='color:{result['color']};font-size:30px;'>{result['score']}</b>"
            f"<span style='color:#6b7280;'>/100</span>"
            f"<span style='float:right;color:{conf_color};font-size:13px;'>"
            f"Confidence: <b>{result['confidence']}</b> · {result['valid_components']}/4 komponen</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── METRIC BOXES ──
        cols = st.columns(4)
        metrics_def = [
            ("Konsentrasi Buyer", "{:.0%}"),
            ("Avg Buyer vs Harga", "{:,.0f}"),
            ("Big vs Retail", "{:+,.0f}"),
            ("Running Pressure", "{:+.0%}"),
        ]
        for col, (name, fmt) in zip(cols, metrics_def):
            raw, sc = result["breakdown"].get(name, (0, 0))
            sc_color = "#16c784" if sc > 0 else "#f85149" if sc < 0 else "#6b7280"
            try:
                disp = fmt.format(raw)
            except Exception:
                disp = str(raw)
            col.markdown(
                f"<div class='metric-box'><div class='lbl'>{name}</div>"
                f"<div class='val' style='color:{sc_color};'>{disp}</div>"
                f"<div style='color:{sc_color};font-size:12px;'>{sc:+.1f} pts</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SUMMARY SPESIFIK (siapa terkuat + arah) ──
        st.markdown("### 📌 SUMMARY FLOW")
        summary = build_flow_summary(result, buy_df, sell_df, last_price, bandar, ticker)
        for label, text, color in summary:
            is_arah = label.startswith("🎯")
            bg = f"{color}1a" if is_arah else "#11161f"
            border = color if is_arah else "#1e2530"
            st.markdown(
                f"<div style='background:{bg};border-left:3px solid {border};"
                f"border-radius:4px;padding:10px 14px;margin:6px 0;'>"
                f"<span style='color:{color};font-size:12px;letter-spacing:0.5px;font-weight:600;'>{label}</span><br>"
                f"<span style='color:#d4d4d4;font-size:14px;'>{text}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CHART ──
        chart_fig = build_chart(buy_df, sell_df, last_price, ticker)
        st.plotly_chart(chart_fig, use_container_width=True)

        avg_fig = build_avg_chart(buy_df, last_price)
        if avg_fig:
            st.plotly_chart(avg_fig, use_container_width=True)

        # ── TABEL ──
        cc1, cc2 = st.columns(2)
        with cc1:
            if buy_df is not None:
                st.markdown("**🟢 Net Buyer**")
                show = buy_df.nlargest(10, "val").copy()
                show["val"] = show["val"].map(lambda v: f"{v/1e9:.2f}B" if v >= 1e9 else f"{v/1e6:.1f}jt")
                st.dataframe(show, use_container_width=True, hide_index=True)
        with cc2:
            if sell_df is not None:
                st.markdown("**🔴 Net Seller**")
                show = sell_df.nlargest(10, "val").copy()
                show["val"] = show["val"].map(lambda v: f"{v/1e9:.2f}B" if v >= 1e9 else f"{v/1e6:.1f}jt")
                st.dataframe(show, use_container_width=True, hide_index=True)

        # ── NARASI AI ──
        _backend = st.session_state.get("ai_backend", "Gemini (API)")
        if _backend.startswith("Ollama"):
            _bname = "Ollama"
        elif _backend.startswith("GitHub"):
            _bname = "GPT-4o"
        else:
            _bname = "Gemini"
        st.markdown(f"### 🤖 ANALISA AI ({_bname})")
        narrative = None
        if not ai_ready():
            st.info("💡 Narasi AI nonaktif. Isi key/token AI di sidebar (Gemini / GitHub / Ollama). "
                    "Skor + summary + chart tetap jalan tanpa ini.")
        else:
            cache_key = f"{ticker}|{last_price}|{result['score']}|{_backend}"
            if st.session_state.get("narr_key") == cache_key and \
               st.session_state.get("narr_text") and \
               not st.session_state.get("narr_text", "").startswith("⚠️"):
                narrative = st.session_state.get("narr_text")
            if not narrative:
                with st.spinner(f"{_bname} lagi bedah flow-nya (full Wyckoff)..."):
                    narrative = generate_ai_narrative(ticker, tf, result, buy_df, sell_df,
                                                      last_price, bandar, running, notes)
                st.session_state["narr_key"] = cache_key
                st.session_state["narr_text"] = narrative

            if narrative in ("⚠️ GEMINI_BUSY", "⚠️ GITHUB_BUSY"):
                src = "Gemini" if "GEMINI" in narrative else "GitHub Models"
                st.warning(f"⏳ **Server {src} lagi sibuk** (high demand). Udah dicoba "
                           "beberapa kali tapi masih penuh. Ini sementara.")
                cbtn1, cbtn2 = st.columns(2)
                with cbtn1:
                    if st.button("🔄 Coba lagi", use_container_width=True):
                        st.session_state.pop("narr_text", None)
                        st.rerun()
                with cbtn2:
                    st.caption("Atau ganti backend AI di sidebar (Gemini ↔ GitHub ↔ Ollama).")
                narrative = None
            elif narrative and narrative.startswith("⚠️"):
                st.warning(narrative)
                narrative = None
            elif narrative:
                st.markdown(narrative)

        # ── KIRIM KE TELEGRAM ──
        st.markdown("---")
        st.markdown("### 📤 KIRIM KE TELEGRAM")
        tg_token, tg_chat = get_telegram_config()
        if not tg_token or not tg_chat:
            st.info("💡 Isi **Bot Token + Chat ID** di sidebar (bagian 📤 Telegram) "
                    "lalu klik 🔌 Test koneksi dulu sampai ✅.")
        else:
            fmt_choice = st.radio(
                "Format kiriman", ["🖼️ Infografis PNG", "📄 Infografis PDF", "📝 Teks biasa"],
                horizontal=True, key="tg_fmt")

            # preview infografis (PNG) di app sebelum kirim — Pillow, selalu jalan
            if fmt_choice.startswith("🖼️") or fmt_choice.startswith("📄"):
                with st.spinner("Render infografis..."):
                    try:
                        preview = render_infographic_pil(ticker, tf, result, card, summary,
                                                         buy_df, sell_df, last_price, narrative)
                    except Exception as e:
                        preview = None
                        st.error(f"Gagal render infografis: {e}")
                if preview:
                    st.image(preview, caption="Preview infografis", use_container_width=True)
                    st.download_button("⬇️ Download PNG", preview,
                                       file_name=f"{ticker}_flow.png", mime="image/png")

            if st.button("📤 Kirim ke Telegram", use_container_width=True, type="primary"):
                with st.spinner("Lagi kirim ke Telegram..."):
                    if fmt_choice.startswith("📝"):
                        msg = build_telegram_message(ticker, tf, result, card, summary,
                                                     buy_df, sell_df, last_price, narrative)
                        png = chart_to_png(chart_fig)
                        cap_short = f"{card['emoji']} {ticker} · {tf} · Score {result['score']}/100"
                        ok, info = send_telegram(msg, photo_bytes=png, caption_short=cap_short)
                    else:
                        is_pdf = fmt_choice.startswith("📄")
                        try:
                            png = render_infographic_pil(ticker, tf, result, card, summary,
                                                        buy_df, sell_df, last_price, narrative)
                        except Exception as e:
                            png = None
                            info_err = str(e)
                        if png is None:
                            ok, info = False, "Gagal render infografis. Coba format Teks biasa."
                        else:
                            cap = f"{card['emoji']} {ticker} · {tf} · Flow Score {result['score']}/100 · {card['label']}"
                            if is_pdf:
                                pdf = png_to_pdf(png)
                                if pdf is None:
                                    ok, info = False, "Gagal konversi ke PDF. Pakai PNG aja."
                                else:
                                    ok, info = send_telegram_document(
                                        pdf, f"{ticker}_flow.pdf", caption=cap, as_photo=False)
                            else:
                                ok, info = send_telegram_document(
                                    png, f"{ticker}_flow.png", caption=cap, as_photo=True)
                if ok:
                    st.success(f"✅ {info}")
                else:
                    st.error(f"❌ Gagal: {info}")

        st.caption("⚠️ Tool bantu baca flow, BUKAN sinyal beli/jual. Validasi sendiri + manajemen risiko.")
