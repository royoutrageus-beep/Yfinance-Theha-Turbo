# ⚡ Flow Reader

Tool baca **flow / akumulasi-distribusi bandar** saham IDX dari broker summary (Stockbit / Neo BDM). Scoring rule-based + narasi AI (Wyckoff & bandarmologi) + infografis + kirim ke Telegram.

## Fitur
- **Flow Score 0–100** dari konsentrasi buyer, avg price, big/retail, running pressure
- **Tick Card**: 🟢 Green / 🔴 Red / ⚪ Grey / 🏆 Golden (bagger)
- **Narasi AI** full Wyckoff — pilih **Gemini**, **GitHub Models (GPT-4o)**, atau **Ollama** (lokal)
- **Infografis** 1 halaman (PNG/PDF) — gak butuh Chrome/wkhtml, murni Pillow
- **Kirim ke Telegram** (summary + chart + data broker)

## Cara Jalanin (Lokal)
```bash
pip install -r requirements.txt
streamlit run flow_reader.py
```

## Setup Kredensial
Copy `.streamlit/secrets.toml.example` jadi `.streamlit/secrets.toml`, lalu isi:
```toml
GEMINI_API_KEY   = "..."   # dari aistudio.google.com (narasi AI)
GITHUB_TOKEN     = "..."   # PAT dengan permission Models:read (GPT-4o)
TELEGRAM_TOKEN   = "..."   # dari @BotFather
TELEGRAM_CHAT_ID = "..."   # dari @getidsbot
```
> ⚠️ **Restart app** setelah edit secrets — Streamlit cuma baca pas start.

Yang gak dipakai, biarin `PASTE_...` (otomatis diabaikan). Skor + chart tetap jalan walau tanpa AI.

## Deploy ke Streamlit Cloud
1. Push repo ini ke GitHub (file `secrets.toml` TIDAK ikut, sudah di-`.gitignore`)
2. Buka share.streamlit.io → connect repo → pilih `flow_reader.py`
3. Di **Settings → Secrets**, paste isi secrets (format sama kayak `secrets.toml`)
4. Buat AI di cloud, pakai **Gemini** atau **GitHub Models** (Ollama gak bisa — itu lokal)

## ⚠️ Disclaimer
Alat bantu baca flow, **BUKAN** sinyal beli/jual. Validasi sendiri + manajemen risiko.
