import yfinance as yf
import pandas as pd
import streamlit as st
import time
import requests
import numpy as np
import pytz
from datetime import datetime

# ════════════════════════════════════════════════════
#  ANTI-BLACKLIST & SESSION CACHE CONFIG
# ════════════════════════════════════════════════════
import requests_cache
from requests import Session

def get_safe_session():
    """
    Membuat session aman dengan header browser asli dan caching lokal 2 menit
    agar tidak terkena limit / ban (HTTP 429) dari Yahoo Finance.
    """
    # Cache lokal selama 120 detik untuk menghemat hit jika Streamlit ke-refresh otomatis
    session = requests_cache.CachedSession('theta_yfinance_cache', expire_after=120)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
        'Origin': 'https://finance.yahoo.com',
        'Referer': 'https://finance.yahoo.com/'
    })
    return session

safe_session = get_safe_session()

# ════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════
TOKEN      = st.secrets.get("TELEGRAM_TOKEN", "")
CHAT_ID    = st.secrets.get("TELEGRAM_CHAT_ID", "")
jakarta_tz = pytz.timezone('Asia/Jakarta')

for _k, _v in [("tt_last_sent", set()), ("wl_results", []),
                ("wl_mode_used", ""), ("scan_results", []),
                ("data_dict", {}), ("last_scan_time", None),
                ("last_scan_mode", "Scalping ⚡"),
                ("active_scan_mode", "Scalping ⚡"),
                ("active_auto_regime", True)]:
    if _k not in st.session_state: st.session_state[_k] = _v

st.set_page_config(layout="wide", page_title="Theta Turbo v5.2", page_icon="🔥",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
:root {
    --bg:#080c10; --surface:#0d1117; --border:#1c2533;
    --accent:#00e5ff; --green:#00ff88; --red:#ff3d5a;
    --amber:#ffb700; --purple:#bf5fff; --orange:#ff7b00;
    --muted:#4a5568; --text:#c9d1d9; --heading:#e6edf3;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'Syne',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebar"]{display:none!important;}
[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:8px!important;margin-bottom:12px!important;}
[data-testid="stExpander"] summary{font-family:'Space Mono',monospace!important;font-size:12px!important;color:var(--accent)!important;letter-spacing:1px!important;}
.settings-label{font-family:'Space Mono',monospace;font-size:10px;color:var(--muted);letter-spacing:2px;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border);}
.tt-header{display:flex;align-items:center;padding:16px 0 12px 0;border-bottom:1px solid var(--border);margin-bottom:16px;}
.tt-logo{font-family:'Space Mono',monospace;font-size:22px;font-weight:700;color:var(--orange);letter-spacing:-1px;}
.tt-sub{font-size:11px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;}
.live-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.3);border-radius:20px;font-family:'Space Mono',monospace;font-size:10px;color:var(--accent);letter-spacing:1px;margin-left:auto;}
.live-dot{width:6px;height:6px;background:var(--green);border-radius:50%;animation:blink 1s infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.2;}}
.metric-row{display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap;}
.metric-card{flex:1;min-width:110px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px;position:relative;overflow:hidden;}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent);}
.metric-card.green::before{background:var(--green);}
.metric-card.red::before{background:var(--red);}
.metric-card.amber::before{background:var(--amber);}
.metric-card.orange::before{background:var(--orange);}
.metric-card.purple::before{background:var(--purple);}
.metric-label{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}
.metric-value{font-family:'Space Mono',monospace;font-size:24px;font-weight:700;color:var(--heading);line-height:1;}
.metric-sub{font-size:10px;color:var(--muted);margin-top:3px;}
.signal-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-bottom:20px;}
.signal-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;position:relative;overflow:hidden;transition:border-color .2s;}
.signal-card.gacor{border-color:rgba(0,255,136,.4);background:rgba(0,255,136,.03);}
.signal-card.potensial{border-color:rgba(255,183,0,.3);background:rgba(255,183,0,.03);}
.signal-card.watch{border-color:rgba(0,229,255,.2);}
.signal-card.bagger{border-color:rgba(191,95,255,.6);background:rgba(191,95,255,.05);box-shadow:0 0 20px rgba(191,95,255,.15);}
.signal-card::after{content:'';position:absolute;top:0;left:0;width:4px;height:100%;}
.signal-card.gacor::after{background:var(--green);}
.signal-card.potensial::after{background:var(--amber);}
.signal-card.watch::after{background:var(--accent);}
.signal-card.bagger::after{background:var(--purple);}
.sc-ticker{font-family:'Space Mono',monospace;font-size:18px;font-weight:700;color:var(--heading);}
.sc-price{font-family:'Space Mono',monospace;font-size:13px;color:var(--muted);}
.sc-signal{font-size:13px;font-weight:700;margin:6px 0;}
.sc-bars{display:flex;gap:3px;margin:8px 0;}
.sc-bar{height:16px;border-radius:2px;}
.sc-bar.filled{background:var(--green);}
.sc-bar.filled-purple{background:var(--purple);}
.sc-bar.empty{background:var(--border);}
.sc-stats{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;}
.sc-stat{font-family:'Space Mono',monospace;font-size:10px;color:var(--muted);}
.sc-stat span{color:var(--text);}
.alert-box{background:rgba(255,61,90,.06);border:1px solid rgba(255,61,90,.4);border-radius:8px;padding:14px 18px;margin-bottom:16px;animation:pulse-border 2s infinite;}
.bagger-alert-box{background:rgba(191,95,255,.06);border:1px solid rgba(191,95,255,.5);border-radius:8px;padding:14px 18px;margin-bottom:16px;animation:pulse-purple 2s infinite;}
@keyframes pulse-border{0%,100%{border-color:rgba(255,61,90,.4);}50%{border-color:rgba(255,61,90,.9);}}
@keyframes pulse-purple{0%,100%{border-color:rgba(191,95,255,.4);}50%{border-color:rgba(191,95,255,.9);}}
.alert-title{color:var(--red);font-family:'Space Mono',monospace;font-size:12px;font-weight:700;letter-spacing:2px;}
.bagger-title{color:var(--purple);font-family:'Space Mono',monospace;font-size:12px;font-weight:700;letter-spacing:2px;}
.tape-wrap{overflow:hidden;white-space:nowrap;border-top:1px solid var(--border);border-bottom:1px solid var(--border);padding:5px 0;margin-bottom:16px;background:var(--surface);}
.tape-inner{display:inline-block;animation:marquee 35s linear infinite;}
@keyframes marquee{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.tape-item{display:inline-block;margin:0 18px;font-family:'Space Mono',monospace;font-size:10px;}
.tape-item.up{color:var(--green);}.tape-item.down{color:var(--red);}.tape-item.flat{color:var(--muted);}.tape-item.bagger{color:var(--purple);}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px!important;}
[data-testid="stDataFrame"] thead th{background:var(--surface)!important;color:var(--muted)!important;font-family:'Space Mono',monospace!important;font-size:11px!important;letter-spacing:1px!important;text-transform:uppercase!important;}
::-webkit-scrollbar{width:4px;height:4px;}::-webkit-scrollbar-track{background:var(--bg);}::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
[data-testid="stNumberInput"] input{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--heading)!important;font-family:'Space Mono',monospace!important;border-radius:6px!important;}
button[data-testid="baseButton-primary"]{background:var(--orange)!important;color:var(--bg)!important;font-family:'Space Mono',monospace!important;font-weight:700!important;border:none!important;}
.section-title{font-family:'Space Mono',monospace;font-size:11px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;border-left:3px solid var(--orange);padding-left:10px;margin:20px 0 10px 0;}
.bt-result{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:20px;margin-top:12px;}
.bt-metric{display:inline-block;margin-right:24px;margin-bottom:8px;}
.bt-metric-val{font-family:'Space Mono',monospace;font-size:22px;font-weight:700;}
.bt-metric-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}
@media(max-width:768px){.main .block-container{padding-left:.75rem!important;padding-right:.75rem!important;}.signal-grid{grid-template-columns:1fr;}}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
#  STOCK LIST — FULL IDX (900+ emiten)
# ════════════════════════════════════════════════════
raw_stocks = [
    "AADI","AALI","ABBA","ABDA","ABMM","ACES","ACRO","ACST","ADCP","ADES",
    "ADHI","ADMF","ADMG","ADMR","ADRO","AEGS","AGAR","AGII","AGRO","AGRS",
    "AHAP","AIMS","AISA","AKKU","AKPI","AKRA","AKSI","ALDO","ALII","ALKA",
    "ALMI","ALTO","AMAG","AMAN","AMAR","AMFG","AMIN","AMMN","AMMS","AMOR",
    "AMRT","ANDI","ANJT","ANTM","APEX","APIC","APII","APLI","APLN","ARCI",
    "AREA","ARGO","ARII","ARKA","ARKO","ARMY","ARNA","ARTA","ARTI","ARTO",
    "ASBI","ASDM","ASGR","ASHA","ASII","ASJT","ASLI","ASLC","ASMI","ASPI",
    "ASPR","ASRI","ASRM","ASSA","ATAP","ATIC","ATLA","AUTO","AVIA","AWAN",
    "AXIO","AYAM","AYLS","BABA","BABP","BABY","BACA","BAIK","BAJA","BALI",
    "BANK","BAPA","BAPI","BATA","BATR","BAUT","BAYU","BBCA","BBHI","BBKP",
    "BBLD","BBMD","BBNI","BBRI","BBRM","BBSI","BBSS","BBTN","BBYB","BCAP",
    "BCIC","BCIP","BDKR","BDMN","BEBS","BEEF","BEER","BEKS","BELI","BELL",
    "BESS","BEST","BFIN","BGTG","BHAT","BHIT","BIAS","BIKA","BIKE","BIMA",
    "BINA","BINO","BIPI","BIPP","BIRD","BISI","BIWA","BJBR","BJTM","BKDP",
    "BKSL","BKSW","BLES","BLOG","BLTA","BLTZ","BLUE","BMAS","BMBL","BMHS",
    "BMRI","BMSR","BMTR","BNBA","BNBR","BNGA","BNII","BNLI","BOAT","BOBA",
    "BOGA","BOLA","BOLT","BOSS","BPFI","BPII","BPTR","BRAM","BREN","BRIS",
    "BRMS","BRNA","BRPT","BRRC","BSBK","BSDE","BSIM","BSML","BSSR","BSWD",
    "BTEK","BTEL","BTON","BTPN","BTPS","BUAH","BUDI","BUKA","BUKK","BULL",
    "BUMI","BUVA","BVIC","BWPT","BYAN","CAKK","CAMP","CANI","CARE","CARS",
    "CASA","CASH","CASS","CBDK","CBPE","CBRE","CBUT","CBMF","CCSI","CDIA",
    "CEKA","CENT","CFIN","CGAS","CHEK","CHEM","CHIP","CINT","CITA","CITY",
    "CLAY","CLEO","CLPI","CMNP","CMNT","CMPP","CMRY","CNKO","CNMA","CNTX",
    "COAL","COCO","COIN","COWL","CPIN","CPRI","CPRO","CRAB","CRSN","CSAP",
    "CSIS","CSMI","CSRA","CTBN","CTRA","CTTH","CUAN","CYBR","DAAZ","DADA",
    "DART","DATA","DAYA","DCII","DEAL","DEFI","DEPO","DEWA","DEWI","DFAM",
    "DGNS","DGWG","DGIK","DIGI","DILD","DIVA","DKFT","DKHH","DLTA","DMAS",
    "DMMX","DMND","DNAR","DNET","DOID","DOOH","DOSS","DPNS","DPUM","DRMA",
    "DSFI","DSNG","DSSA","DUCK","DUTI","DVLA","DWGL","DYAN","EAST","ECII",
    "EDGE","EKAD","ELIT","ELPI","ELSA","ELTY","EMAS","EMDE","EMTK","ENAK",
    "ENRG","ENVY","ENZO","EPAC","EPMT","ERAL","ERAA","ERTX","ESIP","ESSA",
    "ESTA","ESTI","ETWA","EURO","EXCL","FAPA","FAST","FASW","FILM","FIMP",
    "FIRE","FISH","FITT","FLMC","FOLK","FOOD","FORE","FORU","FPNI","FUJI",
    "FUTR","FWCT","GAMA","GDST","GDYR","GEMA","GEMS","GGRP","GGRM","GHON",
    "GIAA","GJTL","GLOB","GLVA","GMFI","GMTD","GOLF","GOLD","GOLL","GOOD",
    "GOTO","GPRA","GPSO","GRIA","GRPH","GRPM","GRII","GSMF","GTBO","GTRA",
    "GTSI","GULA","GUNA","GWSA","GZCO","HADE","HAIS","HAJJ","HALO","HATM",
    "HBAT","HDFA","HDIT","HEAL","HELI","HERO","HEXA","HGII","HILL","HITS",
    "HKMU","HMSP","HOKI","HOME","HOMI","HOPE","HOTL","HRME","HRTA","HRUM",
    "HUMI","HYGN","IATA","IBFN","IBOS","IBST","ICBP","ICON","IDEA","IDPR",
    "IFII","IFSH","IGAR","IIKP","IKAI","IKAN","IKBI","IKPM","IMAS","IMJS",
    "IMPC","INAF","INAI","INCF","INCI","INCO","INDF","INDO","INDR","INDS",
    "INDX","INDY","INET","INKP","INOV","INPC","INPP","INPS","INRU","INTA",
    "INTD","INTP","IOTF","IPAC","IPCC","IPCM","IPOL","IPPE","IPTV","IRRA",
    "IRSX","ISAP","ISAT","ISEA","ISSP","ITIC","ITMA","ITMG","JAAS","JARR",
    "JAST","JATI","JAVA","JAYA","JECC","JGLE","JIHD","JKON","JMAS","JPFA",
    "JRPT","JSKY","JSMR","JSPT","JTPE","KAEF","KAQI","KARW","KARY","KAST",
    "KAYU","KBAG","KBLI","KBLM","KBLV","KBRI","KDSI","KDTN","KEEN","KEJU",
    "KETR","KIAS","KICI","KIJA","KING","KINO","KIOS","KJEN","KKES","KKGI",
    "KLAS","KLBF","KLIN","KMDS","KMTR","KOBX","KOCI","KOIN","KOKA","KONI",
    "KOPI","KOTA","KPIG","KRAH","KRAS","KREN","KSIX","KUAS","LABA","LABS",
    "LAJU","LAND","LAPD","LCGP","LCKM","LEAD","LFLO","LIFE","LINK","LION",
    "LIVE","LMAS","LMPI","LMSH","LOPI","LPCK","LPGI","LPIN","LPKR","LPLI",
    "LPPF","LPPS","LRNA","LSIP","LTLS","LUCK","LUCY","MAAS","MABA","MADA",
    "MAGP","MAHA","MAIN","MANG","MAPA","MAPB","MAPI","MARI","MARK","MASA",
    "MASB","MAYA","MBAP","MBMA","MBSS","MBTO","MCAS","MCOL","MCOR","MDIA",
    "MDKA","MDKI","MDLA","MDLN","MDRN","MEDC","MEDS","MEGA","MEJA","MENN",
    "MERI","MERK","META","MFMI","MGNA","MGRO","MHKI","MICE","MIDI","MIKA",
    "MINA","MINE","MIRA","MITI","MKAP","MKPI","MKTR","MLBI","MLIA","MLPL",
    "MLPT","MMLP","MMIX","MNCN","MOLI","MORA","MPOW","MPMX","MPPA","MPRO",
    "MPXL","MRAT","MREI","MSIE","MSIN","MSJA","MSKY","MSTI","MTDL","MTEL",
    "MTFN","MTLA","MTMH","MTPS","MTRA","MTRN","MTSM","MTWI","MUTU","MYOH",
    "MYOR","MYTX","NAIK","NANO","NASA","NASI","NATO","NAYZ","NCKL","NELY",
    "NEST","NETV","NICE","NICK","NICL","NIKL","NINE","NIRO","NISP","NOBU",
    "NPGF","NRCA","NSSS","NTBK","NUSA","NZIA","OASA","OBAT","OBMD","OCAP",
    "OILS","OKAS","OLIV","OMED","OMRE","OPMS","PACK","PADA","PADI","PALM",
    "PAMG","PANI","PANR","PANS","PART","PBID","PBSA","PBRX","PCAR","PDES",
    "PDPP","PEGE","PEHA","PELI","PENT","PERW","PEVE","PGAS","PGEO","PGJO",
    "PGLI","PGUN","PICO","PIPA","PJAA","PJHB","PKPK","PLAN","PLAS","PLIN",
    "PMJS","PMMP","PMUI","PNBN","PNBS","PNGO","PNIN","PNLF","PNSE","POLA",
    "POLI","POLL","POLU","POLY","POOL","PORT","POSA","POWR","PPGL","PPRI",
    "PPRE","PPRO","PRAY","PRDA","PRIM","PSAB","PSAT","PSDN","PSGO","PSKT",
    "PSSI","PTBA","PTDU","PTIS","PTMP","PTMR","PTPP","PTPS","PTPW","PTRO",
    "PTSN","PTSP","PUDP","PURA","PURE","PURI","PWON","PYFA","PZZA","RAAM",
    "RAFI","RAJA","RALS","RANC","RATU","RBMS","RCCC","RDTX","REAL","RELF",
    "RELI","REPP","RGAS","RICY","RIGS","RIMO","RISE","RLCO","RMBA","RMKE",
    "RMKO","RMLP","ROCK","RODA","ROLI","RONY","ROTI","RSCH","RSGK","RUIS",
    "RUNS","SAFE","SAGE","SAGI","SAME","SAMF","SAMR","SAMP","SANO","SAPX",
    "SATU","SBAT","SBMA","SCCO","SCMA","SCNP","SCPI","SDMU","SDPC","SDRA",
    "SEMA","SFAN","SGER","SGGH","SGJL","SGRO","SHID","SHIP","SICO","SIDO",
    "SIER","SILO","SIMA","SIMP","SINI","SIPD","SKBM","SKLT","SKRN","SKYB",
    "SLIS","SMAR","SMDM","SMDR","SMGA","SMGR","SMKM","SMKL","SMLE","SMMA",
    "SMMT","SMRA","SMRU","SMSM","SNLK","SOCI","SOFA","SOHO","SOLA","SONA",
    "SOSS","SOTS","SOUL","SPMA","SPRE","SPTO","SQMI","SRAJ","SREI","SRIL",
    "SRSN","SRTG","SSIA","SSMS","SSTM","STAA","STAR","STRK","STTP","SUGI",
    "SULI","SUNI","SUPA","SUPR","SURE","SWAT","SWID","SYAI","TALF","TAMA",
    "TAMU","TAPG","TARA","TAXI","TAYS","TBIG","TBLA","TBMS","TCID","TCPI",
    "TDPM","TEBE","TECH","TELE","TFAS","TFCO","TGKA","TGRA","TGUK","TIFA",
    "TINS","TIRA","TIRT","TKIM","TLDN","TLKM","TMAS","TMPO","TNCA","TOBA",
    "TOOL","TOPS","TOSK","TOTL","TOTO","TOWR","TOYS","TPAI","TPIA","TPMA",
    "TRAM","TRGU","TRIL","TRIM","TRIN","TRIO","TRIS","TRJA","TRON","TRST",
    "TRUE","TRUK","TRUS","TSPC","TUGU","TULT","TYRE","UANG","UCID","UDNG",
    "UFOE","ULTJ","UNIC","UNIQ","UNIT","UNSP","UNTR","UNVR","URBN","UVCR",
    "VAST","VATE","VCOK","VERN","VICI","VICO","VINS","VISA","VISI","VIVA",
    "VKTR","VOKS","VOSS","VRNA","VTNY","WAPO","WBSA","WEGE","WEHA","WGSH",
    "WICO","WIDI","WIFI","WIIM","WIKA","WINE","WINR","WINS","WIRG","WITA",
    "WMPP","WMUU","WOMF","WONS","WOOD","WOWS","WPOW","WSBP","WSKT","WTON",
    "YELO","YOII","YPAS","YULE","YUPI","ZATA","ZBRA","ZENI","ZINC","ZONE","ZYRX",
]
seen = set()
raw_stocks = [x for x in raw_stocks if not (x in seen or seen.add(x))]
stocks_yf  = [s + ".JK" for s in raw_stocks]
stock_map  = {s + ".JK": s for s in raw_stocks}


# ════════════════════════════════════════════════════
#  ANTI-BLACKLIST BATCH YFINANCE DOWNLOADER
# ════════════════════════════════════════════════════
def fetch_all_stock_data_safe(tickers_list, period="60d", interval="1d", batch_size=40):
    """
    Menarik data ratusan emiten secara berkelompok (batch) menggunakan 
    Multithreading bawaan yfinance & Custom Cached Session agar terhindar dari IP ban.
    """
    combined_data = {}
    total_tickers = len(tickers_list)
    
    # Elemen penampung progress teks di Streamlit
    progress_text = st.empty()
    
    # Pecah list ticker menjadi chunk/batch kecil (default per 40 emiten)
    for i in range(0, total_tickers, batch_size):
        chunk = tickers_list[i:i + batch_size]
        progress_text.text(f"⏳ Mengunduh batch emiten {i} sampai {min(i+batch_size, total_tickers)} dari {total_tickers}...")
        
        try:
            # Gunakan yf.download secara massal per batch dengan multithread aktif
            raw_batch = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                group_by='ticker',
                threads=True,
                progress=False,
                auto_adjust=True,
                session=safe_session, # Kirim requests via safe session
                timeout=15
            )
            
            # Jika chunk hanya menyisakan 1 emiten terakhir, format kolomnya single index
            if len(chunk) == 1:
                ticker_name = chunk[0]
                if not raw_batch.empty:
                    combined_data[ticker_name] = raw_batch
            else:
                # Masukkan single dataframe per ticker ke dictionary penampung
                for ticker in chunk:
                    if ticker in raw_batch.columns.get_level_values(0):
                        df_ticker = raw_batch[ticker]
                        if not df_ticker.empty:
                            combined_data[ticker] = df_ticker
                            
        except Exception as e:
            st.warning(f"⚠️ Gagal mengunduh kumpulan batch indeks ke-{i}: {str(e)[:50]}")
        
        # Jeda nafas kecil (1 detik) antar-batch agar server tidak mendeteksi lonjakan DDOS
        time.sleep(1.0)
        
    progress_text.empty()
    return combined_data


# ════════════════════════════════════════════════════
#  YFINANCE CLEAN EXTRACTION
# ════════════════════════════════════════════════════
def _yf_extract_clean(df_ticker):
    """
    Ekstraksi data untuk single DataFrame hasil keluaran pemrosesan batch.
    Merapikan kolom dan membuang baris kosong / volume nol.
    """
    try:
        if df_ticker is None or df_ticker.empty:
            return None
            
        df = df_ticker.copy()
        
        # Buang sisa level MultiIndex jika ada
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(-1)

        # Standarisasi huruf kapital (Open, High, Low, Close, Volume)
        rename_map = {c: c.capitalize() for c in df.columns if c.islower()}
        if rename_map:
            df = df.rename(columns=rename_map)
            
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return None

        df = df[required].copy()
        df = df.dropna(subset=['Close'])
        df = df[df['Volume'] > 0]
        
        return df if len(df) > 0 else None
    except:
        return None

# ════════════════════════════════════════════════════
#  MARKET REGIME DETECTOR — IHSG (FIXED)
# ════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_market_regime():
    try:
        df = yf.download("^JKSE", period="60d", interval="1d",
                         progress=False, auto_adjust=True, session=safe_session, timeout=10)
        if df is None or df.empty or len(df) < 10:
            return ("UNKNOWN", 0, 0, 0, "Data IHSG kurang", 0.0)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(-1)
        close = df["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if len(close) < 10:
            return ("UNKNOWN", 0, 0, 0, "Data close kurang", 0.0)

        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema55 = float(close.ewm(span=min(55, len(close)-1), adjust=False).mean().iloc[-1])
        price = float(close.iloc[-1])
        chg   = float(((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100)

        band            = 0.012
        pct_vs_e20      = (price - ema20) / ema20 * 100
        above_e20_any   = price > ema20 * (1 - band)
        above_e20_clear = price > ema20 * (1 + band)
        above_e55       = price > ema55
        recovering      = chg > 0.3
        bearish_confirm = chg < -0.3 and not above_e20_any

        if above_e20_clear and above_e55:
            regime = "GREEN"
            detail = f"IHSG {price:,.0f} > EMA20 & EMA55 → Bullish ✅ ({pct_vs_e20:+.1f}% vs EMA20)"
        elif above_e20_any and above_e55:
            regime = "GREEN"
            detail = f"IHSG {price:,.0f} dekat EMA20({ema20:,.0f}) & > EMA55 → Bullish"
        elif above_e20_any and not above_e55:
            regime = "SIDEWAYS"
            detail = f"IHSG {price:,.0f} > EMA20 tapi < EMA55({ema55:,.0f}) → Sideways"
        elif not above_e20_any and recovering:
            regime = "SIDEWAYS"
            detail = f"IHSG {price:,.0f} recovery {chg:+.2f}% (EMA20={ema20:,.0f}, gap {pct_vs_e20:+.1f}%)"
        elif bearish_confirm:
            regime = "RED"
            detail = f"IHSG {price:,.0f} < EMA20({ema20:,.0f}) {pct_vs_e20:+.1f}% + turun {chg:.2f}% → Bearish"
        else:
            regime = "SIDEWAYS"
            detail = f"IHSG {price:,.0f} sedikit < EMA20({ema20:,.0f}) {pct_vs_e20:+.1f}% → Sideways"

        return (regime, price, ema20, ema55, detail, chg)
    except Exception as e:
        return ("UNKNOWN", 0, 0, 0, f"IHSG error: {str(e)[:40]}", 0.0)

def get_regime_config(regime):
    return {
        "RED": {
            "mode": "Reversal 🎯", "min_score": 5, "min_rvol": 2.0, "sl_mult": 0.6,
            "label": "🔴 MARKET MERAH — Reversal Only, Score ≥ 5",
            "color": "#ff3d5a",
            "desc": "Market bearish. Fokus reversal oversold, filter ketat."
        },
        "GREEN": {
            "mode": "Bagger 💎", "min_score": 4, "min_rvol": 1.5, "sl_mult": 0.8,
            "label": "🟢 MARKET HIJAU — Wyckoff Bagger Hunt (Daily TF)",
            "color": "#00ff88",
            "desc": "Market bullish. Cari akumulasi Wyckoff di chart harian. RVOL ≥ 1.5x."
        },
        "SIDEWAYS": {
            "mode": "Scalping ⚡", "min_score": 4, "min_rvol": 2.0, "sl_mult": 0.7,
            "label": "🟡 MARKET SIDEWAYS — Scalping, RVOL ≥ 2x",
            "color": "#ffb700",
            "desc": "Market sideways. RVOL harus lebih kuat."
        },
        "UNKNOWN": {
            "mode": "Scalping ⚡", "min_score": 4, "min_rvol": 1.5, "sl_mult": 0.8,
            "label": "⚪ REGIME UNKNOWN — Manual Mode",
            "color": "#4a5568",
            "desc": "Tidak bisa deteksi kondisi market."
        },
    }.get(regime, {"mode":"Scalping ⚡","min_score":4,"min_rvol":1.5,"sl_mult":0.8,
                   "label":"⚪ UNKNOWN","color":"#4a5568","desc":""})


# ════════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════════
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi_smooth(s, p=14, smooth=3):
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(p).mean()
    loss  = (-delta.clip(upper=0)).rolling(p).mean()
    rs    = gain / loss.replace(0, np.nan)
    raw   = 100 - 100/(1+rs)
    return raw, ema(raw, smooth)

def stochastic(h, l, c, k=14, d=3):
    ll = l.rolling(k).min(); hh = h.rolling(k).max()
    K  = 100*(c-ll)/(hh-ll).replace(0,np.nan)
    D  = K.rolling(d).mean()
    return K.fillna(50), D.fillna(50)

def macd(s, f=12, sl=26, sg=9):
    ml = ema(s,f)-ema(s,sl); sig = ema(ml,sg)
    return ml, sig, ml-sig

def vwap(df):
    tp = (df['High']+df['Low']+df['Close'])/3
    return (tp*df['Volume']).cumsum()/df['Volume'].cumsum()

def apply_indicators(df):
    """Universal indicator function — works for both 15m and Daily."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(-1)
    df['EMA9']  = ema(df['Close'],9); df['EMA21'] = ema(df['Close'],21)
    df['EMA50'] = ema(df['Close'],50); df['EMA200']= ema(df['Close'],200)
    df['RSI'], df['RSI_EMA'] = rsi_smooth(df['Close'],14,3)
    df['STOCH_K'], df['STOCH_D'] = stochastic(df['High'],df['Low'],df['Close'],14,3)
    df['MACD'], df['MACD_Sig'], df['MACD_Hist'] = macd(df['Close'])
    try:    df['VWAP'] = vwap(df)
    except: df['VWAP'] = df['Close']
    df['BB_mid']   = df['Close'].rolling(20).mean()
    df['BB_std']   = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid']+2*df['BB_std']
    df['BB_lower'] = df['BB_mid']-2*df['BB_std']
    df['BB_pct']   = (df['Close']-df['BB_lower'])/(df['BB_upper']-df['BB_lower'])
    df['AvgVol']   = df['Volume'].rolling(20).mean()
    df['RVOL']     = df['Volume']/df['AvgVol'].replace(0, np.nan)
    df['NetVol']   = np.where(df['Close']>=df['Open'],df['Volume'],-df['Volume'])
    df['NetVol3']  = pd.Series(df['NetVol'],index=df.index).rolling(3).sum()
    df['NetVol8']  = pd.Series(df['NetVol'],index=df.index).rolling(8).sum()
    df['VolSpike'] = df['RVOL']>2.5
    df['Body']     = (df['Close']-df['Open']).abs()
    df['BodyRatio']= df['Body']/(df['High']-df['Low']).replace(0,np.nan)
    df['BullBar']  = (df['Close']>df['Open'])&(df['BodyRatio']>0.5)
    df['ROC3']     = df['Close'].pct_change(3)
    df['ROC8']     = df['Close'].pct_change(8)
    df['HH']= df['High']>df['High'].shift(1); df['HL']= df['Low']>df['Low'].shift(1)
    df['LL']= df['Low']<df['Low'].shift(1);    df['LH']= df['High']<df['High'].shift(1)
    tr = pd.concat([df['High']-df['Low'],
                    (df['High']-df['Close'].shift()).abs(),
                    (df['Low'] -df['Close'].shift()).abs()],axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

# Legacy alias
apply_intraday_indicators = apply_indicators

# ════════════════════════════════════════════════════
#  SCORING — Scalping, Momentum, Reversal
# ════════════════════════════════════════════════════
def score_scalping(r, p, p2):
    score=0; reasons=[]
    if r['EMA9']>r['EMA21']>r['EMA50']:   score+=1.5; reasons.append("EMA stack ▲")
    elif r['EMA9']>r['EMA21']:             score+=0.8; reasons.append("EMA9>21")
    if r['Close']>r['VWAP']:              score+=1; reasons.append("Above VWAP")
    if r['MACD_Hist']>0 and r['MACD_Hist']>float(p['MACD_Hist']):
        score+=1.5; reasons.append("MACD hist expanding ✦")
        if p2 is not None and float(p['MACD_Hist'])>float(p2['MACD_Hist']): score+=0.3
    elif r['MACD_Hist']>0:
        score+=0.5; reasons.append("MACD hist +")
    rsi_e=float(r['RSI_EMA'])
    if 52<rsi_e<68: score+=0.8; reasons.append(f"RSI-EMA={rsi_e:.1f}")
    elif rsi_e>=68: score-=0.5
    rvol=float(r['RVOL'])
    if rvol>2.0:   score+=1; reasons.append(f"RVOL={rvol:.1f}x surge")
    elif rvol>1.5: score+=0.6; reasons.append(f"RVOL={rvol:.1f}x")
    if bool(r['BullBar']): score+=0.5; reasons.append("Bullish bar")
    if float(r['NetVol3'])>0: score+=0.4; reasons.append("Net vol +")
    if r['Close']<r['EMA200']*0.98: score-=0.5
    return max(0,min(6,round(score,1))), reasons, {}

def score_momentum(r, p, p2):
    score=0; reasons=[]
    hh=bool(r['HH']); hl=bool(r['HL'])
    if hh and hl: score+=1.5; reasons.append("HH+HL pattern ▲")
    elif hh: score+=0.8
    rvol=float(r['RVOL'])
    if rvol>3.0:   score+=1.5; reasons.append(f"RVOL={rvol:.1f}x SURGE 🔥")
    elif rvol>2.0: score+=1.0; reasons.append(f"RVOL={rvol:.1f}x")
    elif rvol>1.5: score+=0.5
    roc=float(r['ROC3'])*100
    if roc>2.0:   score+=1.5; reasons.append(f"ROC3={roc:.1f}%")
    elif roc>1.0: score+=0.8; reasons.append(f"ROC3={roc:.1f}%")
    elif roc<0:   score-=0.5
    rsi_e=float(r['RSI_EMA'])
    if 55<rsi_e<75: score+=0.8; reasons.append(f"RSI-EMA={rsi_e:.1f}")
    if rsi_e>78:    score-=0.8; reasons.append("⚠️ RSI overbought")
    sk=float(r['STOCH_K']); sd=float(r['STOCH_D'])
    if sk>60 and sk>sd: score+=0.8; reasons.append("STOCH K>D bullish")
    if r['MACD_Hist']>0 and r['MACD_Hist']>float(p['MACD_Hist']): score+=0.8; reasons.append("MACD expanding")
    if r['Close']>r['VWAP']: score+=0.5; reasons.append("Above VWAP")
    return max(0,min(6,round(score,1))), reasons, {}

def score_reversal(r, p, p2):
    score=0; reasons=[]; os_count=0
    rsi_e=float(r['RSI_EMA'])
    if rsi_e<30:   os_count+=1; score+=1.5; reasons.append(f"RSI-EMA={rsi_e:.1f} OS extreme")
    elif rsi_e<40: os_count+=1; score+=0.8; reasons.append(f"RSI-EMA={rsi_e:.1f} OS")
    sk=float(r['STOCH_K']); sd=float(r['STOCH_D'])
    if sk<20:   os_count+=1; score+=1; reasons.append(f"STOCH={sk:.0f} extreme OS")
    elif sk<30: os_count+=1; score+=0.5
    bp=float(r['BB_pct'])
    if bp<0.05:   os_count+=1; score+=1; reasons.append("BB lower touch")
    elif bp<0.15: os_count+=1; score+=0.5
    if os_count<1.5: return 0,[],{}
    rev=0
    pk=float(p['STOCH_K']); pd_=float(p['STOCH_D'])
    if sk<30 and sk>sd and pk<=pd_: rev+=1; score+=2; reasons.append("STOCH %K cross ↑ OS ✦✦")
    elif sk<25 and sk>sd:           rev+=1; score+=1.2; reasons.append("STOCH K>D extreme OS")
    if p is not None:
        rsi_p=float(p['RSI_EMA'])
        if rsi_e>rsi_p and rsi_e<42: rev+=1; score+=1.2; reasons.append("RSI-EMA pivot ↑")
        mh=float(r['MACD_Hist']); mh_p=float(p['MACD_Hist'])
        if mh>mh_p and mh<0: rev+=1; score+=0.8; reasons.append("MACD hist diverge ↑")
    if rev==0: score*=0.3
    if bool(r['VolSpike']) and float(r['Close'])<float(r['Open']): score+=0.8; reasons.append("Volume climax sell")
    elif float(r['RVOL'])>1.5: score+=0.4
    # Sisa potongan kode di file dilanjutkan ke rendering streamlit...
    return max(0,min(6,round(score,1))), reasons, {}


# ════════════════════════════════════════════════════
#  STREAMLIT INTERFACE / ENGINE EXECUTION
# ════════════════════════════════════════════════════
st.title("Theta Turbo Dashboard v5.2 🔥")

# Sidebar / Kontrol Manual Mode & Scan Trigger
mode_selected = st.selectbox("Pilih Mode Scanning:", ["Scalping ⚡", "Momentum 🚀", "Reversal 🎯", "Bagger 💎"])
st.session_state.active_scan_mode = mode_selected

if st.button("🚀 Mulai Jalankan Scanner Massal", type="primary"):
    # Pengaturan Timeframe & Periode berdasarkan mode aktif
    chosen_tf = "1d" if st.session_state.active_scan_mode == "Bagger 💎" else "15m"
    chosen_period = "60d" if chosen_tf == "1d" else "7d"
    
    st.info(f"Mengaktifkan mode {st.session_state.active_scan_mode} dengan timeframe {chosen_tf}...")
    
    # 1. PANGGIL KELOMPOK BATCH AMAN (TIDAK AKAN KENA BLACKLIST)
    all_raw_data = fetch_all_stock_data_safe(stocks_yf, period=chosen_period, interval=chosen_tf, batch_size=40)
    
    scan_results = []
    
    # 2. PROSES PERHITUNGAN INDIKATOR DI MEMORI (SANGAT CEPAT)
    for ticker_yf, raw_df in all_raw_data.items():
        ticker_original = stock_map.get(ticker_yf, ticker_yf.replace(".JK", ""))
        
        # Bersihkan & Validasi Format Data per emiten
        df_clean = _yf_extract_clean(raw_df)
        if df_clean is None:
            continue
            
        # Hitung kalkulasi indikator
        df_final = apply_indicators(df_clean)
        if len(df_final) < 3:
            continue
            
        # Ambil baris data terbaru
        r_current  = df_final.iloc[-1]
        r_prev     = df_final.iloc[-2]
        r_prev2    = df_final.iloc[-3] if len(df_final) > 2 else None
        
        # Eksekusi fungsi scoring berdasarkan tipe pilihan
        if st.session_state.active_scan_mode == "Scalping ⚡":
            score, reasons, metadata = score_scalping(r_current, r_prev, r_prev2)
        elif st.session_state.active_scan_mode == "Momentum 🚀":
            score, reasons, metadata = score_momentum(r_current, r_prev, r_prev2)
        else:
            score, reasons, metadata = score_reversal(r_current, r_prev, r_prev2)
            
        # Simpan hasil screening jika score memenuhi kualifikasi dasar
        if score > 0:
            scan_results.append({
                "Ticker": ticker_original,
                "Price": r_current['Close'],
                "RVOL": round(r_current['RVOL'], 2),
                "Score": score,
                "Reasons": ", ".join(reasons)
            })
            
    # Tampilkan tabel output akhir
    if scan_results:
        st.session_state.scan_results = scan_results
        df_output = pd.DataFrame(scan_results).sort_values(by="Score", ascending=False)
        st.success(f"Berhasil menemukan {len(df_output)} emiten potensial!")
        st.dataframe(df_output, use_container_width=True)
    else:
        st.warning("Scan selesai, tidak ada emiten yang memenuhi kriteria setup indikator.")

# ════════════════════════════════════════════════════
#  FOOTER + AUTO-REFRESH
# ════════════════════════════════════════════════════
_now_f = datetime.now(jakarta_tz).timestamp()
if st.session_state.last_scan_time:
    _rem2   = max(0, 300-(_now_f-st.session_state.last_scan_time))
    mnt2    = int(_rem2//60); sec2=int(_rem2%60)
    last_t2 = datetime.fromtimestamp(st.session_state.last_scan_time,jakarta_tz).strftime("%H:%M:%S")
    last_mode_f = st.session_state.get("last_scan_mode","")
    tf_f    = "📅 Daily" if last_mode_f=="Bagger 💎" else "⚡ 15M"
    time_info = f"⏱️ Next: <span style='color:#ff7b00'>{mnt2:02d}:{sec2:02d}</span> · Last: <span style='color:#2dd4bf'>{last_t2} WIB</span> · {tf_f}"
else:
    _rem2 = 300
    time_info = "⏱️ Klik Scan untuk mulai"

st.markdown(f"""
<div style="margin-top:28px;padding-top:14px;border-top:1px solid #1c2533;font-size:11px;color:#4a5568;display:flex;justify-content:between;">
    <div>Theta Turbo v5.2 Engine • Safe Batch Mode Active</div>
    <div>{time_info}</div>
</div>
""", unsafe_allow_html=True)
