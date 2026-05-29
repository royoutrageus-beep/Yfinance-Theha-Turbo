import yfinance as yf
import pandas as pd
import streamlit as st
import time
import requests
import numpy as np
import pytz
from datetime import datetime

# ====== PERBAIKAN: MENGGUNAKAN STANDARD SESSION + RETRIES (ANTI-BAN) ======
from requests import Session
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse

def get_safe_session():
    """
    Membuat standard session dengan custom User-Agent browser asli 
    dan mekanisme otomatis Retry untuk bypass pembatasan rate-limit Yahoo.
    """
    session = Session()
    
    # Pasang mekanisme auto-retry jika sewaktu-waktu terkena limit kecil (HTTP 429 atau 503)
    # Ini akan membuat script otomatis menjeda beberapa milidetik sebelum mencoba lagi, bukan langsung crash.
    from urllib3.util import Retry
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    
    # Samarkan header agar menyerupai request organik dari browser Google Chrome
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
        'Origin': 'https://finance.yahoo.com',
        'Referer': 'https://finance.yahoo.com/',
        'Connection': 'keep-alive'
    })
    return session

safe_session = get_safe_session()
