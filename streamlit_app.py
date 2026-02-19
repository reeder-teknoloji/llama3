"""
Reev Fancy Teknik Chatbot – Streamlit UI
Lokal Ollama LLM + BM25 RAG tabanlı Türkçe teknik asistan.
"""
from __future__ import annotations

import json
import math
import re
import socket
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import streamlit as st


def get_local_ip() -> str:
    """Yerel ağ IP adresini döndürür."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ──────────────────────────────────────────────
# SABİT PARAMETRELER (en iyi sonuç için ayarlı)
# ──────────────────────────────────────────────
DEFAULT_INDEX_PATH = "knowledge/reev_technical_index.json"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

TOP_K = 4            # BM25 sonuç sayısı
TEMPERATURE = 0.15   # Düşük → tutarlı, az hayal gücü
MAX_GEN_LEN = 300    # Yeterli uzunlukta yanıt


# ╔══════════════════════════════════════════════╗
# ║              TÜRKÇE NLP ARAÇLARI             ║
# ╚══════════════════════════════════════════════╝

def fold_tr(text: str) -> str:
    """Türkçe karakterleri ASCII eşdeğerine dönüştürür."""
    text = (
        text.replace("İ", "I").replace("I", "I").replace("ı", "i")
        .replace("Ğ", "G").replace("ğ", "g")
        .replace("Ü", "U").replace("ü", "u")
        .replace("Ş", "S").replace("ş", "s")
        .replace("Ö", "O").replace("ö", "o")
        .replace("Ç", "C").replace("ç", "c")
    )
    return text.lower()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", fold_tr(text), flags=re.UNICODE)


# ╔══════════════════════════════════════════════╗
# ║              BM25 RETRIEVER                  ║
# ╚══════════════════════════════════════════════╝

class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.term_freqs: List[Counter[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_freqs: Counter[str] = Counter()
        self.avg_doc_len = 0.0

        for chunk in chunks:
            terms = tokenize(chunk["text"])
            tf = Counter(terms)
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(terms))
            for term in tf.keys():
                self.doc_freqs[term] += 1

        if self.doc_lengths:
            self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)

    def _idf(self, term: str) -> float:
        n_docs = len(self.chunks)
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def _score_doc(self, query_terms: List[str], doc_idx: int) -> float:
        if not self.avg_doc_len:
            return 0.0
        score = 0.0
        tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        norm = self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            score += self._idf(term) * ((f * (self.k1 + 1)) / (f + norm))
        return score

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Dict[str, str], float]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: List[Tuple[int, float]] = []
        for i in range(len(self.chunks)):
            s = self._score_doc(query_terms, i)
            if s > 0:
                scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        return [(self.chunks[i], s) for i, s in top]


# ╔══════════════════════════════════════════════╗
# ║           CHUNK YÜKLEME & FALLBACK           ║
# ╚══════════════════════════════════════════════╝

def load_chunks(index_path: Path) -> List[Dict[str, str]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        raise ValueError(f"No chunks found in {index_path}")
    return chunks


def extractive_fallback_answer(
    question: str,
    retrieved: List[Tuple[Dict[str, str], float]],
    all_chunks: List[Dict[str, str]] | None = None,
) -> str:
    """BM25 + anahtar kelime tabanlı yedek yanıt (LLM çalışmazsa)."""
    if not retrieved:
        return "Bu bilgiye sahip değilim."

    q_fold = fold_tr(question)
    q_terms = {t for t in q_fold.split() if len(t) > 2}

    keyword_to_topic = {
        "motor": ["motor-gucu", "genel-ozellikler", "nedir"],
        "guc": ["motor-gucu", "genel-ozellikler"],
        "batarya": ["batarya", "genel-ozellikler"],
        "pil": ["batarya"],
        "menzil": ["menzil", "genel-ozellikler"],
        "sarj": ["sarj", "genel-ozellikler"],
        "hiz": ["genel-tanitim", "genel-ozellikler"],
        "fren": ["vites-fren", "guvenlik"],
        "vites": ["vites-fren"],
        "jant": ["jant-lastik"],
        "lastik": ["jant-lastik"],
        "sunroof": ["sunroof-cam"],
        "cam": ["sunroof-cam", "on-cam"],
        "park": ["park"],
        "kamera": ["park"],
        "sensor": ["park"],
        "far": ["aydinlatma"],
        "led": ["aydinlatma"],
        "ekran": ["multimedya", "aydinlatma"],
        "gosterge": ["aydinlatma"],
        "multimedya": ["multimedya"],
        "bluetooth": ["multimedya"],
        "radyo": ["multimedya"],
        "mp3": ["multimedya"],
        "usb": ["multimedya"],
        "video": ["multimedya"],
        "avantaj": ["avantajlar"],
        "ekonomi": ["avantajlar"],
        "cevre": ["avantajlar"],
        "emisyon": ["avantajlar"],
        "tirman": ["tirmanma"],
        "sogut": ["sogutma"],
        "fan": ["sogutma"],
        "klima": ["sogutma", "ic-mekan"],
        "havalandirma": ["sogutma"],
        "bugu": ["on-cam"],
        "bagaj": ["genel-tanitim", "boyutlar"],
        "kapasite": ["genel-tanitim", "boyutlar"],
        "ehliyet": ["ehliyet", "genel-tanitim"],
        "yas": ["ehliyet", "genel-tanitim"],
        "fiyat": ["fiyat", "avantajlar"],
        "ucret": ["fiyat"],
        "taksit": ["fiyat"],
        "odeme": ["fiyat"],
        "pesinat": ["fiyat"],
        "satin": ["fiyat"],
        "siparis": ["fiyat"],
        "boyut": ["boyutlar"],
        "uzunluk": ["boyutlar"],
        "genislik": ["boyutlar"],
        "yukseklik": ["boyutlar"],
        "agirlik": ["boyutlar"],
        "kilo": ["boyutlar"],
        "uretici": ["uretici"],
        "reeder": ["uretici", "nedir"],
        "samsun": ["uretici"],
        "uygar": ["uretici"],
        "saral": ["uretici"],
        "yerli": ["uretici", "avantajlar"],
        "uret": ["uretici"],
        "kim": ["uretici"],
        "tasarim": ["tasarim"],
        "tasarimci": ["tasarim"],
        "rifat": ["tasarim"],
        "kromaj": ["tasarim"],
        "dodik": ["tasarim", "govde-kasa"],
        "koltuk": ["ic-mekan"],
        "deri": ["ic-mekan"],
        "izolasyon": ["ic-mekan"],
        "kilit": ["ic-mekan", "guvenlik"],
        "guvenlik": ["guvenlik"],
        "rakip": ["rakipler"],
        "citroen": ["rakipler"],
        "ami": ["rakipler"],
        "karsilastir": ["rakipler"],
        "renk": ["renk", "tasarim"],
        "rengi": ["renk", "tasarim"],
        "mavi": ["renk", "fiyat"],
        "beyaz": ["renk", "ic-mekan"],
        "siyah": ["renk"],
        "nedir": ["nedir", "genel-tanitim", "genel-ozellikler"],
        "ozellik": ["genel-ozellikler", "nedir"],
        "teknik": ["genel-ozellikler"],
        "reev": ["nedir", "genel-ozellikler"],
        "fancy": ["nedir", "genel-ozellikler"],
        "govde": ["govde-kasa"],
        "kasa": ["govde-kasa"],
        "sac": ["govde-kasa"],
        "plastik": ["govde-kasa"],
        "teslim": ["teslim", "fiyat"],
        "teslimat": ["teslim"],
    }

    topic_scores: Dict[str, int] = {}
    for term in q_terms:
        for kw, topics in keyword_to_topic.items():
            if kw in term or term in kw:
                for t in topics:
                    topic_scores[t] = topic_scores.get(t, 0) + 1

    def _extract_text(chunk: Dict[str, str]) -> str:
        text = chunk["text"]
        colon_pos = text.find(":")
        if colon_pos != -1 and colon_pos < 60:
            return text[colon_pos + 1:].strip()
        return text

    if topic_scores:
        max_score = max(topic_scores.values())
        best_topics = {t for t, s in topic_scores.items() if s == max_score}
        for chunk, _ in retrieved:
            cid = chunk.get("chunk_id", "")
            for topic in best_topics:
                if topic in cid:
                    return _extract_text(chunk)
        if all_chunks:
            for chunk in all_chunks:
                cid = chunk.get("chunk_id", "")
                for topic in best_topics:
                    if topic in cid:
                        return _extract_text(chunk)

    best_chunk_text = retrieved[0][0]["text"]
    colon_pos = best_chunk_text.find(":")
    if colon_pos != -1 and colon_pos < 60:
        return best_chunk_text[colon_pos + 1:].strip()
    return best_chunk_text


# ╔══════════════════════════════════════════════╗
# ║        KEYWORD CHUNK INJECTION               ║
# ╚══════════════════════════════════════════════╝

_KEYWORD_TO_TOPIC = {
    "motor": ["motor-gucu"], "guc": ["motor-gucu"], "batarya": ["batarya"],
    "pil": ["batarya"], "menzil": ["menzil"], "sarj": ["sarj"],
    "hiz": ["genel-tanitim"], "fren": ["vites-fren"], "vites": ["vites-fren"],
    "jant": ["jant-lastik"], "lastik": ["jant-lastik"], "sunroof": ["sunroof-cam"],
    "cam": ["sunroof-cam", "on-cam"], "park": ["park"], "kamera": ["park"],
    "sensor": ["park"], "far": ["aydinlatma"], "led": ["aydinlatma"],
    "ekran": ["multimedya"], "gosterge": ["aydinlatma"], "multimedya": ["multimedya"],
    "bluetooth": ["multimedya"], "radyo": ["multimedya"], "usb": ["multimedya"],
    "video": ["multimedya"],
    "avantaj": ["avantajlar"], "ekonomi": ["avantajlar"], "cevre": ["avantajlar"],
    "tirman": ["tirmanma"], "sogut": ["sogutma"], "klima": ["sogutma", "ic-mekan"],
    "havalandirma": ["sogutma"], "bugu": ["on-cam"],
    "bagaj": ["genel-tanitim", "boyutlar"], "kapasite": ["genel-tanitim"],
    "ehliyet": ["ehliyet"], "yas": ["ehliyet"],
    "fiyat": ["fiyat"], "ucret": ["fiyat"], "taksit": ["fiyat"],
    "odeme": ["fiyat"], "pesinat": ["fiyat"], "satin": ["fiyat"],
    "boyut": ["boyutlar"], "uzunluk": ["boyutlar"], "genislik": ["boyutlar"],
    "yukseklik": ["boyutlar"], "agirlik": ["boyutlar"], "kilo": ["boyutlar"],
    "uretici": ["uretici"], "uret": ["uretici"], "reeder": ["uretici", "nedir"],
    "samsun": ["uretici"], "saral": ["uretici"], "yerli": ["uretici"],
    "kim": ["uretici"],
    "tasarim": ["tasarim"], "tasarimci": ["tasarim"], "rifat": ["tasarim"],
    "koltuk": ["ic-mekan"], "deri": ["ic-mekan"], "izolasyon": ["ic-mekan"],
    "kilit": ["ic-mekan"], "guvenlik": ["guvenlik"],
    "rakip": ["rakipler"], "citroen": ["rakipler"], "ami": ["rakipler"],
    "karsilastir": ["rakipler"],
    "renk": ["renk", "tasarim"], "rengi": ["renk", "tasarim"],
    "mavi": ["renk", "fiyat"], "beyaz": ["renk", "ic-mekan"], "siyah": ["renk"],
    "nedir": ["nedir", "genel-tanitim"], "ozellik": ["genel-ozellikler"],
    "teknik": ["genel-ozellikler"], "reev": ["nedir"], "fancy": ["nedir"],
    "govde": ["govde-kasa"], "kasa": ["govde-kasa"], "sac": ["govde-kasa"],
    "plastik": ["govde-kasa"], "dodik": ["govde-kasa", "tasarim"],
    "teslim": ["teslim", "fiyat"], "teslimat": ["teslim"],
}


def inject_keyword_chunks(
    question: str,
    retrieved: List[Tuple[Dict[str, str], float]],
    all_chunks: List[Dict[str, str]],
) -> List[Tuple[Dict[str, str], float]]:
    """Anahtar kelime eşleşmesiyle gerekli chunk'ları garanti altına al."""
    q_fold = fold_tr(question)
    q_terms = {t for t in q_fold.split() if len(t) > 2}

    needed_topics: set[str] = set()
    for term in q_terms:
        for kw, topics in _KEYWORD_TO_TOPIC.items():
            if kw in term or term in kw:
                needed_topics.update(topics)

    if not needed_topics:
        needed_topics = {"genel-ozellikler", "nedir"}

    existing_ids = {c.get("chunk_id", "") for c, _ in retrieved}
    extras: List[Tuple[Dict[str, str], float]] = []
    for chunk in all_chunks:
        cid = chunk.get("chunk_id", "")
        if cid in existing_ids:
            continue
        for topic in needed_topics:
            if topic in cid:
                extras.append((chunk, 0.01))
                existing_ids.add(cid)
                break

    return retrieved + extras


# ╔══════════════════════════════════════════════╗
# ║              PROMPT OLUŞTURMA                ║
# ╚══════════════════════════════════════════════╝

def build_prompt(
    question: str,
    retrieved: List[Tuple[Dict[str, str], float]],
    history: List[Dict[str, str]],
) -> str:
    blocks: List[str] = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        blocks.append(f"[Kaynak {i}]\n{chunk['text']}")
    context = "\n\n".join(blocks) if blocks else "Baglam yok."

    short_history = history[-4:] if history else []
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in short_history)

    return (
        "Sen Reev Fancy elektrikli arac konusunda uzman bir Turkce teknik asistansin.\n\n"
        "GOREV:\n"
        "Kullanicinin sordugu soru Reev Fancy araci ile ilgiliyse, baglamdaki bilgileri kullanarak "
        "DETAYLI ve ACIKLAYICI bir Turkce cevap ver. Eksik birakma, mumkun oldugunca bilgi ver.\n\n"
        "KURALLAR:\n"
        "1. Soru Reev Fancy ile ilgiliyse (aracin herhangi bir ozelligi, parcasi, fiyati, rengi, tasarimi, "
        "lastigi, motoru, bataryasi, menzili, govdesi, teslim suresi vb.) MUTLAKA cevapla.\n"
        "2. Baglamda bilgi varsa onu kullan. Baglamda olmayan bilgiyi UYDURMA.\n"
        "3. Sacma veya imkansiz sorularda (ornegin: ucar mi, yuzer mu, uzaya gider mi) "
        "mantikli ve dogru cevap ver. Ornegin: 'Hayir, Reev Fancy bir kara aracidir, ucamaz.'\n"
        "4. Soru Reev Fancy ile TAMAMEN ALAKASIZ ise (hava durumu, yemek tarifi, spor, siyaset, matematik): "
        "'Ben sadece Reev Fancy elektrikli arac hakkinda sorulari yanitlayabilirim.' de.\n"
        "5. 'Reev Fancy nedir?' gibi genel sorularda aracin TUM onemli ozelliklerini ozetle. "
        "Kisa bir paragraf degil, detayli bir tanitim yap.\n"
        "6. Turkce yaz. Cevaba dogrudan basla, 'Merhaba' veya giris cumlesi YAZMA.\n\n"
        "ORNEKLER:\n"
        "Soru: Motor gucu nedir?\n"
        "Cevap: Reev Fancy'nin motor gucu 6 kW (kilowatt) elektrikli motordur. Bu motor aracin azami 45 km/s hiza ulasmasi icin yeterlidir.\n\n"
        "Soru: Lastik boyutu ne kadar?\n"
        "Cevap: Reev Fancy 13 inc ozel tasarim jantlara sahiptir ve lastik boyutu 145/70 R13'tur.\n\n"
        "Soru: Rengi nedir?\n"
        "Cevap: Reev Fancy beyaz, siyah ve mavi renk secenekleri ile sunulmaktadir. On siparise ozel ilk 100 arac mavi renkte uretilmistir.\n\n"
        "Soru: Fiyati ne kadar?\n"
        "Cevap: Reev Fancy'nin liste fiyati 475.000 TL'dir. Indirimli kampanya fiyati ise 445.000 TL'dir. 12, 24, 36 ve 48 aylik taksit secenekleri mevcuttur.\n\n"
        "Soru: Govdesi neyden yapilmis?\n"
        "Cevap: Reev Fancy'nin govdesi sac kasa malzemeden uretilmistir. Dodikleri (dis kaplamalar) sertlestirilmis plastik malzemeden yapilmistir.\n\n"
        "Soru: Teslim suresi ne kadar?\n"
        "Cevap: Reev Fancy'nin teslim suresi 20 gundur. Siparis verdikten sonra 20 gun icinde arac teslim edilmektedir.\n\n"
        "Soru: Ucar mi?\n"
        "Cevap: Hayir, Reev Fancy bir kara aracidir ve ucma ozelligi yoktur. Azami hizi 45 km/s olan elektrikli bir mikro mobilite aracidir.\n\n"
        "Soru: Bugunun hava durumu nasil?\n"
        "Cevap: Ben sadece Reev Fancy elektrikli arac hakkinda sorulari yanitlayabilirim.\n\n"
        "Soru: Reev Fancy nedir?\n"
        "Cevap: Reev Fancy, Reeder Teknoloji tarafindan uretilen, sehir ici kisa mesafe ulasim icin "
        "tasarlanmis 2 kisilik elektrikli mikro mobilite aracidir. L6e sinifinda yer alir. "
        "6 kW elektrikli motor, 7.68 kWh LiFePO4 batarya, 100 km menzil ve 45 km/s azami hiz sunar. "
        "200 litre bagaj hacmi, otomatik vites, 4 teker disk fren, LED farlar, 7 inc dijital gosterge, "
        "Bluetooth destekli multimedya sistemi, geri gorus kamerasi ve park sensoru gibi donanimlara "
        "sahiptir. Govdesi sac kasa ve sertlestirilmis plastik (dodik) yapidadir. Sunroof standart olarak gelir. "
        "Ev tipi 220V priz ile yaklasik 5 saatte tam sarj edilir. B1 sinifi ehliyet yeterlidir. "
        "Indirimli fiyati 445.000 TL'dir ve teslim suresi 20 gundur.\n\n"
        f"BAGLAM:\n{context}\n\n"
        + (f"GECMIS:\n{history_text}\n\n" if history_text else "")
        + f"Soru: {question}\nCevap:"
    )


# ╔══════════════════════════════════════════════╗
# ║             OLLAMA API ÇAĞRISI               ║
# ╚══════════════════════════════════════════════╝

def ask_ollama(
    ollama_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_gen_len: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_gen_len,
        },
    }
    resp = requests.post(ollama_url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


# ╔══════════════════════════════════════════════╗
# ║          STREAMLIT SAYFA AYARLARI            ║
# ╚══════════════════════════════════════════════╝

st.set_page_config(
    page_title="ReeV Fancy Asistan",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── IP Adresi ──
_LOCAL_IP = get_local_ip()

# ── Özel CSS ile Arayüz Güzelleştirme ──
st.markdown("""
<style>
    /* ── Genel: Açık arka plan, koyu metin ── */
    .stApp {
        background: #f0f4f8 !important;
    }

    /* Tüm metin varsayılan koyu */
    .stApp, .stApp p, .stApp span, .stApp li, .stApp label,
    .stApp div, .stMarkdown, .stMarkdown p {
        color: #1e293b !important;
    }

    /* Başlık alanı */
    .hero-header {
        text-align: center;
        padding: 1.8rem 1rem 0.6rem 1rem;
        margin-bottom: 0.3rem;
    }
    .hero-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0ea5e9, #2563eb, #7c3aed);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 4s ease infinite;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }
    .hero-header .subtitle {
        color: #64748b !important;
        font-size: 0.95rem;
        margin-top: 0;
    }

    /* Spec kartları */
    .spec-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.7rem;
        margin: 0.8rem 0 1rem 0;
    }
    .spec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 0.6rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .spec-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(14,165,233,0.12);
        border-color: #0ea5e9;
    }
    .spec-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0ea5e9 !important;
        display: block;
    }
    .spec-card .label {
        font-size: 0.75rem;
        color: #64748b !important;
        margin-top: 0.25rem;
        display: block;
    }

    /* Ayırıcı */
    .divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 0.8rem 0;
    }

    /* Chat mesajları */
    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.7rem !important;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    }
    [data-testid="stChatMessage"] p {
        color: #1e293b !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #0ea5e9 !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #334155 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        border: 1.5px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #1e293b !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 2px rgba(14,165,233,0.15) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        background: #f8fafc !important;
    }
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #334155 !important;
    }

    /* Buton */
    .stButton > button {
        border-radius: 10px !important;
        border: 1.5px solid #0ea5e9 !important;
        background: rgba(14,165,233,0.06) !important;
        color: #0ea5e9 !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #0ea5e9 !important;
        color: #ffffff !important;
    }

    /* Kaynak rozeti */
    .source-badge-ai {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        background: rgba(14,165,233,0.1);
        color: #0369a1 !important;
        border: 1px solid rgba(14,165,233,0.25);
        margin-top: 0.4rem;
    }
    .source-badge-fallback {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        background: rgba(245,158,11,0.1);
        color: #b45309 !important;
        border: 1px solid rgba(245,158,11,0.3);
        margin-top: 0.4rem;
    }

    /* IP info kartı */
    .ip-card {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin: 0.5rem 0;
    }
    .ip-card .ip-title {
        font-size: 0.7rem;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .ip-card .ip-value {
        font-size: 0.85rem;
        font-family: 'Consolas', 'Monaco', monospace;
        color: #0369a1 !important;
        word-break: break-all;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #94a3b8 !important;
        font-size: 0.72rem;
        padding: 1.5rem 0 0.5rem 0;
    }

    /* Hızlı soru etiketi */
    .quick-label {
        color: #334155 !important;
        font-weight: 600;
    }

    /* Code blokları sidebar'da */
    [data-testid="stSidebar"] code {
        color: #334155 !important;
        background: #f1f5f9 !important;
    }

    /* Text input'lar */
    [data-testid="stSidebar"] input {
        color: #1e293b !important;
        background: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Hero Header ──
st.markdown("""
<div class="hero-header">
    <h1>🚗 ReeV Fancy Asistan</h1>
    <p class="subtitle">Türkiye'nin elektrikli mikro aracı hakkında her şeyi sorun</p>
</div>
""", unsafe_allow_html=True)

# ── Spec Kartları ──
st.markdown("""
<div class="spec-grid">
    <div class="spec-card">
        <span class="value">6 kW</span>
        <span class="label">Motor Gücü</span>
    </div>
    <div class="spec-card">
        <span class="value">100 km</span>
        <span class="label">Menzil</span>
    </div>
    <div class="spec-card">
        <span class="value">445.000 ₺</span>
        <span class="label">İndirimli Fiyat</span>
    </div>
    <div class="spec-card">
        <span class="value">20 Gün</span>
        <span class="label">Teslim Süresi</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Session State ──
if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = DEFAULT_OLLAMA_URL
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar (gelişmiş ayarlar – varsayılan kapalı) ──
with st.sidebar:
    st.markdown("## ⚙️ Gelişmiş Ayarlar")
    st.caption("Bu ayarlar optimize edilmiş şekilde sabitlenmiştir. Değiştirmenize gerek yoktur.")

    st.markdown("---")
    st.markdown("**Model Bilgileri**")
    st.code(f"Model: {DEFAULT_OLLAMA_MODEL}\nTop K: {TOP_K}\nTemperature: {TEMPERATURE}\nMax Token: {MAX_GEN_LEN}", language=None)

    st.markdown("---")
    st.markdown("**Bağlantı**")
    st.session_state.ollama_url = st.text_input("Ollama URL", value=st.session_state.ollama_url)
    st.session_state.ollama_model = st.text_input("Model Adı", value=st.session_state.ollama_model)

    st.markdown("---")
    if st.button("🔌 Bağlantıyı Test Et", use_container_width=True):
        try:
            test = requests.get(
                st.session_state.ollama_url.replace("/api/generate", "/api/tags"),
                timeout=8,
            )
            test.raise_for_status()
            st.success("✅ Ollama bağlantısı başarılı!")
        except Exception as exc:
            st.error(f"❌ Bağlantı hatası: {exc}")

    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**🌐 Erişim Adresleri**")
    st.markdown(
        f'<div class="ip-card">'
        f'<div class="ip-title">Bu Bilgisayar</div>'
        f'<div class="ip-value">http://localhost:8501</div>'
        f'</div>'
        f'<div class="ip-card">'
        f'<div class="ip-title">Ağdaki Diğer Cihazlar</div>'
        f'<div class="ip-value">http://{_LOCAL_IP}:8501</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<p style="color:#94a3b8;font-size:0.7rem;text-align:center;">'
        "Lokal Ollama + BM25 RAG<br>28 bilgi parçacığı"
        "</p>",
        unsafe_allow_html=True,
    )


# ── Hızlı Sorular (ilk açılışta) ──
if not st.session_state.messages:
    st.markdown('<span class="quick-label">💡 Hızlı Sorular:</span>', unsafe_allow_html=True)
    quick_cols = st.columns(4)
    quick_questions = [
        "Reev Fancy nedir?",
        "Fiyatı ne kadar?",
        "Menzili kaç km?",
        "Batarya bilgileri",
    ]
    for i, q in enumerate(quick_questions):
        if quick_cols[i].button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()


# ── Geçmiş Mesajları Göster ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "answer_source" in message:
            src = message["answer_source"]
            if src == "ai":
                st.markdown(
                    '<span class="source-badge-ai">🤖 Yapay zeka yanıtı</span>',
                    unsafe_allow_html=True,
                )
            elif src == "fallback":
                st.markdown(
                    '<span class="source-badge-fallback">📄 Anahtar kelime eşleşmesi</span>',
                    unsafe_allow_html=True,
                )


# ── Chat Input ──
question = st.chat_input("Reev Fancy hakkında bir soru sorun...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    answer_source = ""
    retrieved: List[Tuple[Dict[str, str], float]] = []

    try:
        with st.spinner("🔍 Bilgi taranıyor, yanıt üretiliyor..."):
            chunks = load_chunks(Path(DEFAULT_INDEX_PATH))
            retriever = BM25Retriever(chunks)
            retrieved = retriever.retrieve(question, k=TOP_K)
            retrieved = inject_keyword_chunks(question, retrieved, chunks)
            prompt = build_prompt(question, retrieved, st.session_state.messages[:-1])

            try:
                answer = ask_ollama(
                    ollama_url=st.session_state.ollama_url,
                    model=st.session_state.ollama_model,
                    prompt=prompt,
                    temperature=TEMPERATURE,
                    max_gen_len=MAX_GEN_LEN,
                )
                if answer:
                    answer_source = "ai"
                else:
                    answer = extractive_fallback_answer(question, retrieved, chunks)
                    answer_source = "fallback"
            except Exception as ollama_exc:
                st.warning(f"⚠️ Ollama yanıt veremedi ({ollama_exc}). Yedek sistem kullanıldı.")
                answer = extractive_fallback_answer(question, retrieved, chunks)
                answer_source = "fallback"

    except Exception as exc:
        answer = f"❌ Hata: {exc}"
        answer_source = "error"
        retrieved = []

    # Yanıtı kaydet (source bilgisi ile)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "answer_source": answer_source,
    })

    with st.chat_message("assistant"):
        st.markdown(answer)
        if answer_source == "ai":
            st.markdown(
                '<span class="source-badge-ai">🤖 Yapay zeka yanıtı</span>',
                unsafe_allow_html=True,
            )
        elif answer_source == "fallback":
            st.markdown(
                '<span class="source-badge-fallback">📄 Anahtar kelime eşleşmesi</span>',
                unsafe_allow_html=True,
            )

    if retrieved:
        with st.expander("📚 Kullanılan Kaynaklar", expanded=False):
            for idx, (chunk, score) in enumerate(retrieved, start=1):
                cid = chunk.get("chunk_id", "").split(":")[-1]
                st.markdown(
                    f"**{idx}.** `{cid}` · skor: `{score:.2f}` · _{chunk['source']}_"
                )


# ── Footer ──
st.markdown(
    f'<p class="footer-text">ReeV Fancy Teknik Asistan · Lokal LLM + RAG · '
    f'Ollama + Streamlit · <span style="font-family:monospace;">http://{_LOCAL_IP}:8501</span></p>',
    unsafe_allow_html=True,
)
