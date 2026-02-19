# 📘 ReeV Fancy Teknik Chatbot – Kullanım Kılavuzu

## İçindekiler

1. [Genel Bakış](#1-genel-bakış)
2. [Sistem Gereksinimleri](#2-sistem-gereksinimleri)
3. [Kurulum](#3-kurulum)
4. [Uygulamayı Başlatma](#4-uygulamayı-başlatma)
5. [Arayüz Kullanımı](#5-arayüz-kullanımı)
6. [Sorulabilecek Soru Örnekleri](#6-sorulabilecek-soru-örnekleri)
7. [Teknik Mimari](#7-teknik-mimari)
8. [Yapılandırma & Parametreler](#8-yapılandırma--parametreler)
9. [Bilgi Tabanı (Knowledge Base)](#9-bilgi-tabanı-knowledge-base)
10. [Sorun Giderme (Troubleshooting)](#10-sorun-giderme-troubleshooting)
11. [SSS (Sıkça Sorulan Sorular)](#11-sss-sıkça-sorulan-sorular)

---

## 1. Genel Bakış

**ReeV Fancy Teknik Chatbot**, Reev Fancy elektrikli mikro mobilite aracı hakkındaki teknik sorulara **tamamen lokal olarak** (internet gerekmeden) yanıt veren bir yapay zeka asistanıdır.

### Temel Özellikler

| Özellik | Açıklama |
|---------|----------|
| **LLM (Dil Modeli)** | Ollama üzerinden çalışan `llama3.1:8b` modeli |
| **RAG Sistemi** | BM25 tabanlı bilgi arama + anahtar kelime enjeksiyonu |
| **Dil** | Türkçe (Türkçe karakter normalizasyonu dahil) |
| **Bilgi Kaynağı** | 28 adet konu bazlı bilgi parçacığı |
| **Arayüz** | Streamlit tabanlı modern web arayüzü |
| **Gizlilik** | Tüm veriler ve işlemler lokalde çalışır |

---

## 2. Sistem Gereksinimleri

### Minimum Gereksinimler

| Bileşen | Gereksinim |
|---------|-----------|
| **İşletim Sistemi** | Windows 10/11 (64-bit) |
| **İşlemci** | Intel i5 / AMD Ryzen 5 veya üzeri |
| **RAM** | 12 GB (16 GB önerilir) |
| **Disk** | 10 GB boş alan |
| **GPU** | Opsiyonel (NVIDIA GPU varsa hızlandırır) |
| **Python** | 3.10 veya üzeri |

### Yazılım Gereksinimleri

- **Python 3.10+** → [python.org](https://www.python.org/downloads/)
- **Ollama** → [ollama.com](https://ollama.com/download)
- **Git** (opsiyonel) → [git-scm.com](https://git-scm.com/download/win)

---

## 3. Kurulum

### 3.1 Ollama Kurulumu

1. [ollama.com/download](https://ollama.com/download) adresinden Ollama'yı indirin ve kurun.
2. Kurulum tamamlandığında Ollama otomatik olarak arka planda çalışmaya başlar.
3. Gerekli modeli indirin:

```powershell
ollama pull llama3.1:8b
```

> ⚠️ Model yaklaşık **4.9 GB** boyutundadır. İndirme internet hızınıza göre zaman alabilir.

### 3.2 Python Bağımlılıkları

Proje klasörüne gidin ve sanal ortam oluşturun:

```powershell
cd "c:\Users\<KULLANICI>\Desktop\survey-reeder-frontend-main\llama3"

# Sanal ortam oluştur
python -m venv .venv

# Sanal ortamı aktifleştir
.venv\Scripts\activate

# Bağımlılıkları kur
pip install streamlit requests
```

### 3.3 Dosya Yapısı

Kurulum sonrası proje yapısı şu şekilde olmalıdır:

```
llama3/
├── streamlit_app.py          ← Ana uygulama
├── knowledge/
│   └── reev_technical_index.json  ← Bilgi tabanı (28 parça)
├── start_app.bat             ← Hızlı başlatma scripti
├── .venv/                    ← Python sanal ortamı
├── chatbot.py                ← CLI chatbot (alternatif)
└── requirements.txt          ← Bağımlılık listesi
```

---

## 4. Uygulamayı Başlatma

### Yöntem 1: BAT Dosyası ile (En Kolay)

`start_app.bat` dosyasına çift tıklayın. Otomatik olarak:
- Sanal ortamı aktifleştirir
- Streamlit uygulamasını başlatır
- Tarayıcıda açar

### Yöntem 2: Terminal ile

```powershell
cd "c:\Users\<KULLANICI>\Desktop\survey-reeder-frontend-main\llama3"
.venv\Scripts\activate
streamlit run streamlit_app.py
```

### Yöntem 3: Doğrudan Python ile

```powershell
.venv\Scripts\streamlit.exe run streamlit_app.py --server.headless true
```

> Uygulama varsayılan olarak **http://localhost:8501** adresinde açılır.

### Başlatma Öncesi Kontrol Listesi

- [ ] Ollama'nın çalıştığından emin olun (sistem tepsisinde Ollama simgesi görünür)
- [ ] `llama3.1:8b` modelinin indirilmiş olduğunu doğrulayın: `ollama list`
- [ ] Python sanal ortamının `.venv` klasöründe bulunduğundan emin olun

---

## 5. Arayüz Kullanımı

### 5.1 Ana Ekran

Uygulama açıldığında şunları göreceksiniz:

- **Başlık**: "🚗 ReeV Fancy Asistan"
- **Spec Kartları**: 4 adet öne çıkan teknik özellik (Motor, Menzil, Fiyat, Teslim)
- **Hızlı Sorular**: İlk açılışta 4 adet hazır soru butonu
- **Chat Alanı**: Soru yazma ve yanıt görüntüleme alanı

### 5.2 Soru Sorma

1. Ekranın alt kısmındaki metin kutusuna sorunuzu yazın
2. Enter tuşuna basın veya gönder butonuna tıklayın
3. Yanıt üretilirken "🔍 Bilgi taranıyor, yanıt üretiliyor..." mesajı görünür
4. Yanıt üretildikten sonra:
   - **🤖 Yapay zeka yanıtı**: LLM tarafından üretilen yanıt
   - **📄 Anahtar kelime eşleşmesi**: Yedek sistem tarafından üretilen yanıt

### 5.3 Kaynakları Görüntüleme

Her yanıtın altında **"📚 Kullanılan Kaynaklar"** bölümünü genişleterek hangi bilgi parçacıklarının kullanıldığını görebilirsiniz.

### 5.4 Sidebar (Sol Panel)

Sol panele erişmek için sol üst köşedeki **">"** simgesine tıklayın:

- **Model Bilgileri**: Kullanılan model ve optimize edilmiş parametreler
- **Bağlantı Ayarları**: Ollama URL ve model adı (değiştirilebilir)
- **🔌 Bağlantıyı Test Et**: Ollama bağlantısını kontrol eder
- **🗑️ Sohbeti Temizle**: Tüm sohbet geçmişini siler

### 5.5 Hızlı Sorular

İlk açılışta 4 hazır soru butonu gösterilir:
- "Reev Fancy nedir?"
- "Fiyatı ne kadar?"
- "Menzili kaç km?"
- "Batarya bilgileri"

Butona tıklamak soruyu otomatik olarak gönderir.

---

## 6. Sorulabilecek Soru Örnekleri

### ✅ Desteklenen Sору Kategorileri

| Kategori | Örnek Sorular |
|----------|---------------|
| **Genel** | Reev Fancy nedir? / Ne tür bir araçtır? |
| **Motor** | Motor gücü nedir? / Kaç kW? |
| **Batarya** | Batarya kapasitesi nedir? / Batarya tipi ne? |
| **Menzil** | Menzili kaç km? / Tek şarjla ne kadar gider? |
| **Şarj** | Şarj süresi ne kadar? / Nasıl şarj edilir? |
| **Fiyat** | Fiyatı ne kadar? / Taksit var mı? |
| **Boyutlar** | Boyutları nedir? / Ağırlığı kaç kg? |
| **Tasarım** | Kim tasarladı? / Tasarımcısı kim? |
| **Renk** | Renk seçenekleri nelerdir? / Hangi renklerde var? |
| **Lastik** | Lastik boyutu nedir? / Jant kaç inç? |
| **Gövde** | Gövdesi neyden yapılmış? / Kasa malzemesi ne? |
| **Güvenlik** | Güvenlik özellikleri nelerdir? / Fren sistemi ne? |
| **İç Mekan** | İç mekanında ne var? / Klima var mı? |
| **Multimedya** | Multimedya sistemi ne? / Bluetooth var mı? |
| **Ehliyet** | Hangi ehliyetle kullanılır? / Kaç yaşından itibaren? |
| **Üretici** | Kim üretiyor? / Nerede üretiliyor? |
| **Rakipler** | Rakipleri kimler? / Citroen Ami ile karşılaştır |
| **Teslim** | Teslim süresi ne kadar? |
| **Sunroof** | Sunroof var mı? |
| **Park** | Park sensörü var mı? / Geri görüş kamerası var mı? |

### ❌ Desteklenmeyen Sorular

Aşağıdaki konularda bot nazikçe yönlendirme yapar:
- Hava durumu, yemek tarifleri, spor, siyaset, matematik
- Diğer araç markaları hakkında detaylı bilgi
- Kişisel tavsiye veya yorumlar

### 🤔 Saçma/İmkansız Sorular

Bot saçma soruları mantıklı şekilde yanıtlar:
- "Uçar mı?" → "Hayır, Reev Fancy bir kara aracıdır"
- "Yüzer mi?" → "Hayır, kara aracıdır"

---

## 7. Teknik Mimari

### Sistem Akışı

```
Kullanıcı Sorusu
       │
       ▼
┌──────────────────────┐
│   Türkçe Karakter    │
│   Normalizasyonu     │  (İ→I, ş→s, ö→o, ç→c, ğ→g, ü→u)
│   (fold_tr)          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     BM25 Arama       │  28 bilgi parçacığı içinde
│  (Top 4 sonuç)       │  anahtar kelime bazlı sıralama
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Anahtar Kelime      │  Eşleşen chunk'ları
│  Enjeksiyonu         │  sonuçlara ekle
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Prompt Oluşturma    │  Kurallar + Örnekler +
│  (build_prompt)      │  Bağlam + Geçmiş + Soru
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Ollama LLM          │  llama3.1:8b modeli
│  (ask_ollama)        │  temp=0.15, max_token=300
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Yanıt + Kaynak      │  🤖 AI yanıtı veya
│  Gösterimi           │  📄 Fallback yanıtı
└──────────────────────┘
```

### Bileşenler

| Bileşen | Teknoloji | Açıklama |
|---------|-----------|----------|
| **Arayüz** | Streamlit | Modern, responsive web UI |
| **LLM** | Ollama + llama3.1:8b | 4.9 GB lokal dil modeli |
| **Arama** | BM25 | TF-IDF tabanlı bilgi arama |
| **Enjeksiyon** | Keyword Mapping | Anahtar kelime → chunk eşleme |
| **Fallback** | Extractive | LLM çalışmazsa doğrudan chunk |
| **NLP** | fold_tr + tokenize | Türkçe karakter normalizasyonu |

---

## 8. Yapılandırma & Parametreler

### Sabitlenmiş Optimal Parametreler

Bu parametreler kapsamlı testler sonucunda en iyi sonuç verecek şekilde ayarlanmıştır:

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **Model** | `llama3.1:8b` | En iyi Türkçe performans/kaynak dengesi |
| **Top K** | `4` | BM25'ten alınacak en alakalı parça sayısı |
| **Temperature** | `0.15` | Düşük → tutarlı, güvenilir yanıtlar |
| **Max Gen Len** | `300` | Token bazında maksimum yanıt uzunluğu |
| **BM25 k1** | `1.5` | Terim frekansı doygunluk parametresi |
| **BM25 b** | `0.75` | Doküman uzunluğu normalizasyonu |

### Değiştirilebilir Ayarlar (Sidebar)

| Ayar | Varsayılan | Nerede |
|------|-----------|--------|
| Ollama URL | `http://127.0.0.1:11434/api/generate` | Sidebar → Bağlantı |
| Model Adı | `llama3.1:8b` | Sidebar → Model Adı |

### Neden Bu Değerler?

- **Temperature 0.15**: Çok düşük olursa (0.0) tekrarlayan yanıtlar, çok yüksek olursa (>0.5) uydurma bilgi riski artar. 0.15 en güvenilir dengeyi sağlar.
- **Top K 4**: 4 bilgi parçacığı + keyword enjeksiyonu ile genellikle 5-8 parça bağlam sağlanır. Daha fazlası prompt'u gereksiz uzatır.
- **Max Gen Len 300**: Detaylı yanıtlar için yeterli, gereksiz uzatma olmadan.
- **llama3.1:8b**: 3b modeli Türkçe'de yetersiz kalıyor, 8b modeli 16 GB RAM ile sorunsuz çalışıyor.

---

## 9. Bilgi Tabanı (Knowledge Base)

### Dosya Konumu

```
knowledge/reev_technical_index.json
```

### Yapı

```json
{
  "chunks": [
    {
      "chunk_id": "reev-fancy:konu-adi",
      "source": "Kaynak bilgisi",
      "text": "Konu ile ilgili detaylı metin..."
    }
  ]
}
```

### Mevcut Bilgi Parçacıkları (28 adet)

| # | Chunk ID | Konu |
|---|----------|------|
| 1 | `nedir` | Genel tanım ve özet |
| 2 | `genel-tanitim` | Araç genel tanıtımı |
| 3 | `uretici` | Reeder Teknoloji, Uygar Saral, Samsun |
| 4 | `tasarim` | Rıfat Baltaoğlu, tasarım detayları |
| 5 | `motor-gucu` | 6 kW elektrikli motor |
| 6 | `batarya` | 7.68 kWh LiFePO4 batarya |
| 7 | `menzil` | 100 km menzil |
| 8 | `sarj` | 5 saat, 220V ev tipi şarj |
| 9 | `boyutlar` | 2494×1334×1649 mm, 471 kg |
| 10 | `fiyat` | 475.000 TL (liste) / 445.000 TL (indirimli) |
| 11 | `vites-fren` | Otomatik vites, 4 teker disk fren |
| 12 | `ic-mekan` | Koltuk, panel, konfor özellikleri |
| 13 | `sogutma` | Çift fan, klima yok |
| 14 | `on-cam` | Buğu çözücü |
| 15 | `jant-lastik` | 13 inç jant, 145/70 R13 lastik |
| 16 | `sunroof-cam` | Manuel sunroof, otomatik camlar |
| 17 | `park` | Geri görüş kamerası, park sensörü |
| 18 | `aydinlatma` | LED farlar, dijital gösterge |
| 19 | `multimedya` | Dokunmatik ekran, Bluetooth, radyo |
| 20 | `avantajlar` | Ekonomik, çevre dostu, kolay kullanım |
| 21 | `tirmanma` | 30 derece tırmanma kapasitesi |
| 22 | `guvenlik` | Fren, kilit, izolasyon |
| 23 | `ehliyet` | B1 sınıfı, 16 yaş üzeri |
| 24 | `rakipler` | Citroen Ami karşılaştırma |
| 25 | `genel-ozellikler` | Tüm teknik özellikler özeti |
| 26 | `renk` | Beyaz, siyah, mavi |
| 27 | `govde-kasa` | Sac kasa, sertleştirilmiş plastik dodik |
| 28 | `teslim` | 20 gün teslim süresi |

### Güncel Teknik Veriler

| Özellik | Değer |
|---------|-------|
| Motor | 6 kW elektrikli |
| Batarya | 7,68 kWh LiFePO4 |
| Menzil | 100 km |
| Azami Hız | 45 km/s |
| Şarj | 5 saat (220V) |
| Liste Fiyatı | 475.000 ₺ |
| İndirimli Fiyat | 445.000 ₺ |
| Jant | 13 inç özel tasarım |
| Lastik | 145/70 R13 |
| Gövde | Sac kasa + sertleştirilmiş plastik dodik |
| Teslim Süresi | 20 gün |
| Boyutlar | 2494×1334×1649 mm |
| Ağırlık | 471 kg (700 kg max) |
| Kapasite | 2 kişi |
| Bagaj | 200 litre |
| Ehliyet | B1 (16+ yaş) |
| Sunroof | Evet (manuel) |
| Geri Görüş Kamerası | Evet |
| Park Sensörü | Evet |
| Multimedya | Radyo, MP3, Video, Bluetooth |

### Bilgi Tabanını Güncelleme

Yeni bilgi eklemek için `knowledge/reev_technical_index.json` dosyasını düzenleyin:

1. `chunks` dizisine yeni bir chunk ekleyin:
```json
{
  "chunk_id": "reev-fancy:yeni-konu",
  "source": "Kaynak bilgisi",
  "text": "Yeni konu ile ilgili detaylı metin"
}
```

2. `streamlit_app.py` içindeki anahtar kelime haritalarına yeni anahtar kelimeleri ekleyin:
   - `extractive_fallback_answer()` içindeki `keyword_to_topic`
   - `_KEYWORD_TO_TOPIC` sözlüğü

3. Uygulamayı yeniden başlatın.

---

## 10. Sorun Giderme (Troubleshooting)

### Ollama Bağlantı Hatası

**Sorun:** "Ollama yanıt veremedi" veya bağlantı hatası

**Çözüm:**
1. Ollama'nın çalıştığından emin olun (sistem tepsisini kontrol edin)
2. Terminal'de kontrol edin:
   ```powershell
   ollama list
   ```
3. Model yüklü değilse:
   ```powershell
   ollama pull llama3.1:8b
   ```
4. Sidebar'dan "🔌 Bağlantıyı Test Et" butonunu kullanın

### Yavaş Yanıt

**Sorun:** Yanıt üretimi çok uzun sürüyor (>60 saniye)

**Çözüm:**
- İlk soru genellikle modeli yüklemek için uzun sürer (20-40 sn), sonraki sorular daha hızlıdır
- RAM yetersizse Ollama yavaşlar. Task Manager'dan RAM kullanımını kontrol edin
- GPU destekli çalışıyorsanız VRAM yeterliliğini kontrol edin

### Port Çakışması

**Sorun:** "Port 8501 is not available"

**Çözüm:**
```powershell
# Farklı port kullanın
streamlit run streamlit_app.py --server.port 8502
```

### Boş/Kısa Yanıtlar

**Sorun:** Bot çok kısa veya boş yanıt veriyor

**Çözüm:**
- Ollama'nın düzgün çalıştığından emin olun
- Terminal'de doğrudan test edin:
  ```powershell
  ollama run llama3.1:8b "Merhaba, sen kimsin?"
  ```
- Yanıt geliyorsa sorun RAG sistemindedir, knowledge JSON'ı kontrol edin

### Hatalı Bilgi

**Sorun:** Bot yanlış bilgi veriyor

**Çözüm:**
- `knowledge/reev_technical_index.json` dosyasındaki verileri güncelleyin
- Uygulamayı yeniden başlatın
- Sohbeti "🗑️ Sohbeti Temizle" butonu ile temizleyin

---

## 11. SSS (Sıkça Sorulan Sorular)

### İnternet gerekli mi?
**Hayır.** Tüm işlemler lokalde çalışır. Sadece ilk Ollama model indirmesi için internet gerekir.

### Hangi bilgisayarlarda çalışır?
Minimum 12 GB RAM ve Intel i5 veya eşdeğeri işlemci yeterlidir. GPU opsiyoneldir.

### Verilerim güvende mi?
**Evet.** Hiçbir veri dış sunuculara gönderilmez. Tüm işlemler bilgisayarınızda gerçekleşir.

### Model değiştirilebilir mi?
Evet, sidebar'dan model adını değiştirebilirsiniz. Önerilen modeller:
- `llama3.1:8b` (varsayılan, en iyi denge)
- `llama3.2:3b` (daha hafif, daha az doğru)
- `qwen2.5:7b` (alternatif, iyi Türkçe performansı)

### Bilgi tabanına nasıl yeni bilgi eklenir?
Bkz. [Bilgi Tabanını Güncelleme](#bilgi-tabanını-güncelleme) bölümü.

### Yanıtlar neden bazen farklı oluyor?
Temperature parametresi 0.15 olduğundan az miktarda çeşitlilik olabilir. Bu normaldir ve yanıtların doğallığını artırır.

### "📄 Anahtar kelime eşleşmesi" ne demek?
LLM yanıt üretemediğinde (bağlantı hatası vb.) yedek sistem devreye girer ve bilgi tabanından doğrudan ilgili metni gösterir.

### Sohbet geçmişi kaydediliyor mu?
Sohbet geçmişi sadece mevcut oturum boyunca tarayıcı belleğinde tutulur. Sayfayı kapattığınızda silinir.

---

## Versiyon Bilgisi

| Bilgi | Değer |
|-------|-------|
| **Versiyon** | 2.0 (Final) |
| **Tarih** | 18 Şubat 2026 |
| **Model** | llama3.1:8b |
| **Bilgi Parçacığı** | 28 adet |
| **Güncel Fiyat** | 445.000 ₺ (indirimli) |
| **Test Sonucu** | 10/10 başarılı |

---

*Bu döküman ReeV Fancy Teknik Chatbot projesi için hazırlanmıştır.*
