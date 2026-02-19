# Reev Fancy Chatbot Kurulumu

Bu repo, Llama 3 ile yerel bir RAG chatbot calistirmak icin guncellendi.

## 1. Bilgiyi ekle

`knowledge/` klasorune Reev Fancy ile ilgili dosyalari koyun.
Desteklenen formatlar:
- `.txt`
- `.md`
- `.rst`
- `.csv`
- `.json`
- `.jsonl`
- `.html`
- `.htm`

## 2. Bilgi indeksini olustur

```bash
python ingest.py --knowledge_dir knowledge --output_path knowledge/index.json
```

## 3. Chatbotu calistir

```bash
python chatbot.py \
  --ckpt_dir Meta-Llama-3-8B-Instruct \
  --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
  --index_path knowledge/index.json
```

## 4. Web arayuzu (Streamlit)

```bash
streamlit run streamlit_app.py
```

Varsayilan olarak teknik index yolu:
- `knowledge/reev_technical_index.json`

Alternatif olarak, dagitik calistirma icin:

```bash
torchrun --nproc_per_node 1 chatbot.py \
  --ckpt_dir Meta-Llama-3-8B-Instruct \
  --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
  --index_path knowledge/index.json
```

## Notlar

- Bu repo CUDA tabanli Llama 3 inference bekler.
- Cevaplar yalnizca indexteki baglama dayanir.
- Bilgi bulunamazsa asistan "Bu bilgiye sahip degilim" demelidir.
