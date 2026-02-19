from __future__ import annotations

# ── MUST be first: force loopback BEFORE any torch import ──────────
import os

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
# ────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from llama import Llama


def ensure_single_process_env() -> None:
    # Force safe local defaults; existing global env values can break local runs.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


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


def load_chunks(index_path: Path) -> List[Dict[str, str]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        raise ValueError(f"No chunks found in {index_path}")
    return chunks


def build_system_prompt(retrieved: List[Tuple[Dict[str, str], float]], assistant_scope: str) -> str:
    context_blocks: List[str] = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Kaynak {i}] {chunk['source']} ({chunk['chunk_id']}, skor={score:.2f})\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "Hic baglam bulunamadi."

    if assistant_scope == "technical":
        rules = (
            "Sen sadece Reev Fancy teknik detaylarini anlatan bir asistansin.\n"
            "Sadece teknik icerik cevapla (ozellik, olcu, performans, malzeme, elektrik, montaj, bakim vb.).\n"
            "Teknik olmayan sorulara: 'Bu bot sadece teknik detay verir.' de.\n"
            "Sadece verilen baglama dayanarak cevap ver.\n"
            "Baglamda acikca yoksa: 'Bu bilgiye sahip degilim.' de.\n"
            "Uydurma yapma, net ve kisa ol.\n\n"
        )
    else:
        rules = (
            "Sen Reev Fancy hakkinda bilgi veren bir asistansin.\n"
            "Sadece verilen baglama dayanarak cevap ver.\n"
            "Baglamda acikca gecmiyorsa: 'Bu bilgiye sahip degilim.' de.\n"
            "Uydurma yapma, net ve kisa ol.\n\n"
        )

    return f"{rules}Baglam:\n{context}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local RAG chatbot with Llama 3.")
    parser.add_argument("--ckpt_dir", required=True, help="Path to model checkpoint directory.")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer.model.")
    parser.add_argument("--index_path", default="knowledge/index.json", help="Path to the knowledge index.")
    parser.add_argument("--top_k", type=int, default=4, help="How many chunks to retrieve.")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--history_turns", type=int, default=3, help="How many previous turns to include.")
    parser.add_argument(
        "--assistant_scope",
        choices=["general", "technical"],
        default="general",
        help="Assistant behavior scope.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_single_process_env()

    chunks = load_chunks(Path(args.index_path))
    retriever = BM25Retriever(chunks)
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    history: List[Dict[str, str]] = []
    print("Reev Fancy chatbot hazir. Cikmak icin 'exit' yazin.")

    while True:
        question = input("\nSen: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Gorusuruz.")
            break

        retrieved = retriever.retrieve(question, k=args.top_k)
        system_prompt = build_system_prompt(retrieved, assistant_scope=args.assistant_scope)

        window = history[-(args.history_turns * 2) :] if args.history_turns > 0 else []
        dialog = [{"role": "system", "content": system_prompt}, *window, {"role": "user", "content": question}]
        result = generator.chat_completion(
            [dialog],
            temperature=args.temperature,
            top_p=args.top_p,
            max_gen_len=args.max_gen_len,
        )[0]["generation"]["content"]

        print(f"Asistan: {result}")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()
