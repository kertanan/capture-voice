"""retriever.py - Pencarian semantik pertanyaan ICORIS berbasis FAISS.

Ide (sesuai permintaan): daripada menerjemahkan seluruh dataset lebih dulu,
kita simpan dataset dalam Bahasa Indonesia lalu:
  1. Bangun index FAISS dari embedding 300 pertanyaan (model multilingual,
     sehingga pertanyaan lisan berbahasa Inggris tetap cocok ke pertanyaan ID).
  2. Saat ada pertanyaan masuk (mis. hasil transkrip Whisper), embed pertanyaan
     itu -> cari tetangga terdekat di FAISS -> kembalikan Q&A paling mirip.
  3. Terjemahan dilakukan ON-THE-FLY hanya bila diminta (mis. answer_lang="en"),
     bukan menerjemahkan seluruh dataset di awal.

Dipakai sebagai library:
    from retriever import QARetriever
    r = QARetriever()                      # muat dataset + index (auto-build/cached)
    hit = r.search("what is mAP50?")       # -> dict {id, question, answer, score, ...}
    ans = r.get_answer("what is mAP50?", answer_lang="id", min_score=0.45)

CLI cepat:
    python retriever.py "apa arti mAP50"
    python retriever.py --build           # paksa bangun ulang index
"""
import json
import os
import sys

import numpy as np

DATASET = os.path.join("dataset", "qa_icoris_356.jsonl")
INDEX_PATH = os.path.join("dataset", "faiss_index.bin")
META_PATH = os.path.join("dataset", "faiss_meta.json")
EMB_MODEL = os.getenv("EMB_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


class QARetriever:
    def __init__(self, dataset=DATASET, model_name=EMB_MODEL, rebuild=False):
        import faiss  # impor lokal agar import modul tetap ringan
        from sentence_transformers import SentenceTransformer

        self.faiss = faiss
        self.dataset_path = dataset
        self.model = SentenceTransformer(model_name)
        self.records = self._load_records(dataset)
        # Hanya index pertanyaan yang punya jawaban (agar hasil selalu berguna).
        self.indexable = [r for r in self.records if r.get("answer")]
        self.questions = [r["question"] for r in self.indexable]

        if not rebuild and self._cache_valid():
            self._load_index()
        else:
            self._build_index()

    # ---------- data ----------
    @staticmethod
    def _load_records(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _cache_valid(self):
        """Cache valid bila file index+meta ada dan jumlah pertanyaan cocok
        dengan dataset saat ini (mencegah index basi)."""
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            return False
        try:
            with open(META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
            return (meta.get("model") == self.model.get_sentence_embedding_dimension()
                    or meta.get("count") == len(self.indexable))
        except Exception:
            return False

    # ---------- index ----------
    def _embed(self, texts):
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False),
            dtype="float32",
        )

    def _build_index(self):
        emb = self._embed(self.questions)
        dim = emb.shape[1]
        # Inner product pada vektor ternormalisasi == cosine similarity.
        self.index = self.faiss.IndexFlatIP(dim)
        self.index.add(emb)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        self.faiss.write_index(self.index, INDEX_PATH)
        # Simpan urutan id agar posisi FAISS -> record konsisten setelah reload.
        meta = {
            "count": len(self.indexable),
            "dim": dim,
            "ids": [r["id"] for r in self.indexable],
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _load_index(self):
        self.index = self.faiss.read_index(INDEX_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        # Susun ulang self.indexable sesuai urutan id di meta (jaga konsistensi
        # posisi vektor terhadap record).
        by_id = {r["id"]: r for r in self.indexable}
        self.indexable = [by_id[i] for i in meta["ids"] if i in by_id]
        self.questions = [r["question"] for r in self.indexable]

    # ---------- query ----------
    def search(self, query, k=1):
        """Kembalikan list hasil teratas: [{score, id, question, answer, ...}]."""
        qe = self._embed([query])
        scores, idxs = self.index.search(qe, k)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            rec = dict(self.indexable[i])
            rec["score"] = float(score)
            out.append(rec)
        return out

    def get_answer(self, query, answer_lang="id", min_score=0.45, translator=None):
        """Ambil jawaban paling mirip. Bila similarity < min_score, kembalikan
        None (biar pemanggil fallback ke LLM live). answer_lang="en" +
        translator opsional -> terjemahkan on-the-fly."""
        hits = self.search(query, k=1)
        if not hits or hits[0]["score"] < min_score:
            return None
        hit = hits[0]
        answer = hit["answer"]
        if answer_lang == "en":
            answer = (translator or _default_translate)(answer, "en")
        hit["answer_out"] = answer
        return hit


# ---------- terjemahan on-the-fly (opsional) ----------
def _default_translate(text, target_lang):
    """Terjemahan ringan memakai LLM router qa_assistant bila tersedia.
    Bila gagal, kembalikan teks asli (aman: jangan crash)."""
    try:
        import qa_assistant as qa
        llm = getattr(_default_translate, "_llm", None)
        if llm is None:
            llm = qa.LLM()
            _default_translate._llm = llm
        tgt = "English" if target_lang == "en" else "Indonesian"
        msgs = [
            {"role": "system", "content": f"Translate to {tgt}. Output only the "
             "translation, keep the bullet format and numbers unchanged."},
            {"role": "user", "content": text},
        ]
        buf = []
        for kind, val in llm.stream(msgs):
            if kind == "token":
                buf.append(val)
        out = "".join(buf).strip()
        return out or text
    except Exception:
        return text


def _cli():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = [a for a in sys.argv[1:]]
    rebuild = "--build" in args
    args = [a for a in args if a != "--build"]
    r = QARetriever(rebuild=rebuild)
    print(f"Index siap: {len(r.indexable)} pertanyaan berjawaban.")
    if not args:
        print('Pakai: python retriever.py "pertanyaan kamu"')
        return
    query = " ".join(args)
    hits = r.search(query, k=3)
    print(f"\nQuery: {query}\n")
    for h in hits:
        print(f"[score {h['score']:.3f}] #{h['id']:03d}  {h['question']}")
        print(f"    -> {h['answer'][:160]}\n")


if __name__ == "__main__":
    _cli()
