"""build_qa_dataset.py - Generate jawaban untuk 300 pertanyaan ICORIS memakai
router LLM di qa_assistant.py (grounded ke konteks paper), lalu simpan sebagai
dataset Q&A.

Jalankan:
  # pakai backend yang aktif (mis. Groq lalu Gemini)
  set BACKEND_ORDER=groq,groq2,gemini   (atau claude,... bila key workspace-scoped siap)
  python build_qa_dataset.py

Fitur:
  - Grounding: memuat konteks paper (load_context) & memilih bagian relevan
    per pertanyaan (trim_context) persis seperti aplikasi live.
  - Resume: melewati pertanyaan yang jawabannya sudah ada di file output.
  - Tulis UTF-8 (aman untuk karakter khusus), incremental (tahan interupsi).
  - Jeda antar-request untuk menghormati rate limit free tier.

Keluaran (folder dataset/):
  - qa_icoris_356.jsonl : {id, question, answer, backend, priority, group_id, group}
  - qa_icoris_356.csv
"""
import csv
import json
import os
import sys
import time

# Pastikan stdout tidak crash di konsol Windows (cp1252) saat mencetak progres.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import qa_assistant as qa

IN_JSONL = os.path.join("dataset", "pertanyaan_icoris_356.jsonl")
OUT_JSONL = os.path.join("dataset", "qa_icoris_356.jsonl")
OUT_CSV = os.path.join("dataset", "qa_icoris_356.csv")

# Jeda antar pertanyaan (detik) untuk menghormati rate limit token/menit.
SLEEP_BETWEEN = float(os.getenv("QA_SLEEP", "2.0"))

# Bahasa jawaban dataset: "id" (default) atau "en".
# Aplikasi live sengaja menjawab dalam bahasa Inggris (untuk presentasi ICORIS).
# Untuk dataset latihan presenter Indonesia, default kita Bahasa Indonesia.
ANSWER_LANG = os.getenv("QA_LANG", "id").lower()

# Prompt khusus dataset: sama grounding-nya dengan aplikasi, tapi jawaban ID
# dan sedikit lebih lengkap (untuk bahan latihan, bukan overlay live).
SYSTEM_PROMPT_ID = """Anda membantu Ferdy menyiapkan jawaban untuk sesi tanya jawab
paper "YOLOv8-Based River Debris Monitoring System With Meteorological Data for
Flood Mitigation" di ICORIS 2026.

ATURAN:
- Jawab HANYA berdasarkan KONTEKS PAPER di bawah. Jangan mengarang angka, metrik,
  atau hasil. Bila konteks tidak memuat jawaban, katakan singkat dan tambahkan satu
  kalimat jujur seperti "Ini keterbatasan yang kami rencanakan untuk diperbaiki pada
  penelitian lanjutan."
- Jawab langsung dalam Bahasa Indonesia, maksimal 4 poin bullet, tiap poin satu
  kalimat utuh maksimal 28 kata. Teks polos, tanpa markdown/tebal/judul.
- Setiap baris diawali "\u2022 ". Jangan menulis apa pun sebelum bullet pertama.
- Sertakan angka konkret dari konteks bila relevan (mis. mAP50 0,789; 89,3 FPS).
- Jangan mengulang atau mengutip pertanyaan.

=== KONTEKS PAPER ===
{context}"""


def load_questions():
    with open(IN_JSONL, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done():
    """Kembalikan dict id->record untuk pertanyaan yang jawabannya sudah ada
    (agar bisa resume tanpa mengulang)."""
    done = {}
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("answer"):
                    done[r["id"]] = r
    return done


def answer_one(llm, context, question):
    """Panggil router untuk satu pertanyaan (grounded), kembalikan
    (answer_text, backend_name)."""
    tctx = qa.trim_context(context, question) if context else ""
    prompt_tmpl = SYSTEM_PROMPT_ID if ANSWER_LANG == "id" else qa.SYSTEM_PROMPT
    system = prompt_tmpl.format(context=tctx or "(no context loaded)")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Complete question: " + question},
    ]
    buf, backend = [], ""
    for kind, val in llm.stream(messages):
        if kind == "backend":
            backend = val
        elif kind == "token":
            buf.append(val)
        elif kind == "error":
            # error sebelum ada jawaban -> catat sebagai info, biarkan router pindah
            if not backend:
                print("   [info]", val)
    answer = qa._display_answer("".join(buf))
    return answer, backend


def write_outputs(records):
    """Tulis ulang JSONL + CSV terurut berdasarkan id."""
    ordered = sorted(records.values(), key=lambda r: r["id"])
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "question", "answer", "backend", "priority", "group_id", "group"])
        w.writeheader()
        for r in ordered:
            w.writerow(r)


def main():
    questions = load_questions()
    done = load_done()
    print(f"Total pertanyaan: {len(questions)} | sudah ada jawaban: {len(done)}")

    context = qa.load_context()
    print(f"Konteks paper dimuat: {len(context)} karakter")
    llm = qa.LLM()

    records = dict(done)  # mulai dari yang sudah selesai
    todo = [q for q in questions if q["id"] not in done]
    print(f"Akan diproses: {len(todo)} pertanyaan\n")

    for i, q in enumerate(todo, start=1):
        qid = q["id"]
        answer, backend = answer_one(llm, context, q["question"])
        records[qid] = {
            "id": qid,
            "question": q["question"],
            "answer": answer,
            "backend": backend,
            "priority": q["priority"],
            "group_id": q["group_id"],
            "group": q["group"],
        }
        status = backend or "GAGAL"
        preview = answer.replace("\n", " ")[:80] if answer else "(kosong)"
        print(f"[{i}/{len(todo)}] #{qid:03d} {status:9s} | {preview}")

        # Tulis incremental tiap 5 pertanyaan (dan di akhir) agar tahan interupsi.
        if i % 5 == 0:
            write_outputs(records)
        time.sleep(SLEEP_BETWEEN)

    write_outputs(records)
    filled = sum(1 for r in records.values() if r["answer"])
    print(f"\nSelesai. Jawaban terisi: {filled}/{len(questions)}")
    print(f"JSONL -> {OUT_JSONL}")
    print(f"CSV   -> {OUT_CSV}")


if __name__ == "__main__":
    main()
