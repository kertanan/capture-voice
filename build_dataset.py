"""build_dataset.py - Ekstrak 300 pertanyaan dari
konteks/Bank_Pertanyaan_Presentasi_ICORIS_356.pdf menjadi dataset terstruktur.

Keluaran (folder dataset/):
  - pertanyaan_icoris_356.jsonl : satu objek JSON per baris (untuk LLM/pipeline)
  - pertanyaan_icoris_356.csv   : tabel (untuk spreadsheet)

Setiap record berisi:
  id        : nomor pertanyaan (int, 1..300)
  question  : teks pertanyaan (sudah dibersihkan dari line-break PDF)
  priority  : True bila pertanyaan bertanda "P" (30 pertanyaan prioritas)
  group_id  : nomor kelompok (1..25)
  group     : nama kelompok pertanyaan
"""
import csv
import json
import os
import re

from pypdf import PdfReader

PDF = os.path.join("konteks", "Bank_Pertanyaan_Presentasi_ICORIS_356.pdf")
OUT_DIR = "dataset"

# Peta kelompok (dari halaman "Peta pertanyaan" di PDF): rentang nomor -> nama.
GROUPS = [
    (1, 12, "Pembukaan host dan gambaran penelitian"),
    (13, 24, "Dasar masalah dan hubungan dengan banjir"),
    (25, 36, "Kebaruan dan posisi ilmiah"),
    (37, 48, "Tujuan ruang lingkup dan definisi keluaran"),
    (49, 60, "Asal komposisi dan keterwakilan dataset"),
    (61, 72, "Anotasi taksonomi dan kualitas label"),
    (73, 84, "Pembagian data dan pencegahan kebocoran"),
    (85, 96, "Preprocessing augmentasi dan resolusi"),
    (97, 108, "Pemilihan model dan arsitektur"),
    (109, 120, "Pelatihan pemilihan checkpoint dan reproduksibilitas"),
    (121, 132, "Metrik deteksi dan interpretasi hasil"),
    (133, 144, "Hasil per kelas dan kegagalan deteksi"),
    (145, 156, "Kecepatan komputasi dan cakupan pengukuran"),
    (157, 168, "Perbandingan dengan penelitian sebelumnya"),
    (169, 180, "Integrasi CCTV dan arsitektur sistem"),
    (181, 192, "Sumber cuaca makna curah hujan dan keselarasan waktu"),
    (193, 204, "Dasar SAW dan bobot jenis sampah"),
    (205, 216, "Normalisasi luas frame dan ketidaksesuaian naskah"),
    (217, 228, "Ambang SAW dan pertanyaan matematika"),
    (229, 240, "Desain validasi SAW dan penyebab kegagalan"),
    (241, 252, "Metrik SAW baseline dan ketidakpastian"),
    (253, 264, "Konsistensi klaim naskah dan batas bukti"),
    (265, 276, "Eksperimen perbaikan dan penelitian berikutnya"),
    (277, 288, "Penggunaan operasional biaya dan tanggung jawab"),
    (289, 300, "Pertanyaan penutup kontribusi penulis dan kritik tajam"),
]


def group_for(n):
    for gi, (lo, hi, name) in enumerate(GROUPS, start=1):
        if lo <= n <= hi:
            return gi, name
    return None, None


def extract_questions(pdf_path):
    reader = PdfReader(pdf_path)
    full = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Normalkan spasi di setiap baris & buang footer "Paper 356 | N".
    raw_lines = []
    for ln in full.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if re.match(r"^Paper 356\s*\|\s*\d+\s*$", ln):
            continue
        raw_lines.append(ln)

    # Pertanyaan dimulai dengan nomor 3 digit (001..300), boleh diikuti "P".
    # Teks bisa terpotong ke beberapa baris -> gabungkan sampai nomor berikutnya.
    start_re = re.compile(r"^(\d{3})\s+(P\s+)?(.*)$")
    items = {}          # nomor -> {priority, text_parts}
    current = None
    for ln in raw_lines:
        m = start_re.match(ln)
        # Anggap awal pertanyaan HANYA jika nomornya di rentang 1..300 dan
        # merupakan urutan yang masuk akal (hindari salah tangkap angka dalam teks).
        if m and 1 <= int(m.group(1)) <= 300:
            num = int(m.group(1))
            priority = bool(m.group(2))
            text = m.group(3).strip()
            items[num] = {"priority": priority, "parts": [text] if text else []}
            current = num
        elif current is not None:
            # Baris lanjutan dari pertanyaan berjalan. Hentikan penggabungan bila
            # baris jelas berupa judul kelompok (mis. "03 Kebaruan ...") atau header.
            if re.match(r"^\d{2}\s+[A-Z]", ln):  # "03 Kebaruan dan posisi ilmiah"
                current = None
                continue
            items[current]["parts"].append(ln)

    records = []
    for num in sorted(items):
        text = " ".join(items[num]["parts"])
        text = re.sub(r"\s+", " ", text).strip()
        gid, gname = group_for(num)
        records.append({
            "id": num,
            "question": text,
            "priority": items[num]["priority"],
            "group_id": gid,
            "group": gname,
        })
    return records


def main():
    records = extract_questions(PDF)
    os.makedirs(OUT_DIR, exist_ok=True)

    jsonl_path = os.path.join(OUT_DIR, "pertanyaan_icoris_356.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = os.path.join(OUT_DIR, "pertanyaan_icoris_356.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "priority", "group_id", "group"])
        w.writeheader()
        for r in records:
            w.writerow(r)

    # Ringkasan / validasi.
    total = len(records)
    prio = sum(1 for r in records if r["priority"])
    empty = [r["id"] for r in records if not r["question"]]
    ids = [r["id"] for r in records]
    missing = [n for n in range(1, 301) if n not in ids]

    print(f"Total pertanyaan  : {total}")
    print(f"Pertanyaan prioritas P: {prio}")
    print(f"ID hilang         : {missing if missing else 'tidak ada'}")
    print(f"Teks kosong       : {empty if empty else 'tidak ada'}")
    print(f"JSONL -> {jsonl_path}")
    print(f"CSV   -> {csv_path}")
    print("\nContoh 3 record pertama:")
    for r in records[:3]:
        print(" ", json.dumps(r, ensure_ascii=False))
    print("Contoh record prioritas:")
    for r in records:
        if r["priority"]:
            print(" ", json.dumps(r, ensure_ascii=False))
            break


if __name__ == "__main__":
    main()
