# ICORIS Q&A — Asisten Tanya-Jawab Live untuk Presentasi Zoom

Asisten real-time yang **mendengar** audio dari Zoom (atau aplikasi apa pun),
**menuliskan** pertanyaan audiens memakai speech-to-text lokal (faster-whisper),
lalu menampilkan **poin jawaban singkat** di jendela overlay yang selalu di atas
layar — supaya presenter tinggal melirik dan menjawab.

Dibuat untuk mendampingi presentasi paper
*"YOLOv8-Based River Debris Monitoring System With Meteorological Data for Flood Mitigation"*
di ICORIS 2026.

```
Audio Zoom (WASAPI loopback)
      -> faster-whisper (lokal, GPU/CPU)   = ubah suara jadi teks
      -> LLM router (Groq -> Groq -> Gemini) = buat poin jawaban dari konteks paper
      -> jendela overlay always-on-top      = tampil ringkas untuk presenter
```

---

## Daftar Isi
1. [Cara Kerja Singkat](#1-cara-kerja-singkat)
2. [Kebutuhan Sistem](#2-kebutuhan-sistem)
3. [Instalasi](#3-instalasi)
4. [Konfigurasi API Key (.env)](#4-konfigurasi-api-key-env)
5. [Menyiapkan Folder `konteks`](#5-menyiapkan-folder-konteks)
6. [Menjalankan Aplikasi](#6-menjalankan-aplikasi)
7. [Cara Pemakaian saat Presentasi](#7-cara-pemakaian-saat-presentasi)
8. [Pengaturan Lanjutan (.env)](#8-pengaturan-lanjutan-env)
9. [Troubleshooting](#9-troubleshooting)
10. [Catatan Privasi & Biaya](#10-catatan-privasi--biaya)

---

## 1. Cara Kerja Singkat

- **Mendengar**: aplikasi merekam suara yang keluar dari *speaker default* PC
  (loopback WASAPI). Jadi apa pun yang terdengar dari Zoom ikut terekam —
  termasuk suara audiens yang bertanya.
- **Transkripsi**: audio diubah jadi teks oleh **faster-whisper** yang berjalan
  **lokal** di komputer (pakai GPU NVIDIA kalau ada, kalau tidak otomatis pakai CPU).
- **Jawaban**: teks pertanyaan dikirim ke LLM (Groq/Gemini) bersama **konteks paper**.
  LLM hanya menjawab dari isi folder `konteks`, dan membalas maksimal 3 poin singkat.
- **Tampilan**: jawaban muncul di jendela kecil yang selalu di atas layar.

---

## 2. Kebutuhan Sistem

| Komponen | Keterangan |
|----------|------------|
| OS | Windows 10/11 (audio loopback memakai WASAPI/MediaFoundation) |
| Python | **3.10** (disarankan 3.10–3.12) |
| RAM | Minimal 8 GB |
| GPU (opsional) | NVIDIA (mis. RTX 2050) untuk transkripsi lebih cepat. Tanpa GPU tetap jalan di CPU |
| Internet | Diperlukan untuk LLM cloud (Groq/Gemini) |
| Akun API | Groq (gratis) dan/atau Gemini (Google AI Studio, gratis) |

---

## 3. Instalasi

### 3.1. Pasang Python
Unduh & pasang Python 3.10+ dari <https://www.python.org/downloads/>.
Saat instalasi, centang **"Add Python to PATH"**.

Cek di terminal (PowerShell / CMD):
```powershell
python --version
```

### 3.2. Ambil kode proyek
Salin seluruh folder proyek `icoris-qa` ke komputer, lalu buka terminal di folder itu:
```powershell
cd "D:\zoom\voice zoom\icoris-qa"
```
> Ganti path sesuai lokasi folder di komputer Anda.

### 3.3. (Disarankan) Buat virtual environment
Supaya paket tidak tercampur dengan Python sistem:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> Kalau PowerShell menolak menjalankan script, jalankan sekali:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` lalu ulangi Activate.

### 3.4. Pasang dependensi
```powershell
pip install -r requirements.txt
```
Isi `requirements.txt`:
```
faster-whisper
soundcard>=0.4.3
openai
python-dotenv
numpy
pypdf
```

### 3.5. (Opsional) Aktifkan percepatan GPU NVIDIA
Kalau punya GPU NVIDIA dan ingin transkripsi lebih cepat, pasang runtime CUDA 12:
```powershell
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```
Aplikasi otomatis mendeteksi dan mendaftarkan DLL CUDA ini saat start.
**Tanpa GPU tidak perlu langkah ini** — program otomatis memakai CPU.

---

## 4. Konfigurasi API Key (.env)

Aplikasi memakai LLM cloud. Buat file `.env` dari contoh:
```powershell
copy .env.example .env
```
Lalu buka `.env` dengan editor teks dan isi API key Anda.

### Cara mendapatkan API key (gratis)
- **Groq** (utama, cepat): daftar di <https://console.groq.com> → *API Keys* → *Create*.
  Key diawali `gsk_...`.
- **Gemini** (cadangan): buka <https://aistudio.google.com/app/apikey> → *Create API key*.

Isi ke `.env`:
```dotenv
# Urutan fallback: Groq 120b -> Groq 20b -> Gemini
BACKEND_ORDER=groq,groq2,gemini

# Groq: satu key untuk dua model (kuota rate-limit terpisah per model)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=openai/gpt-oss-120b
GROQ_MODEL_2=openai/gpt-oss-20b

# Gemini (Google AI Studio)
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_MODEL=gemini-3.5-flash
GEMINI_REASONING=none
```
> **Penting:** jangan bagikan file `.env` ke siapa pun — isinya kunci rahasia.
> File ini sudah masuk `.gitignore` supaya tidak ikut ter-commit ke Git.

Anda tidak harus mengisi semuanya. Minimal **satu** backend berfungsi sudah cukup;
backend yang tidak diisi API key-nya otomatis dilewati.

---

## 5. Menyiapkan Folder `konteks`

Folder `konteks/` berisi materi paper yang jadi sumber jawaban. LLM **hanya**
menjawab dari isi folder ini.

Format yang **didukung dan dibaca**: `.md`, `.txt`, `.pdf`
(dibaca urut nama file, jadi beri awalan angka seperti `01_`, `02_`).

Contoh isi saat ini:
```
konteks/
  01_qa_persiapan.md        <- daftar Q&A yang sudah disiapkan
  02_ringkasan_paper.md     <- ringkasan metode & hasil paper
```

> **Catatan penting:** file **`.docx` TIDAK dibaca** oleh aplikasi. Kalau materi Anda
> berupa `.docx`, ubah dulu ke `.pdf` (Save As -> PDF) atau salin isinya ke file `.md`/`.txt`.
> Semakin ringkas & relevan isinya, semakin baik dan hemat kuota jawabannya.

Aplikasi otomatis memilih bagian konteks yang paling relevan dengan tiap
pertanyaan (hemat token), jadi tidak masalah kalau konteksnya cukup panjang.

---

## 6. Menjalankan Aplikasi

Pastikan berada di folder proyek (dan venv aktif kalau memakainya), lalu:
```powershell
python qa_assistant.py
```

Saat pertama kali dijalankan, faster-whisper akan **mengunduh model** (`small.en`)
otomatis — butuh internet dan beberapa saat. Setelah itu tersimpan di cache.

Jika berhasil, di terminal akan muncul kira-kira:
```
Urutan backend: Groq-120b(openai/gpt-oss-120b) -> Groq-20b(openai/gpt-oss-20b) -> Gemini(gemini-3.5-flash)
Konteks dimuat: 8111 karakter
```
dan sebuah jendela kecil **"ICORIS Q&A"** muncul di atas semua jendela.

---

## 7. Cara Pemakaian saat Presentasi

1. **Atur audio Zoom ke speaker** yang sama dengan speaker default Windows
   (aplikasi merekam speaker default). Kalau pakai headset, jadikan headset itu
   sebagai *Default Playback Device* di Windows.
2. **Jalankan** `python qa_assistant.py` sebelum sesi tanya-jawab.
3. Saat audiens bertanya lewat Zoom:
   - Pertanyaan muncul sebagai teks `Q: ...` (dengan `...` = masih berjalan,
     `[selesai]` = pertanyaan dianggap selesai).
   - Beberapa detik kemudian, poin jawaban muncul di bawahnya (maksimal 3 bullet `• `).
   - Baris status menampilkan backend yang menjawab, mis. `FINAL | Groq-120b`.
4. **Uji manual (tanpa suara):** ketik pertanyaan di kotak input bawah lalu tekan
   **Enter**. Berguna untuk latihan sebelum presentasi.

### Arti label di layar
| Tampilan | Arti |
|----------|------|
| `Q: ... ...` | Transkrip sementara, audiens masih bicara |
| `Q: ...  [selesai]` | Pertanyaan dianggap selesai, jawaban final dibuat |
| `Q: ...  [manual]` | Pertanyaan yang Anda ketik sendiri |
| `DRAFT \| Groq-120b` | Jawaban draft (cepat, saat pertanyaan belum selesai) |
| `FINAL \| Groq-120b` | Jawaban final dari backend tersebut |

### Format jawaban
- Maksimal **3 poin**, tiap poin ≤20 kata, bahasa Inggris, tanpa markdown.
- Kalau konteks tidak memuat jawabannya, asisten mengatakannya dengan jujur
  (mis. menyarankan "*a limitation we plan to address in future work*").
- Asisten **tidak mengarang** angka/metrik yang tidak ada di konteks.

---

## 8. Pengaturan Lanjutan (.env)

Semua bisa diubah tanpa menyentuh kode. Nilai default ada di kurung.

| Variabel | Fungsi |
|----------|--------|
| `BACKEND_ORDER` (`groq,groq2,gemini`) | Urutan backend dicoba. Bisa tambah `local`. |
| `GROQ_MODEL` (`openai/gpt-oss-120b`) | Model Groq utama |
| `GROQ_MODEL_2` (`openai/gpt-oss-20b`) | Model Groq cadangan (kuota terpisah) |
| `GEMINI_MODEL` (`gemini-3.5-flash`) | Model Gemini |
| `WHISPER_DEVICE` (`cuda`) | `cuda` = GPU NVIDIA, `cpu` = tanpa GPU |
| `WHISPER_MODEL` (`small.en`) | Ukuran model ASR: `base.en` (ringan/cepat), `small.en`, `medium.en` (akurat) |
| `SILENCE_RMS` (`0.01`) | Ambang suara. Naikkan (mis. `0.02`) kalau noise ikut terdeteksi |
| `MAX_CONTEXT_CHARS` (`12000`) | Batas total konteks yang dimuat |
| `CONTEXT_BUDGET_CHARS` (`6000`) | Konteks yang benar-benar dikirim per pertanyaan (hemat token) |
| `MAX_TOKENS` (`200`) | Panjang maksimal jawaban LLM |
| `RATE_LIMIT_COOLDOWN` (`60`) | Detik istirahat backend saat kena limit (429) |
| `ERROR_COOLDOWN` (`15`) | Detik istirahat backend saat error koneksi |

**Menjalankan tanpa GPU:** set di `.env`
```dotenv
WHISPER_DEVICE=cpu
WHISPER_MODEL=base.en
```

**Menambah LLM lokal (LM Studio):** jalankan server LM Studio, lalu di `.env`
tambahkan `local` ke urutan dan sesuaikan:
```dotenv
BACKEND_ORDER=groq,groq2,gemini,local
LOCAL_BASE_URL=http://localhost:1234/v1
LOCAL_MODEL=granite-4.0-micro
```

---

## 9. Troubleshooting

**`RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`**
Runtime CUDA belum terpasang. Pilih salah satu:
- Pasang: `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`, atau
- Pakai CPU: set `WHISPER_DEVICE=cpu` di `.env`.

**Jendela muncul tapi status "Semua backend gagal. Cek internet / kuota API."**
- Cek koneksi internet.
- Cek API key di `.env` benar dan masih aktif.
- Kalau muncul `model '...' tidak ada`, nama model sudah usang — perbarui
  `GROQ_MODEL`/`GEMINI_MODEL` sesuai model terbaru di dashboard provider.
- Kalau muncul `kena limit (429)`, kuota per menit habis; tunggu sebentar,
  otomatis pindah ke backend berikutnya.

**Tidak ada pertanyaan yang terdeteksi / `Q:` tidak berubah**
- Pastikan audio Zoom keluar ke **speaker default** Windows.
- Coba naikkan volume, atau turunkan `SILENCE_RMS` (mis. `0.008`).
- Peringatan `SoundcardRuntimeWarning: data discontinuity in recording` itu
  **normal** (bukan error), boleh diabaikan.

**`RuntimeError: Error 0x88890004` saat merekam**
Perangkat audio berubah/terputus di tengah rekaman (mis. ganti headset atau Zoom
mengambil alih perangkat). Jalankan ulang aplikasi setelah perangkat audio stabil.

**Transkripsi lambat**
Pakai GPU (lihat 3.5), atau turunkan model ke `base.en` di `.env`.

**Materi `.docx` tidak terbaca**
Ubah ke `.pdf` atau salin isinya ke file `.md`/`.txt` di folder `konteks`.

---

## 10. Catatan Privasi & Biaya

- **Transkripsi** berjalan **lokal** — audio tidak dikirim ke mana pun untuk diubah jadi teks.
- **Teks pertanyaan + potongan konteks paper** dikirim ke penyedia LLM (Groq/Gemini)
  untuk membuat jawaban. Jangan taruh informasi rahasia di folder `konteks`.
- Groq dan Gemini punya **tier gratis** dengan batas kuota per menit. Aplikasi
  sudah dirancang hemat token (konteks dipangkas, jawaban dibatasi ~200 token).
- Jaga kerahasiaan file `.env`. Jangan unggah ke tempat publik.

---

Selamat mencoba, semoga presentasinya lancar! 🎤
