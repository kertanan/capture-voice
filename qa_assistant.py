"""
qa_assistant.py - Asisten Q&A real-time untuk presentasi Zoom (ICORIS)

Alur:
  Audio Zoom (WASAPI loopback) -> faster-whisper (lokal) -> LLM router
  (Groq gpt-oss-120b -> Groq gpt-oss-20b -> Gemini) -> jendela overlay always-on-top

Jalankan:  python qa_assistant.py
"""
import glob
import os
import queue
import re
import threading
import time
import tkinter as tk

import numpy as np
import soundcard as sc
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    OpenAI, RateLimitError)

load_dotenv()


# ---- Daftarkan DLL CUDA dari paket pip nvidia-* agar faster-whisper (CTranslate2)
#      bisa menemukan cublas64_12.dll & cudnn64_9.dll di GPU NVIDIA (Windows) ----
def _register_cuda_dlls():
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return

    import ctypes

    dll_dirs = []
    for base in nvidia.__path__:
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            dll_dir = os.path.join(base, sub, "bin")
            if os.path.isdir(dll_dir):
                dll_dirs.append(dll_dir)

    # 1) Tambahkan ke PATH (berlaku untuk seluruh proses & thread, termasuk
    #    lazy-load DLL oleh CTranslate2 saat encode()).
    if dll_dirs:
        os.environ["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ.get("PATH", "")

    # 2) Daftarkan sebagai DLL directory (Python 3.8+).
    if hasattr(os, "add_dll_directory"):
        for d in dll_dirs:
            try:
                os.add_dll_directory(d)
            except OSError:
                pass

    # 3) Preload eksplisit dengan urutan dependensi yang benar agar sudah ada
    #    di memori sebelum CTranslate2 memuatnya.
    for dll in ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
        for d in dll_dirs:
            p = os.path.join(d, dll)
            if os.path.isfile(p):
                try:
                    ctypes.CDLL(p)
                except OSError:
                    pass
                break


_register_cuda_dlls()

# ================== KONFIGURASI (bisa diubah lewat .env) ==================
# Dua model Groq (kuota rate-limit terpisah per model) + Gemini sebagai cadangan.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")          # primer
GROQ_MODEL_2 = os.getenv("GROQ_MODEL_2", "openai/gpt-oss-20b")       # fallback (kuota terpisah)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")         # cadangan
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "granite-4.0-micro")  # nama model persis di LM Studio
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")      # base.en kalau pakai CPU
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")         # "cpu" kalau tanpa GPU NVIDIA
CONTEXT_DIR = os.getenv("CONTEXT_DIR", "konteks")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))  # ~3k token, hemat kuota
# Batas karakter konteks yang benar-benar dikirim ke LLM per pertanyaan
# (~2000 token total request; 4 char ~= 1 token). Hemat kuota Groq 8k token/menit.
CONTEXT_BUDGET_CHARS = int(os.getenv("CONTEXT_BUDGET_CHARS", "6000"))

# Kosakata khusus agar Whisper tidak salah dengar istilah teknis
ASR_VOCAB = os.getenv(
    "ASR_VOCAB",
    "YOLOv8, YOLOv8n, SAW, Simple Additive Weighting, BMKG, debris, "
    "flood mitigation, rainfall, SAFE, WARNING, DANGER, mAP, precision, recall, "
    "river debris, meteorological data, bounding box, dataset, inference, IoU",
)

REC_RATE = 48000          # sample rate rekaman loopback
ASR_RATE = 16000          # sample rate yang dibutuhkan Whisper
CHUNK_SEC = 0.25
SILENCE_RMS = float(os.getenv("SILENCE_RMS", "0.01"))  # naikkan kalau audio berisik
END_SILENCE_SEC = 0.9     # hening selama ini = pertanyaan dianggap selesai
PARTIAL_EVERY_SEC = 1.5   # frekuensi update transkrip parsial
DRAFT_MIN_WORDS = 6       # minimal kata sebelum draft jawaban dibuat
DRAFT_EVERY_SEC = 3.0     # jarak minimal antar draft (hemat rate limit)
MAX_UTTER_SEC = 30

SYSTEM_PROMPT = """You are a live Q&A helper for Ferdy, who is presenting the paper
"YOLOv8-Based River Debris Monitoring System With Meteorological Data for
Flood Mitigation" at ICORIS 2026. The question comes from speech recognition
and may contain transcription errors; infer the most likely intended question.
Answer ONLY from the provided context files (folder `konteks`). If the context
does not cover the question, say so briefly and suggest an honest answer such as
"That is a limitation we plan to address in future work." Never invent numbers,
metrics, or results.
Format: at most 3 bullet points, each max 20 words, in English, plain text only.
No markdown, no bold, no headings. Start each bullet with "\u2022 ".

=== PAPER CONTEXT ===
{context}"""


# ================== UTIL: BERSIHKAN MARKDOWN ==================
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^(\s*)[-*]\s+", re.MULTILINE)
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_BOLD2 = re.compile(r"__(.+?)__")
_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_MD_ITALIC2 = re.compile(r"(?<!_)_(?!\s)(.+?)(?<!\s)_(?!_)")
_MD_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9]*\n?(.*?)```", re.DOTALL)
_MD_INLINE_CODE = re.compile(r"`([^`]+)`")


def clean_md(text):
    """Hilangkan markdown (bold/italic/heading/backtick) dan ubah bullet
    '- '/'* ' menjadi '\u2022 '. Kembalikan teks polos yang rapi."""
    if not text:
        return ""
    t = text
    t = _MD_CODE_BLOCK.sub(r"\1", t)     # ```code``` -> code
    t = _MD_INLINE_CODE.sub(r"\1", t)    # `code` -> code
    t = _MD_HEADING.sub("", t)           # ## Judul -> Judul
    t = _MD_BOLD.sub(r"\1", t)           # **tebal** -> tebal
    t = _MD_BOLD2.sub(r"\1", t)          # __tebal__ -> tebal
    t = _MD_ITALIC.sub(r"\1", t)         # *miring* -> miring
    t = _MD_ITALIC2.sub(r"\1", t)        # _miring_ -> miring
    t = _MD_BULLET.sub("\\1\u2022 ", t)  # "- " / "* " -> "\u2022 "
    # rapikan spasi berlebih di tiap baris
    lines = [ln.rstrip() for ln in t.splitlines()]
    return "\n".join(lines).strip()


# ================== KONTEKS ==================
def load_context():
    """Baca semua .md/.txt/.pdf di folder konteks/ (urut nama file)."""
    parts = []
    for path in sorted(glob.glob(os.path.join(CONTEXT_DIR, "*"))):
        name = os.path.basename(path)
        if path.lower().endswith((".txt", ".md")):
            with open(path, encoding="utf-8") as f:
                parts.append(f"### {name}\n{f.read()}")
        elif path.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
                parts.append(f"### {name}\n{text}")
            except ImportError:
                print(f"[!] pypdf belum terpasang, {name} dilewati")
    context = "\n\n".join(parts)
    if len(context) > MAX_CONTEXT_CHARS:
        print(f"[!] Konteks {len(context)} karakter dipotong ke {MAX_CONTEXT_CHARS}")
        context = context[:MAX_CONTEXT_CHARS]
    return context


# ================== TRIM KONTEKS PER PERTANYAAN ==================
_WORD = re.compile(r"[a-zA-Z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "how", "what", "why", "did", "do", "does", "you", "your", "this",
    "that", "these", "those", "with", "from", "by", "at", "as", "it", "be", "we",
    "our", "can", "will", "about", "which", "when", "there", "their", "they",
}


def _keywords(text):
    return {w.lower() for w in _WORD.findall(text) if len(w) > 2 and w.lower() not in _STOP}


def trim_context(context, question, budget=CONTEXT_BUDGET_CHARS):
    """Pilih potongan konteks paling relevan dengan pertanyaan agar request
    hemat token. Konteks dipecah per paragraf, di-skor berdasarkan kecocokan
    kata kunci pertanyaan, lalu diambil sampai memenuhi budget karakter."""
    if not context:
        return ""
    if len(context) <= budget:
        return context
    kws = _keywords(question)
    # pecah per paragraf (baris kosong sebagai pemisah)
    paras = [p.strip() for p in re.split(r"\n\s*\n", context) if p.strip()]
    if not kws:
        # tak ada kata kunci berarti: ambil bagian awal saja
        return context[:budget]

    scored = []
    for i, p in enumerate(paras):
        pk = _keywords(p)
        overlap = len(kws & pk)
        # sedikit bonus untuk paragraf yang mengandung angka (metrik/hasil)
        score = overlap + (0.5 if re.search(r"\d", p) else 0)
        scored.append((score, i, p))

    # urutkan berdasarkan skor (tinggi dulu), tetapi jaga urutan asli saat merakit
    ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
    chosen, total = [], 0
    for score, i, p in ranked:
        if score <= 0 and chosen:
            break  # sudah tak relevan, hentikan
        if total + len(p) > budget and chosen:
            continue
        chosen.append((i, p))
        total += len(p)
        if total >= budget:
            break
    if not chosen:  # fallback: tidak ada yang cocok -> potong awal
        return context[:budget]
    chosen.sort(key=lambda x: x[0])  # kembalikan ke urutan asli dokumen
    return "\n\n".join(p for _, p in chosen)


# ================== LLM ROUTER (Groq-120b -> Groq-20b -> Gemini) ==================
# Token budget: gpt-oss-120b Groq free tier = 8.000 token/menit. Jaga tiap request
# < ~2.000 token -> konteks di-trim + max_tokens kecil.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))
RATE_LIMIT_COOLDOWN = int(os.getenv("RATE_LIMIT_COOLDOWN", "60"))  # detik, bila tak ada Retry-After
ERROR_COOLDOWN = int(os.getenv("ERROR_COOLDOWN", "15"))            # detik, untuk timeout/koneksi
RETRY_BACKOFFS = (1, 2)  # detik: retry 429 di backend yang sama (2x) sebelum pindah


class Backend:
    def __init__(self, key, name, client, model, extra):
        self.key, self.name, self.client, self.model, self.extra = key, name, client, model, extra
        self.cooldown_until = 0.0
        self.reason = ""

    def available(self):
        return time.time() >= self.cooldown_until

    def cool_down(self, seconds, reason):
        self.cooldown_until = time.time() + seconds
        self.reason = reason


def _retry_after(err):
    """Ambil header Retry-After dari error 429 kalau ada."""
    try:
        val = err.response.headers.get("retry-after")
        return max(1, int(float(val))) if val else None
    except Exception:
        return None


class LLM:
    """Router backend dengan fallback berurutan.

    Default urutan (BACKEND_ORDER): 'groq,groq2,gemini'.
      - groq   : Groq GROQ_MODEL   (primer, mis. openai/gpt-oss-120b) -> label "Groq-120b"
      - groq2  : Groq GROQ_MODEL_2 (kuota terpisah, mis. gpt-oss-20b) -> label "Groq-20b"
      - gemini : Gemini GEMINI_MODEL                                  -> label "Gemini"
      - local  : LM Studio lokal (opsional)                          -> label "Lokal"

    Perilaku error:
      - 429 (rate limit): retry backend yang SAMA setelah backoff 1s lalu 2s;
        jika tetap 429, istirahatkan backend & pindah ke backend berikutnya.
      - 4xx selain 429 (400/401/403/404/...): langsung skip backend
        (401/403 diistirahatkan lama karena kemungkinan key/model salah).
      - timeout / koneksi: istirahat ERROR_COOLDOWN detik lalu pindah.
    """

    def __init__(self):
        def label_for(key, model):
            if key in ("groq", "groq2"):
                m = re.search(r"(\d+)b", model.lower())  # 120b / 20b
                return f"Groq-{m.group(1)}b" if m else "Groq"
            if key == "gemini":
                return "Gemini"
            return "Lokal"

        # (key, display_env, base_url, env_prefix, model)
        clouds = {
            "groq": ("https://api.groq.com/openai/v1", "GROQ", GROQ_MODEL),
            "groq2": ("https://api.groq.com/openai/v1", "GROQ", GROQ_MODEL_2),
            "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/",
                       "GEMINI", GEMINI_MODEL),
        }
        order = [b.strip().lower() for b in
                 os.getenv("BACKEND_ORDER", "groq,groq2,gemini").split(",") if b.strip()]
        self.backends = []
        for b in order:
            if b in clouds:
                url, env, model = clouds[b]
                api_key = os.getenv(f"{env}_API_KEY")
                if not api_key:
                    continue
                extra = {}
                effort = os.getenv(f"{env}_REASONING")
                if not effort and "gpt-oss" in model:
                    effort = "low"  # model reasoning: pakai usaha rendah agar cepat
                if effort and effort.lower() != "none":
                    extra["reasoning_effort"] = effort
                client = OpenAI(base_url=url, api_key=api_key, timeout=8, max_retries=0)
                self.backends.append(Backend(b, label_for(b, model), client, model, extra))
            elif b == "local":
                client = OpenAI(base_url=LOCAL_BASE_URL, api_key="lm-studio",
                                timeout=30, max_retries=0)
                self.backends.append(Backend(b, "Lokal", client, LOCAL_MODEL, {}))
        print("Urutan backend:", " -> ".join(f"{x.name}({x.model})" for x in self.backends))

    def _call(self, be, messages):
        """Panggil satu backend (streaming). Menangani 429 dengan retry backoff
        di backend yang sama. Mengembalikan generator (kind, val), atau me-raise
        error non-429 agar router memutuskan skip/pindah."""
        last_rate_err = None
        for attempt in range(len(RETRY_BACKOFFS) + 1):
            try:
                resp = be.client.chat.completions.create(
                    model=be.model, messages=messages, stream=True,
                    max_tokens=MAX_TOKENS, temperature=0.3, **be.extra,
                )
                return resp
            except RateLimitError as e:
                last_rate_err = e
                if attempt < len(RETRY_BACKOFFS):
                    time.sleep(RETRY_BACKOFFS[attempt])  # 1s, lalu 2s
                    continue
                raise  # habis jatah retry -> biar router istirahatkan & pindah
        raise last_rate_err  # pengaman (tak seharusnya sampai sini)

    def stream(self, messages):
        tried_any = False
        for be in self.backends:
            if not be.available():
                continue
            tried_any = True
            try:
                resp = self._call(be, messages)
                yield ("backend", be.name)
                for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield ("token", chunk.choices[0].delta.content)
                return
            except RateLimitError as e:
                wait = _retry_after(e) or RATE_LIMIT_COOLDOWN
                be.cool_down(wait, "rate limit")
                yield ("error", f"{be.name} kena limit (429), istirahat {wait}s -> pindah backend")
            except (APITimeoutError, APIConnectionError) as e:
                be.cool_down(ERROR_COOLDOWN, "koneksi")
                yield ("error", f"{be.name} tidak merespons ({type(e).__name__}) -> pindah backend")
            except APIStatusError as e:
                code = getattr(e, "status_code", 0)
                if code in (401, 403):
                    be.cool_down(10**9, "API key ditolak")  # key salah: jangan dicoba lagi
                    yield ("error", f"{be.name}: API key ditolak (cek .env)")
                elif code == 404:
                    be.cool_down(10**9, "model tidak ada")  # nama model salah
                    yield ("error", f"{be.name}: model '{be.model}' tidak ada (cek .env)")
                elif 400 <= code < 500:
                    # 4xx selain 429 -> skip segera, jangan buang waktu retry
                    be.cool_down(ERROR_COOLDOWN, f"HTTP {code}")
                    yield ("error", f"{be.name} HTTP {code} -> skip ke backend berikutnya")
                else:
                    be.cool_down(ERROR_COOLDOWN, f"HTTP {code}")
                    yield ("error", f"{be.name} error HTTP {code} -> pindah backend")
            except Exception as e:
                be.cool_down(ERROR_COOLDOWN, type(e).__name__)
                yield ("error", f"{be.name} gagal ({type(e).__name__}) -> pindah backend")
        if not tried_any:
            # semua sedang istirahat: paksa coba yang paling cepat pulih
            be = min(self.backends, key=lambda x: x.cooldown_until)
            be.cooldown_until = 0
            yield ("error", f"Semua backend istirahat, paksa coba {be.name}")
            yield from self.stream(messages)
            return
        yield ("error", "Semua backend gagal. Cek internet / kuota API.")


# ================== ASISTEN ==================
class Assistant:
    def __init__(self, ui_q):
        self.ui_q = ui_q
        self.llm = LLM()
        self.context = load_context()  # konteks penuh; di-trim per pertanyaan
        self.asr = self._load_asr()
        self.audio_q = queue.Queue()
        self.gen_id = 0
        self.lock = threading.Lock()

    def _load_asr(self):
        """Muat model Whisper. Coba GPU (CUDA) dulu, jika library CUDA
        (mis. cublas64_12.dll / cuDNN) tidak ada, otomatis fallback ke CPU."""
        if WHISPER_DEVICE == "cuda":
            try:
                model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
                self.ui_q.put(("info", f"ASR: {WHISPER_MODEL} @ CUDA (float16)"))
                return model
            except Exception as e:
                self.ui_q.put(("info",
                    f"CUDA tidak tersedia ({e}). Beralih ke CPU..."))
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        self.ui_q.put(("info", f"ASR: {WHISPER_MODEL} @ CPU (int8)"))
        return model

    # ---------- LLM ----------
    def ask(self, question, final=True):
        with self.lock:
            self.gen_id += 1
            my_id = self.gen_id
        threading.Thread(target=self._run_llm, args=(question, final, my_id),
                         daemon=True).start()

    def _build_messages(self, question):
        """Bangun pesan LLM: trim konteks ke bagian paling relevan dengan
        pertanyaan (hemat token) lalu susun system + user."""
        ctx = trim_context(self.context, question) if self.context else ""
        system = SYSTEM_PROMPT.format(context=ctx or "(no context loaded)")
        return system, ctx

    def _run_llm(self, question, final, my_id):
        prefix = "Complete question: " if final else "Partial question (still being asked): "
        system, _ = self._build_messages(question)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prefix + question},
        ]
        label = "FINAL" if final else "DRAFT"
        buf = []          # kumpulkan token untuk dibersihkan sebagai teks utuh
        got_backend = False
        for kind, val in self.llm.stream(messages):
            if my_id != self.gen_id:  # ada pertanyaan/draft lebih baru -> hentikan
                return
            if kind == "backend":
                got_backend = True
                buf = []
                # tampilkan label backend di status bar, mis. "FINAL | Groq-120b"
                self.ui_q.put(("answer_reset", f"{label}  |  {val}"))
            elif kind == "token":
                buf.append(val)
                # tampilkan versi bersih secara bertahap (tanpa markdown)
                self.ui_q.put(("answer_set", clean_md("".join(buf))))
            elif kind == "error":
                if not got_backend:  # error sebelum ada jawaban -> tampil di status
                    self.ui_q.put(("info", val))
        # finalisasi: pastikan yang tampil adalah teks bersih utuh
        if got_backend and my_id == self.gen_id:
            self.ui_q.put(("answer_set", clean_md("".join(buf))))

    # ---------- ASR ----------
    def transcribe(self, audio):
        segments, _ = self.asr.transcribe(
            audio, language="en", beam_size=1, vad_filter=True,
            condition_on_previous_text=False, initial_prompt=ASR_VOCAB,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    # ---------- AUDIO ----------
    def record_loop(self):
        """Rekam suara yang keluar dari speaker default (termasuk Zoom)."""
        speaker = sc.default_speaker()
        loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        self.ui_q.put(("info", f"Mendengarkan: {speaker.name}"))
        frames = int(REC_RATE * CHUNK_SEC)
        with loopback.recorder(samplerate=REC_RATE) as rec:
            while True:
                data = rec.record(numframes=frames)
                mono = data.mean(axis=1) if data.ndim > 1 else data
                mono = mono[: len(mono) // 3 * 3].reshape(-1, 3).mean(axis=1)  # 48k -> 16k
                self.audio_q.put(mono.astype(np.float32))

    def process_loop(self):
        buf, voiced, silence = [], False, 0.0
        last_partial = last_draft = 0.0
        last_draft_words = 0
        while True:
            chunk = self.audio_q.get()
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > SILENCE_RMS:
                voiced, silence = True, 0.0
                buf.append(chunk)
            elif voiced:
                silence += CHUNK_SEC
                buf.append(chunk)
            if not voiced:
                continue

            audio = np.concatenate(buf)
            now = time.time()
            ended = silence >= END_SILENCE_SEC or len(audio) / ASR_RATE > MAX_UTTER_SEC

            if ended:
                text = self.transcribe(audio)
                if len(text.split()) >= 3:
                    self.ui_q.put(("question", text + "  [selesai]"))
                    self.ask(text, final=True)
                buf, voiced, silence, last_draft_words = [], False, 0.0, 0
            elif now - last_partial >= PARTIAL_EVERY_SEC:
                last_partial = now
                text = self.transcribe(audio)
                if text:
                    self.ui_q.put(("question", text + " ..."))
                n = len(text.split())
                if (n >= DRAFT_MIN_WORDS and n > last_draft_words
                        and now - last_draft >= DRAFT_EVERY_SEC):
                    last_draft, last_draft_words = now, n
                    self.ask(text, final=False)

    def run(self):
        threading.Thread(target=self.record_loop, daemon=True).start()
        self.process_loop()


# ================== OVERLAY UI ==================
class Overlay:
    BG, FG, DIM, ACC = "#111418", "#f2f2f2", "#8a8f98", "#7fb3ff"

    def __init__(self, ui_q):
        self.q = ui_q
        self.on_manual = None
        r = self.root = tk.Tk()
        r.title("ICORIS Q&A")
        r.attributes("-topmost", True)
        r.geometry("540x380+40+40")
        r.configure(bg=self.BG)

        self.status = tk.Label(r, text="Memuat model...", fg=self.DIM, bg=self.BG,
                               font=("Segoe UI", 9), anchor="w")
        self.status.pack(fill="x", padx=10, pady=(8, 0))
        self.qlabel = tk.Label(r, text="Q: -", fg=self.ACC, bg=self.BG, justify="left",
                               wraplength=515, font=("Segoe UI", 10), anchor="w")
        self.qlabel.pack(fill="x", padx=10, pady=4)
        self.ans = tk.Text(r, bg=self.BG, fg=self.FG, font=("Segoe UI", 13),
                           wrap="word", bd=0, highlightthickness=0)
        self.ans.pack(fill="both", expand=True, padx=10)

        self.entry = tk.Entry(r, bg="#1c2128", fg=self.FG, insertbackground=self.FG,
                              font=("Segoe UI", 10), bd=0)
        self.entry.pack(fill="x", padx=10, pady=8, ipady=4)
        self.entry.insert(0, "Ketik pertanyaan manual lalu Enter (untuk tes)")
        self.entry.bind("<FocusIn>", lambda e: self.entry.delete(0, "end"))
        self.entry.bind("<Return>", self._manual)
        r.after(50, self._poll)

    def _manual(self, _):
        text = self.entry.get().strip()
        if text and self.on_manual:
            self.qlabel.config(text="Q: " + text + "  [manual]")
            self.on_manual(text, True)
            self.entry.delete(0, "end")

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "question":
                    self.qlabel.config(text="Q: " + val)
                elif kind == "answer_reset":
                    self.ans.delete("1.0", "end")
                    self.status.config(text=val)
                elif kind == "answer_set":
                    # ganti seluruh isi dengan teks bersih (tanpa markdown)
                    self.ans.delete("1.0", "end")
                    self.ans.insert("end", val)
                elif kind == "answer_token":
                    self.ans.insert("end", val)
                elif kind in ("answer_error", "info"):
                    self.status.config(text=val)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)


if __name__ == "__main__":
    ui_q = queue.Queue()
    ui = Overlay(ui_q)

    def start():
        try:
            a = Assistant(ui_q)
        except Exception as e:
            ui_q.put(("info", f"Gagal memuat: {e}"))
            return
        ui.on_manual = a.ask
        print(f"Konteks dimuat: {len(a.context)} karakter")
        a.run()

    threading.Thread(target=start, daemon=True).start()
    ui.root.mainloop()
