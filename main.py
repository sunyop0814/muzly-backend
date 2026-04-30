from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import parselmouth
import parselmouth.praat as praat
import numpy as np
import tempfile, os, wave, json, httpx
import av

app = FastAPI(title="MUZLY Voice Analysis API v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

def to_wav_with_av(src_path: str, dst_path: str) -> bool:
    try:
        container = av.open(src_path)
        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
        if audio_stream is None: return False
        sr = audio_stream.sample_rate or 44100
        chunks = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            if arr.ndim == 2: arr = arr.mean(axis=0)
            chunks.append(arr.astype(np.float32))
        container.close()
        if not chunks: return False
        audio = np.concatenate(chunks)
        peak = np.max(np.abs(audio))
        if peak > 0: audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16)
        with wave.open(dst_path, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(sr); wf.writeframes(pcm.tobytes())
        size = os.path.getsize(dst_path)
        print(f"[WAV변환 성공] {sr}Hz, {len(audio)/sr:.1f}초, {size//1024}KB")
        return size > 1000
    except Exception as e:
        print(f"[WAV변환 실패] {e}"); return False

def analyze_voice(wav_path: str) -> dict:
    snd = parselmouth.Sound(wav_path)
    duration = snd.duration
    try:
        harm = praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = praat.call(harm, "Get mean", 0, 0)
        hnr = round(float(hnr), 2) if np.isfinite(hnr) else 0.0
    except: hnr = 0.0
    pp = None
    try:
        pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = praat.call([snd, pp], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter = round(float(jitter)*100, 3) if np.isfinite(jitter) else 0.0
    except: jitter = 0.0
    try:
        if pp is None: pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        shimmer = praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer = round(float(shimmer)*100, 3) if np.isfinite(shimmer) else 0.0
    except: shimmer = 0.0
    try:
        pitch = praat.call(snd, "To Pitch", 0, 75, 500)
        f0 = praat.call(pitch, "Get mean", 0, 0, "Hertz")
        f0 = round(float(f0), 1) if np.isfinite(f0) else 0.0
    except: f0 = 0.0
    try:
        fmt = praat.call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        f1 = praat.call(fmt, "Get mean", 1, 0, 0, "Hertz")
        f2 = praat.call(fmt, "Get mean", 2, 0, 0, "Hertz")
        f1 = round(float(f1),1) if np.isfinite(f1) else 0.0
        f2 = round(float(f2),1) if np.isfinite(f2) else 0.0
    except: f1, f2 = 0.0, 0.0
    result = dict(duration=round(duration,1), hnr=hnr, jitter=jitter, shimmer=shimmer, f0_mean=f0, f1=f1, f2=f2)
    print(f"[분석결과] {result}")
    return result

def to_score(val, lo, hi, higher=True) -> int:
    if higher:
        if val>=hi: return 95
        if val<=lo: return 30
        return int(30+(val-lo)/(hi-lo)*65)
    else:
        if val<=lo: return 95
        if val>=hi: return 30
        return int(95-(val-lo)/(hi-lo)*65)

def compute_scores(m):
    return {"hnr_score":to_score(m["hnr"],7,22), "jitter_score":to_score(m["jitter"],0,1.0,False),
            "shimmer_score":to_score(m["shimmer"],0,6.0,False), "f0_score":75 if m["f0_mean"]>50 else 40}

async def get_report(metrics, scores):
    if not CLAUDE_API_KEY: return rule_report(metrics, scores)
    prompt = f"""전문 보컬 트레이너로서 음성 분석 수치를 보고 뮤즐리 보이스 인바디 리포트를 JSON만으로 작성하세요.
HNR:{metrics['hnr']}dB(점수:{scores['hnr_score']}), Jitter:{metrics['jitter']}%(점수:{scores['jitter_score']}), Shimmer:{metrics['shimmer']}%(점수:{scores['shimmer_score']}), F0:{metrics['f0_mean']}Hz, F1:{metrics['f1']}Hz, F2:{metrics['f2']}Hz, 길이:{metrics['duration']}초
JSON만 출력:{{"voice_type":"보컬타입","overall_grade":"A","summary":"총평2문장","strengths":["강점1","강점2"],"improvements":["개선1","개선2"],"training_tip":"오늘팁","celebrity_voice":"가수"}}"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key":CLAUDE_API_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
                json={"model":"claude-haiku-4-5-20251001","max_tokens":600,"messages":[{"role":"user","content":prompt}]})
        raw = resp.json()["content"][0]["text"].strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[Claude오류] {e}"); return rule_report(metrics, scores)

def rule_report(m, s):
    avg = (s["hnr_score"]+s["jitter_score"]+s["shimmer_score"])/3
    g = "S" if avg>=88 else "A" if avg>=75 else "B" if avg>=60 else "C" if avg>=45 else "D"
    return {"voice_type":"보이스분석완료","overall_grade":g,
            "summary":f"HNR {m['hnr']}dB, Jitter {m['jitter']}%로 측정됐습니다.",
            "strengths":["측정완료"],"improvements":["Claude API 연결 시 상세 분석"],"training_tip":"매일 허밍 15분","celebrity_voice":"-"}

@app.get("/")
def root(): return {"status":"MUZLY API v2","ffmpeg":"bundled via PyAV"}

@app.get("/health")
def health(): return {"ok":True}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) < 500: raise HTTPException(400, "파일이 너무 짧습니다.")
    fname = (file.filename or "voice.wav").lower()
    suffix = ".webm"
    for ext in [".wav",".mp3",".m4a",".ogg",".webm",".mp4"]:
        if fname.endswith(ext): suffix = ext; break
    src_fd, src_path = tempfile.mkstemp(suffix=suffix)
    wav_path = src_path.replace(suffix, "_out.wav")
    try:
        with os.fdopen(src_fd, 'wb') as f: f.write(content)
        print(f"[수신] {fname}, {len(content)//1024}KB")
        if suffix == ".wav": wav_path = src_path; ok = True
        else: ok = to_wav_with_av(src_path, wav_path)
        if not ok: raise HTTPException(500, "오디오 변환 실패")
        metrics = analyze_voice(wav_path)
        scores = compute_scores(metrics)
        report = await get_report(metrics, scores)
        return {"success":True,"metrics":metrics,"scores":scores,"report":report}
    except HTTPException: raise
    except Exception as e: print(f"[오류] {e}"); raise HTTPException(500, str(e))
    finally:
        for p in set([src_path, wav_path]):
            try:
                if p and os.path.exists(p): os.unlink(p)
            except: pass
