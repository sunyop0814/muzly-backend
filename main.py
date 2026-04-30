from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import parselmouth, parselmouth.praat as praat
import numpy as np
import tempfile, os, wave, json, httpx
import av

# ── CREPE (torchcrepe) 선택적 로드 ────────────────────────
try:
    import torch, torchcrepe, soundfile as sf
    from scipy import signal as sig
    CREPE_OK = True
    print("[CREPE] torchcrepe 로드 성공 — 고정밀 F0 분석 활성화")
except Exception as e:
    CREPE_OK = False
    print(f"[CREPE] 비활성화: {e}")

app = FastAPI(title="MUZLY Voice Analysis API v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")


# ── PyAV: 모든 포맷 → WAV 변환 ────────────────────────────
def to_wav(src: str, dst: str) -> bool:
    try:
        container = av.open(src)
        stream = next((s for s in container.streams if s.type == 'audio'), None)
        if not stream:
            return False
        sr = stream.sample_rate or 44100
        chunks = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            chunks.append(arr.astype(np.float32))
        container.close()
        if not chunks:
            return False
        audio = np.concatenate(chunks)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16)
        with wave.open(dst, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        size = os.path.getsize(dst)
        print(f"[WAV변환] {sr}Hz {len(audio)/sr:.1f}초 {size//1024}KB")
        return size > 1000
    except Exception as e:
        print(f"[WAV변환 실패] {e}")
        return False


# ── CREPE: 고정밀 F0 분석 ─────────────────────────────────
def analyze_crepe(wav_path: str) -> dict:
    """
    torchcrepe tiny 모델로 음정을 밀리초 단위로 추적.
    반환 지표: f0_mean_crepe, f0_std_crepe, f0_stability,
               f0_min, f0_max, f0_range_st, voiced_ratio
    """
    if not CREPE_OK:
        return {"crepe_available": False}
    try:
        audio, sr = sf.read(wav_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # 16kHz 리샘플링 (torchcrepe 권장)
        if sr != 16000:
            n = int(len(audio) * 16000 / sr)
            audio = sig.resample(audio, n)
            sr = 16000

        audio_t = torch.from_numpy(audio).unsqueeze(0)
        freq, conf = torchcrepe.predict(
            audio_t, sr,
            fmin=50, fmax=1200,
            model='tiny',
            decoder=torchcrepe.decode.weighted_argmax,
            return_periodicity=True,
            batch_size=512,
            device='cpu'
        )
        freq_np = freq.squeeze().numpy()
        conf_np = conf.squeeze().numpy()

        valid = conf_np > 0.6
        if valid.sum() < 10:
            return {"crepe_available": True, "crepe_valid": False}

        f0v = freq_np[valid]
        f0_mean  = round(float(f0v.mean()), 1)
        f0_std   = round(float(f0v.std()), 2)
        f0_min   = round(float(f0v.min()), 1)
        f0_max   = round(float(f0v.max()), 1)
        stability = round(max(0, min(100, (1 - f0_std / (f0_mean + 1e-8)) * 100)), 1)
        semitones = round(12 * np.log2((f0_max + 1e-8) / (f0_min + 1e-8)), 1) if f0_min > 0 else 0.0
        voiced    = round(float(valid.mean()) * 100, 1)

        result = {
            "crepe_available": True,
            "crepe_valid":     True,
            "f0_mean_crepe":   f0_mean,
            "f0_std_crepe":    f0_std,
            "f0_stability":    stability,
            "f0_min":          f0_min,
            "f0_max":          f0_max,
            "f0_range_st":     semitones,
            "voiced_ratio":    voiced,
        }
        print(f"[CREPE] {result}")
        return result
    except Exception as e:
        print(f"[CREPE 오류] {e}")
        return {"crepe_available": True, "crepe_valid": False}


# ── Parselmouth: HNR·Jitter·Shimmer·Formant ───────────────
def analyze_parselmouth(wav_path: str) -> dict:
    snd = parselmouth.Sound(wav_path)
    duration = snd.duration

    try:
        harm = praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr  = praat.call(harm, "Get mean", 0, 0)
        hnr  = round(float(hnr), 2) if np.isfinite(hnr) else 0.0
    except: hnr = 0.0

    pp = None
    try:
        pp     = praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = praat.call([snd, pp], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter = round(float(jitter) * 100, 3) if np.isfinite(jitter) else 0.0
    except: jitter = 0.0

    try:
        if pp is None:
            pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        shimmer = praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer = round(float(shimmer) * 100, 3) if np.isfinite(shimmer) else 0.0
    except: shimmer = 0.0

    try:
        pitch = praat.call(snd, "To Pitch", 0, 75, 500)
        f0    = praat.call(pitch, "Get mean", 0, 0, "Hertz")
        f0    = round(float(f0), 1) if np.isfinite(f0) else 0.0
    except: f0 = 0.0

    try:
        fmt = praat.call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        f1  = praat.call(fmt, "Get mean", 1, 0, 0, "Hertz")
        f2  = praat.call(fmt, "Get mean", 2, 0, 0, "Hertz")
        f1  = round(float(f1), 1) if np.isfinite(f1) else 0.0
        f2  = round(float(f2), 1) if np.isfinite(f2) else 0.0
    except: f1, f2 = 0.0, 0.0

    result = dict(duration=round(duration, 1), hnr=hnr, jitter=jitter,
                  shimmer=shimmer, f0_mean=f0, f1=f1, f2=f2)
    print(f"[Parselmouth] {result}")
    return result


# ── 통합 분석 ─────────────────────────────────────────────
def analyze_voice(wav_path: str) -> dict:
    metrics = analyze_parselmouth(wav_path)
    crepe   = analyze_crepe(wav_path)
    metrics.update(crepe)
    # CREPE F0가 유효하면 Parselmouth F0 대체
    if crepe.get("crepe_valid") and crepe.get("f0_mean_crepe", 0) > 0:
        metrics["f0_mean"] = crepe["f0_mean_crepe"]
    return metrics


# ── 점수 환산 ─────────────────────────────────────────────
def to_score(val, lo, hi, higher=True) -> int:
    if higher:
        if val >= hi: return 95
        if val <= lo: return 30
        return int(30 + (val - lo) / (hi - lo) * 65)
    else:
        if val <= lo: return 95
        if val >= hi: return 30
        return int(95 - (val - lo) / (hi - lo) * 65)

def compute_scores(m: dict) -> dict:
    scores = {
        "hnr_score":     to_score(m["hnr"],     7, 22, True),
        "jitter_score":  to_score(m["jitter"],  0, 1.0, False),
        "shimmer_score": to_score(m["shimmer"], 0, 6.0, False),
        "f0_score":      75 if m["f0_mean"] > 50 else 40,
    }
    if m.get("crepe_valid"):
        scores["f0_stability_score"] = min(95, max(30, int(m.get("f0_stability", 70))))
        scores["f0_range_score"]     = min(95, max(30, int(30 + m.get("f0_range_st", 0) * 2.5)))
    return scores


# ── Claude 리포트 ─────────────────────────────────────────
async def get_report(metrics: dict, scores: dict) -> dict:
    if not CLAUDE_API_KEY:
        return rule_report(metrics, scores)

    crepe_line = ""
    if metrics.get("crepe_valid"):
        crepe_line = f"""
[CREPE 정밀 F0 측정]
- 음정 안정성: {metrics.get('f0_stability', 0)}% (높을수록 안정)
- 음역 범위: {metrics.get('f0_min', 0)}~{metrics.get('f0_max', 0)}Hz ({metrics.get('f0_range_st', 0)} 반음)
- 유성음 비율: {metrics.get('voiced_ratio', 0)}%
- F0 표준편차: {metrics.get('f0_std_crepe', 0)}Hz"""

    prompt = f"""전문 보컬 트레이너로서 아래 음성 측정 수치를 분석해 뮤즐리 보이스 인바디 리포트를 작성하세요.

[Parselmouth 측정]
- HNR(성대건강도): {metrics['hnr']}dB (점수:{scores['hnr_score']}/100, 기준:14~20dB)
- Jitter(음정떨림): {metrics['jitter']}% (점수:{scores['jitter_score']}/100, 기준:0.5% 이하)
- Shimmer(성량떨림): {metrics['shimmer']}% (점수:{scores['shimmer_score']}/100, 기준:3% 이하)
- Formant F1:{metrics['f1']}Hz / F2:{metrics['f2']}Hz
- 평균F0: {metrics['f0_mean']}Hz
- 녹음길이: {metrics['duration']}초{crepe_line}

JSON만 출력(다른텍스트금지):
{{"voice_type":"창의적보컬타입이름","overall_grade":"S/A/B/C/D","summary":"2문장총평","strengths":["강점1","강점2"],"improvements":["개선1","개선2"],"training_tip":"오늘바로할수있는구체적팁","celebrity_voice":"비슷한스타일가수"}}"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 600,
                      "messages": [{"role": "user", "content": prompt}]}
            )
        raw = resp.json()["content"][0]["text"].strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[Claude 오류] {e}")
        return rule_report(metrics, scores)

def rule_report(m, s):
    avg = (s["hnr_score"] + s["jitter_score"] + s["shimmer_score"]) / 3
    g = "S" if avg>=88 else "A" if avg>=75 else "B" if avg>=60 else "C" if avg>=45 else "D"
    return {"voice_type": "보이스 분석 완료", "overall_grade": g,
            "summary": f"HNR {m['hnr']}dB, Jitter {m['jitter']}%로 측정됐습니다.",
            "strengths": ["측정 완료"], "improvements": ["꾸준한 발성 훈련 권장"],
            "training_tip": "매일 15분 허밍으로 성대 건강을 유지하세요.",
            "celebrity_voice": "-"}


# ── 엔드포인트 ────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "MUZLY API v3", "crepe": CREPE_OK, "pyav": True}

@app.get("/health")
def health():
    return {"ok": True, "crepe": CREPE_OK}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) < 500:
        raise HTTPException(400, "파일이 너무 짧습니다.")

    fname  = (file.filename or "voice.wav").lower()
    suffix = ".webm"
    for ext in [".wav", ".mp3", ".m4a", ".ogg", ".webm", ".mp4"]:
        if fname.endswith(ext): suffix = ext; break

    src_fd, src_path = tempfile.mkstemp(suffix=suffix)
    wav_path = src_path.replace(suffix, "_out.wav")

    try:
        with os.fdopen(src_fd, 'wb') as f: f.write(content)
        print(f"[수신] {fname} {len(content)//1024}KB")

        if suffix == ".wav":
            wav_path = src_path; ok = True
        else:
            ok = to_wav(src_path, wav_path)

        if not ok:
            raise HTTPException(500, "오디오 변환 실패")

        metrics = analyze_voice(wav_path)
        scores  = compute_scores(metrics)
        report  = await get_report(metrics, scores)

        return {"success": True, "metrics": metrics, "scores": scores, "report": report}

    except HTTPException: raise
    except Exception as e:
        print(f"[오류] {e}"); raise HTTPException(500, str(e))
    finally:
        for p in set([src_path, wav_path]):
            try:
                if p and os.path.exists(p): os.unlink(p)
            except: pass
