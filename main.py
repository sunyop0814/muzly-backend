from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import parselmouth, parselmouth.praat as praat
import numpy as np
import tempfile, os, wave, json, httpx
import av

# ── CREPE 로드 ────────────────────────────────────────────
try:
    import torch, torchcrepe, soundfile as sf
    from scipy import signal as sig
    CREPE_OK = True
    print("[CREPE] torchcrepe 로드 성공")
except Exception as e:
    CREPE_OK = False
    print(f"[CREPE] 비활성화: {e}")

app = FastAPI(title="MUZLY Voice Analysis API v4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")


# ══════════════════════════════════════════════════════════
# PyAV: 모든 포맷 → WAV 변환
# ══════════════════════════════════════════════════════════
def to_wav(src: str, dst: str) -> bool:
    try:
        container = av.open(src)
        stream = next((s for s in container.streams if s.type == 'audio'), None)
        if not stream: return False
        sr = stream.sample_rate or 44100
        chunks = []
        for frame in container.decode(stream):
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
        with wave.open(dst, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(sr); wf.writeframes(pcm.tobytes())
        size = os.path.getsize(dst)
        print(f"[WAV] {sr}Hz {len(audio)/sr:.1f}초 {size//1024}KB")
        return size > 1000
    except Exception as e:
        print(f"[WAV실패] {e}"); return False


# ══════════════════════════════════════════════════════════
# Parselmouth 직접 측정 (검증 완료 8개 지표)
# ══════════════════════════════════════════════════════════
def measure_parselmouth(wav: str) -> dict:
    snd = parselmouth.Sound(wav)
    duration = snd.duration
    out = {"duration": round(duration, 2)}

    # 1. HNR — 성대 건강도
    try:
        h = praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        v = praat.call(h, "Get mean", 0, 0)
        out["hnr"] = round(float(v), 2) if np.isfinite(v) else 0.0
    except: out["hnr"] = 0.0

    # 2. Jitter — 음정 미세 떨림
    pp = None
    try:
        pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
        v = praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        out["jitter_pct"] = round(float(v) * 100, 4) if np.isfinite(v) else 0.0
    except: out["jitter_pct"] = 0.0

    # 3. Shimmer — 성량 미세 떨림
    try:
        if pp is None:
            pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
        v = praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        out["shimmer_pct"] = round(float(v) * 100, 4) if np.isfinite(v) else 0.0
    except: out["shimmer_pct"] = 0.0

    # 4. F0 평균 (Parselmouth fallback용)
    try:
        pitch = praat.call(snd, "To Pitch", 0.0, 75, 600)
        f0_mean = praat.call(pitch, "Get mean", 0, 0, "Hertz")
        f0_std  = praat.call(pitch, "Get standard deviation", 0, 0, "Hertz")
        out["f0_mean_praat"] = round(float(f0_mean), 1) if np.isfinite(f0_mean) else 0.0
        out["f0_std_praat"]  = round(float(f0_std), 2) if np.isfinite(f0_std) else 0.0

        # 5. 비브라토 — F0 contour FFT (4~8Hz 피크)
        n = int(praat.call(pitch, "Get number of frames"))
        f0_vals = []
        for i in range(1, n+1):
            f = praat.call(pitch, "Get value in frame", i, "Hertz")
            if not (f != f) and f > 0:
                f0_vals.append(f)

        f0_arr = np.array(f0_vals)
        if len(f0_arr) > 30:
            f0_d = f0_arr - np.mean(f0_arr)
            ts = praat.call(pitch, "Get time step")
            frame_rate = 1.0 / ts if ts > 0 else 100.0
            fft_mag = np.abs(np.fft.rfft(f0_d))
            freqs   = np.fft.rfftfreq(len(f0_d), 1/frame_rate)
            vib_mask = (freqs >= 4.0) & (freqs <= 8.0)
            if vib_mask.sum() > 0 and fft_mag.max() > 0:
                vib_idx  = np.argmax(fft_mag[vib_mask])
                vib_rate = float(freqs[vib_mask][vib_idx])
                vib_strength = float(fft_mag[vib_mask].max() / fft_mag.max() * 100)
                out["vibrato_rate_hz"] = round(vib_rate, 2)
                out["vibrato_strength_pct"] = round(vib_strength, 1)
                out["vibrato_depth_hz"] = round(float(np.std(f0_d)), 2)
            else:
                out["vibrato_rate_hz"] = 0.0
                out["vibrato_strength_pct"] = 0.0
                out["vibrato_depth_hz"] = 0.0
        else:
            out["vibrato_rate_hz"] = 0.0
            out["vibrato_strength_pct"] = 0.0
            out["vibrato_depth_hz"] = 0.0
    except Exception as e:
        print(f"[F0/비브라토 오류] {e}")
        out["f0_mean_praat"] = 0.0
        out["f0_std_praat"] = 0.0
        out["vibrato_rate_hz"] = 0.0
        out["vibrato_strength_pct"] = 0.0
        out["vibrato_depth_hz"] = 0.0

    # 6. CPP — 발성 안정성 (Cepstral Peak Prominence)
    try:
        pc = praat.call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        cpp = praat.call(pc, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05,
                         "Parabolic", 0.001, 0, "Exponential decay", "Robust")
        out["cpp_db"] = round(float(cpp), 2) if np.isfinite(cpp) else 0.0
    except Exception as e:
        print(f"[CPP 오류] {e}"); out["cpp_db"] = 0.0

    # 7~8. Formant F1/F2/F3 — 공명 위치
    try:
        fmt = praat.call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        f1 = praat.call(fmt, "Get mean", 1, 0, 0, "Hertz")
        f2 = praat.call(fmt, "Get mean", 2, 0, 0, "Hertz")
        f3 = praat.call(fmt, "Get mean", 3, 0, 0, "Hertz")
        out["f1_hz"] = round(float(f1), 0) if np.isfinite(f1) else 0.0
        out["f2_hz"] = round(float(f2), 0) if np.isfinite(f2) else 0.0
        out["f3_hz"] = round(float(f3), 0) if np.isfinite(f3) else 0.0
    except: out["f1_hz"], out["f2_hz"], out["f3_hz"] = 0.0, 0.0, 0.0

    print(f"[Parselmouth] HNR={out['hnr']} Jit={out['jitter_pct']} Shim={out['shimmer_pct']} CPP={out['cpp_db']} VibRate={out['vibrato_rate_hz']}")
    return out


# ══════════════════════════════════════════════════════════
# CREPE 측정 (CNN 기반 F0)
# ══════════════════════════════════════════════════════════
def measure_crepe(wav: str) -> dict:
    if not CREPE_OK:
        return {"crepe_active": False}
    try:
        audio, sr = sf.read(wav, dtype='float32')
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != 16000:
            n = int(len(audio) * 16000 / sr)
            audio = sig.resample(audio, n).astype(np.float32)
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
        f_np = freq.squeeze().numpy()
        c_np = conf.squeeze().numpy()
        valid = c_np > 0.6

        if valid.sum() < 10:
            return {"crepe_active": True, "crepe_valid": False}

        f0v = f_np[valid]
        f0_mean = float(f0v.mean())
        f0_std  = float(f0v.std())
        f0_min  = float(f0v.min())
        f0_max  = float(f0v.max())
        stability = max(0.0, min(100.0, (1 - f0_std / (f0_mean + 1e-8)) * 100))
        st_range  = 12 * np.log2((f0_max + 1e-8) / (f0_min + 1e-8)) if f0_min > 0 else 0
        voiced    = float(valid.mean()) * 100

        return {
            "crepe_active":   True,
            "crepe_valid":    True,
            "f0_mean":        round(f0_mean, 1),
            "f0_std":         round(f0_std, 2),
            "f0_min":         round(f0_min, 1),
            "f0_max":         round(f0_max, 1),
            "f0_stability":   round(stability, 1),
            "range_semitone": round(st_range, 1),
            "voiced_ratio":   round(voiced, 1),
        }
    except Exception as e:
        print(f"[CREPE 오류] {e}")
        return {"crepe_active": True, "crepe_valid": False}


# ══════════════════════════════════════════════════════════
# 통합 분석
# ══════════════════════════════════════════════════════════
def analyze_voice(wav_path: str) -> dict:
    p = measure_parselmouth(wav_path)
    c = measure_crepe(wav_path)
    p.update(c)
    # CREPE F0가 유효하면 사용
    if c.get("crepe_valid"):
        p["f0_mean_used"] = c["f0_mean"]
    else:
        p["f0_mean_used"] = p.get("f0_mean_praat", 0)
    return p


# ══════════════════════════════════════════════════════════
# 8개 보컬 점수 산출 (직접 측정값 기반)
# ══════════════════════════════════════════════════════════
def lerp(val, lo, hi, higher=True):
    """ lo→0점, hi→100점 선형 보간 """
    if higher:
        if val >= hi: return 95
        if val <= lo: return 20
        return int(20 + (val - lo) / (hi - lo) * 75)
    else:
        if val <= lo: return 95
        if val >= hi: return 20
        return int(95 - (val - lo) / (hi - lo) * 75)


def compute_voice_scores(m: dict) -> dict:
    """
    8축 직접 측정 점수
    각 점수는 임상/음성과학 정상 범위 기준으로 산출
    """
    scores = {}

    # ① 성대 건강도 (HNR) — 정상 14~22dB
    scores["clarity"] = lerp(m.get("hnr", 0), 7, 22, True)

    # ② 음정 정밀도 (Jitter + CREPE F0 안정성)
    jit_score = lerp(m.get("jitter_pct", 0), 0.2, 1.5, False)
    if m.get("crepe_valid"):
        f0_score = lerp(m.get("f0_stability", 0), 70, 99, True)
        scores["pitch_precision"] = int((jit_score * 0.4 + f0_score * 0.6))
    else:
        scores["pitch_precision"] = jit_score

    # ③ 성량 일관성 (Shimmer) — 정상 1~6%
    scores["volume_consistency"] = lerp(m.get("shimmer_pct", 0), 1.0, 6.0, False)

    # ④ 발성 안정성 (CPP) — 정상 11~16dB
    scores["phonation_stability"] = lerp(m.get("cpp_db", 0), 6, 16, True)

    # ⑤ 비브라토 표현력 — 4~8Hz 영역 피크 강도
    vib_rate = m.get("vibrato_rate_hz", 0)
    vib_strength = m.get("vibrato_strength_pct", 0)
    if vib_rate >= 4.5 and vib_rate <= 7.5:
        # 이상적 비브라토 영역
        scores["vibrato_quality"] = int(min(95, 50 + vib_strength * 0.45))
    elif vib_rate > 0:
        scores["vibrato_quality"] = int(min(70, 35 + vib_strength * 0.35))
    else:
        scores["vibrato_quality"] = 30

    # ⑥ 음역 범위 (CREPE 반음 단위)
    if m.get("crepe_valid"):
        st = m.get("range_semitone", 0)
        scores["vocal_range"] = lerp(st, 3, 24, True)
    else:
        scores["vocal_range"] = None  # 측정 불가

    # ⑦ 공명 균형 (Singer's Formant — F3 영역, 2400~3200Hz가 이상적)
    f3 = m.get("f3_hz", 0)
    if f3 >= 2400 and f3 <= 3400:
        scores["resonance"] = int(75 + min(20, (f3 - 2400) / 50))
    elif f3 > 0:
        scores["resonance"] = int(40 + min(30, abs(f3 - 2800) / 50))
    else:
        scores["resonance"] = 30

    # ⑧ 발성 비율 (CREPE voiced_ratio)
    if m.get("crepe_valid"):
        vr = m.get("voiced_ratio", 0)
        scores["voiced_ratio"] = lerp(vr, 30, 90, True)
    else:
        scores["voiced_ratio"] = None

    return scores


# ══════════════════════════════════════════════════════════
# Claude AI 리포트
# ══════════════════════════════════════════════════════════
async def get_report(metrics: dict, scores: dict) -> dict:
    if not CLAUDE_API_KEY:
        return rule_report(metrics, scores)

    crepe_block = ""
    if metrics.get("crepe_valid"):
        crepe_block = f"""
[CREPE CNN 측정]
- 평균 F0: {metrics['f0_mean']}Hz
- F0 안정성: {metrics['f0_stability']}%
- 음역: {metrics['f0_min']}~{metrics['f0_max']}Hz ({metrics['range_semitone']} 반음)
- 유성음 비율: {metrics['voiced_ratio']}%"""

    prompt = f"""전문 보컬 트레이너로서 음성 분석 측정값을 바탕으로 뮤즐리 보이스 인바디 리포트 작성하세요.

[Parselmouth 음향 측정]
- HNR(성대건강): {metrics['hnr']}dB | 점수 {scores['clarity']}/100
- Jitter(음정떨림): {metrics['jitter_pct']}% | 점수 {scores['pitch_precision']}/100
- Shimmer(성량떨림): {metrics['shimmer_pct']}% | 점수 {scores['volume_consistency']}/100
- CPP(발성안정): {metrics['cpp_db']}dB | 점수 {scores['phonation_stability']}/100
- 비브라토: {metrics['vibrato_rate_hz']}Hz, 강도 {metrics['vibrato_strength_pct']}% | 점수 {scores['vibrato_quality']}/100
- Formant F1:{metrics['f1_hz']}Hz / F2:{metrics['f2_hz']}Hz / F3:{metrics['f3_hz']}Hz
- 공명 점수: {scores['resonance']}/100
- 녹음길이: {metrics['duration']}초{crepe_block}

JSON만 출력 (다른텍스트 절대 금지):
{{"voice_type":"창의적 보컬 타입","overall_grade":"S/A/B/C/D","summary":"측정값 기반 2문장 총평","strengths":["측정값 근거 강점1","강점2"],"improvements":["수치 기반 개선점1","개선점2"],"training_tip":"측정 결과 기반 구체적 훈련 1가지","celebrity_voice":"비슷한 가수"}}"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": CLAUDE_API_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 700,
                      "messages": [{"role": "user", "content": prompt}]}
            )
        raw = resp.json()["content"][0]["text"].strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[Claude오류] {e}")
        return rule_report(metrics, scores)


def rule_report(m, s):
    valid_scores = [v for v in s.values() if v is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 50
    g = "S" if avg>=88 else "A" if avg>=75 else "B" if avg>=60 else "C" if avg>=45 else "D"
    return {
        "voice_type": "보이스 분석 완료",
        "overall_grade": g,
        "summary": f"HNR {m.get('hnr',0)}dB, CPP {m.get('cpp_db',0)}dB로 측정됐습니다. 종합 {int(avg)}점.",
        "strengths": ["측정 완료"],
        "improvements": ["꾸준한 발성 훈련 권장"],
        "training_tip": "매일 15분 허밍으로 성대 건강을 유지하세요.",
        "celebrity_voice": "-"
    }


# ══════════════════════════════════════════════════════════
# 엔드포인트
# ══════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "MUZLY API v4", "crepe": CREPE_OK,
            "axes": ["clarity", "pitch_precision", "volume_consistency",
                     "phonation_stability", "vibrato_quality", "vocal_range",
                     "resonance", "voiced_ratio"]}

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
    for ext in [".wav",".mp3",".m4a",".ogg",".webm",".mp4"]:
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
        scores  = compute_voice_scores(metrics)
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
