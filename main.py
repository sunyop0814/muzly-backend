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

# ── librosa 로드 (다이나믹 레인지용) ──────────────────────
try:
    import librosa
    LIBROSA_OK = True
    print("[librosa] 로드 성공")
except Exception as e:
    LIBROSA_OK = False
    print(f"[librosa] 비활성화: {e}")

app = FastAPI(title="MUZLY Voice Analysis API v5")
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
# 성부 분류 (Voice Type Classification)
# ══════════════════════════════════════════════════════════
def hz_to_note(hz: float) -> str:
    """Hz → 음표 표기 (예: 220Hz → A3)"""
    if hz <= 0: return "-"
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    n = 12 * np.log2(hz / 440.0) + 49  # A4 = MIDI 49 in this scheme
    midi = int(round(12 * np.log2(hz / 440.0) + 69))  # 표준 MIDI
    octave = midi // 12 - 1
    name = note_names[midi % 12]
    return f"{name}{octave}"

def classify_voice_type(f0_mean: float, f0_min: float, f0_max: float, gender_hint: str = "auto") -> dict:
    """
    F0 평균 + 음역으로 성부 자동 분류
    참고: Vocal Pedagogy 표준 분류
    """
    if f0_mean <= 0:
        return {"type": "측정 불가", "type_en": "Unknown", "confidence": 0}

    # 평균 F0 기반 1차 분류
    candidates = []
    if 80 <= f0_mean <= 130:
        candidates.append(("베이스", "Bass", 80, 130))
    if 100 <= f0_mean <= 165:
        candidates.append(("바리톤", "Baritone", 100, 165))
    if 130 <= f0_mean <= 220:
        candidates.append(("테너", "Tenor", 130, 220))
    if 175 <= f0_mean <= 260:
        candidates.append(("알토", "Alto", 175, 260))
    if 220 <= f0_mean <= 330:
        candidates.append(("메조소프라노", "Mezzo-Soprano", 220, 330))
    if 260 <= f0_mean <= 440:
        candidates.append(("소프라노", "Soprano", 260, 440))

    if not candidates:
        # 매우 낮거나 매우 높은 경우
        if f0_mean < 80:
            return {"type": "베이스", "type_en": "Bass", "confidence": 70}
        else:
            return {"type": "소프라노", "type_en": "Soprano", "confidence": 70}

    # 평균에 가장 가까운 성부 선택
    best = min(candidates, key=lambda c: abs(f0_mean - (c[2] + c[3]) / 2))
    name_kr, name_en, lo, hi = best

    # 신뢰도: 음역이 해당 성부 범위와 얼마나 일치하는가
    overlap_lo = max(f0_min, lo)
    overlap_hi = min(f0_max, hi)
    overlap = max(0, overlap_hi - overlap_lo)
    user_range = max(f0_max - f0_min, 1)
    target_range = hi - lo
    confidence = int(min(95, (overlap / target_range) * 60 + 35))

    return {
        "type":     name_kr,
        "type_en":  name_en,
        "confidence": confidence,
        "range_low_note":  hz_to_note(f0_min),
        "range_high_note": hz_to_note(f0_max),
    }


# ══════════════════════════════════════════════════════════
# 음정 정확도 (Cents 단위)
# ══════════════════════════════════════════════════════════
def measure_pitch_accuracy_cents(f0_array: np.ndarray) -> dict:
    """
    각 F0 값을 가장 가까운 표준 음정과 비교 → cents 오차
    1 cent = 1/100 반음. 50 cents 이상 어긋나면 다른 음.
    """
    if len(f0_array) < 5:
        return {"avg_cents_error": 0, "max_cents_error": 0, "in_tune_ratio": 0}

    valid = f0_array[f0_array > 50]
    if len(valid) < 5:
        return {"avg_cents_error": 0, "max_cents_error": 0, "in_tune_ratio": 0}

    # 각 F0를 가장 가까운 반음 음표로 매핑 (cents 단위 오차)
    midi_float = 69 + 12 * np.log2(valid / 440.0)
    nearest = np.round(midi_float)
    cents_error = (midi_float - nearest) * 100  # cents 단위
    abs_error = np.abs(cents_error)

    # 25 cents 이내면 "정확", 50 cents 넘으면 "음정 이탈"
    in_tune = (abs_error < 25).sum()
    in_tune_ratio = round(in_tune / len(valid) * 100, 1)

    return {
        "avg_cents_error":  round(float(abs_error.mean()), 1),
        "max_cents_error":  round(float(abs_error.max()),  1),
        "in_tune_ratio":    in_tune_ratio,  # 25 cents 이내 비율 %
    }


# ══════════════════════════════════════════════════════════
# 다이나믹 레인지 (Dynamic Range)
# ══════════════════════════════════════════════════════════
def measure_dynamics(wav_path: str) -> dict:
    """
    시간대별 RMS 음량 측정 → 다이나믹 레인지
    """
    if not LIBROSA_OK:
        return {"dynamics_available": False}

    try:
        audio, sr = sf.read(wav_path, dtype='float32') if CREPE_OK else (None, None)
        if audio is None:
            audio, sr = librosa.load(wav_path, sr=None, mono=True)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 무음 제거 (-40dB 이하)
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)  # 0dB 기준

        # 무음 구간 제외
        voiced_mask = rms_db > -40
        if voiced_mask.sum() < 10:
            return {"dynamics_available": True, "dynamics_valid": False}

        rms_voiced = rms_db[voiced_mask]
        max_db = float(rms_voiced.max())
        min_db = float(rms_voiced.min())
        mean_db = float(rms_voiced.mean())
        dyn_range = round(max_db - min_db, 2)

        # 호흡 안정성: dB 표준편차 (작을수록 일관)
        std_db = float(rms_voiced.std())
        breath_stability = max(0, min(100, int(100 - std_db * 6)))

        # Crest Factor (피크 대 평균 비율)
        peak = float(np.max(np.abs(audio)))
        rms_lin = float(np.sqrt(np.mean(audio**2)) + 1e-8)
        crest_factor = round(20 * np.log10(peak / rms_lin), 2)

        return {
            "dynamics_available": True,
            "dynamics_valid":     True,
            "dynamic_range_db":   dyn_range,
            "rms_max_db":         round(max_db, 2),
            "rms_min_db":         round(min_db, 2),
            "rms_mean_db":        round(mean_db, 2),
            "breath_stability":   breath_stability,
            "crest_factor_db":    crest_factor,
        }
    except Exception as e:
        print(f"[다이나믹 오류] {e}")
        return {"dynamics_available": True, "dynamics_valid": False}


# ══════════════════════════════════════════════════════════
# Parselmouth 직접 측정 (검증된 8개 지표)
# ══════════════════════════════════════════════════════════
def measure_parselmouth(wav: str) -> dict:
    snd = parselmouth.Sound(wav)
    duration = snd.duration
    out = {"duration": round(duration, 2)}

    # HNR
    try:
        h = praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        v = praat.call(h, "Get mean", 0, 0)
        out["hnr"] = round(float(v), 2) if np.isfinite(v) else 0.0
    except: out["hnr"] = 0.0

    # Jitter
    pp = None
    try:
        pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
        v = praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        out["jitter_pct"] = round(float(v) * 100, 4) if np.isfinite(v) else 0.0
    except: out["jitter_pct"] = 0.0

    # Shimmer
    try:
        if pp is None:
            pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)
        v = praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        out["shimmer_pct"] = round(float(v) * 100, 4) if np.isfinite(v) else 0.0
    except: out["shimmer_pct"] = 0.0

    # F0 + 비브라토
    try:
        pitch = praat.call(snd, "To Pitch", 0.0, 75, 600)
        f0_mean = praat.call(pitch, "Get mean", 0, 0, "Hertz")
        f0_std  = praat.call(pitch, "Get standard deviation", 0, 0, "Hertz")
        out["f0_mean_praat"] = round(float(f0_mean), 1) if np.isfinite(f0_mean) else 0.0
        out["f0_std_praat"]  = round(float(f0_std), 2) if np.isfinite(f0_std) else 0.0

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

    # CPP
    try:
        pc = praat.call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        cpp = praat.call(pc, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05,
                         "Parabolic", 0.001, 0, "Exponential decay", "Robust")
        out["cpp_db"] = round(float(cpp), 2) if np.isfinite(cpp) else 0.0
    except: out["cpp_db"] = 0.0

    # Formant
    try:
        fmt = praat.call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        f1 = praat.call(fmt, "Get mean", 1, 0, 0, "Hertz")
        f2 = praat.call(fmt, "Get mean", 2, 0, 0, "Hertz")
        f3 = praat.call(fmt, "Get mean", 3, 0, 0, "Hertz")
        out["f1_hz"] = round(float(f1), 0) if np.isfinite(f1) else 0.0
        out["f2_hz"] = round(float(f2), 0) if np.isfinite(f2) else 0.0
        out["f3_hz"] = round(float(f3), 0) if np.isfinite(f3) else 0.0
    except: out["f1_hz"], out["f2_hz"], out["f3_hz"] = 0.0, 0.0, 0.0

    return out


# ══════════════════════════════════════════════════════════
# CREPE 측정 (CNN F0 + 음정 정확도 cents)
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

        # 음정 정확도 cents
        accuracy = measure_pitch_accuracy_cents(f0v)

        # 옥타브 단위 음역
        octaves = round(st_range / 12, 2)

        return {
            "crepe_active":   True,
            "crepe_valid":    True,
            "f0_mean":        round(f0_mean, 1),
            "f0_std":         round(f0_std, 2),
            "f0_min":         round(f0_min, 1),
            "f0_max":         round(f0_max, 1),
            "f0_stability":   round(stability, 1),
            "range_semitone": round(st_range, 1),
            "range_octave":   octaves,
            "voiced_ratio":   round(voiced, 1),
            "low_note":       hz_to_note(f0_min),
            "high_note":      hz_to_note(f0_max),
            "mean_note":      hz_to_note(f0_mean),
            "avg_cents_error":  accuracy["avg_cents_error"],
            "max_cents_error":  accuracy["max_cents_error"],
            "in_tune_ratio":    accuracy["in_tune_ratio"],
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
    d = measure_dynamics(wav_path)
    p.update(c); p.update(d)

    # F0 평균 결정
    if c.get("crepe_valid"):
        p["f0_mean_used"] = c["f0_mean"]
    else:
        p["f0_mean_used"] = p.get("f0_mean_praat", 0)

    # 성부 분류
    voice = classify_voice_type(
        f0_mean = p["f0_mean_used"],
        f0_min  = c.get("f0_min", p["f0_mean_used"] * 0.7),
        f0_max  = c.get("f0_max", p["f0_mean_used"] * 1.4),
    )
    p["voice_type"]            = voice["type"]
    p["voice_type_en"]         = voice["type_en"]
    p["voice_type_confidence"] = voice["confidence"]

    return p


# ══════════════════════════════════════════════════════════
# 점수 산출 (8축 + 다이나믹 레인지 = 9축)
# ══════════════════════════════════════════════════════════
def lerp(val, lo, hi, higher=True):
    if higher:
        if val >= hi: return 95
        if val <= lo: return 20
        return int(20 + (val - lo) / (hi - lo) * 75)
    else:
        if val <= lo: return 95
        if val >= hi: return 20
        return int(95 - (val - lo) / (hi - lo) * 75)

def compute_voice_scores(m: dict) -> dict:
    scores = {}
    scores["clarity"] = lerp(m.get("hnr", 0), 7, 22, True)

    # 음정 정밀도: Jitter + CREPE F0 안정성 + cents 정확도
    jit_score = lerp(m.get("jitter_pct", 0), 0.2, 1.5, False)
    if m.get("crepe_valid"):
        f0_score    = lerp(m.get("f0_stability", 0), 70, 99, True)
        cents_score = lerp(m.get("avg_cents_error", 50), 5, 30, False)
        scores["pitch_precision"] = int((jit_score * 0.3 + f0_score * 0.4 + cents_score * 0.3))
    else:
        scores["pitch_precision"] = jit_score

    scores["volume_consistency"]   = lerp(m.get("shimmer_pct", 0), 1.0, 6.0, False)
    scores["phonation_stability"]  = lerp(m.get("cpp_db", 0), 6, 16, True)

    vib_rate = m.get("vibrato_rate_hz", 0)
    vib_strength = m.get("vibrato_strength_pct", 0)
    if 4.5 <= vib_rate <= 7.5:
        scores["vibrato_quality"] = int(min(95, 50 + vib_strength * 0.45))
    elif vib_rate > 0:
        scores["vibrato_quality"] = int(min(70, 35 + vib_strength * 0.35))
    else:
        scores["vibrato_quality"] = 30

    if m.get("crepe_valid"):
        st = m.get("range_semitone", 0)
        scores["vocal_range"] = lerp(st, 3, 24, True)
    else:
        scores["vocal_range"] = None

    f3 = m.get("f3_hz", 0)
    if 2400 <= f3 <= 3400:
        scores["resonance"] = int(75 + min(20, (f3 - 2400) / 50))
    elif f3 > 0:
        scores["resonance"] = int(40 + min(30, abs(f3 - 2800) / 50))
    else:
        scores["resonance"] = 30

    if m.get("crepe_valid"):
        vr = m.get("voiced_ratio", 0)
        scores["voiced_ratio"] = lerp(vr, 30, 90, True)
    else:
        scores["voiced_ratio"] = None

    # 다이나믹 레인지
    if m.get("dynamics_valid"):
        dr = m.get("dynamic_range_db", 0)
        scores["dynamics"] = lerp(dr, 6, 25, True)
    else:
        scores["dynamics"] = None

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
[CREPE CNN]
- 평균 F0: {metrics['f0_mean']}Hz ({metrics['mean_note']})
- 음역: {metrics['low_note']}~{metrics['high_note']} ({metrics['range_octave']} 옥타브)
- F0 안정성: {metrics['f0_stability']}%
- 평균 cents 오차: {metrics['avg_cents_error']} (25 이내 비율 {metrics['in_tune_ratio']}%)
- 유성음 비율: {metrics['voiced_ratio']}%"""

    dyn_block = ""
    if metrics.get("dynamics_valid"):
        dyn_block = f"""
[다이나믹]
- 다이나믹 레인지: {metrics['dynamic_range_db']}dB
- 호흡 안정성: {metrics['breath_stability']}/100
- Crest Factor: {metrics['crest_factor_db']}dB"""

    prompt = f"""전문 보컬 트레이너로서 음성 측정값을 바탕으로 뮤즐리 보이스 인바디 리포트 작성하세요.

[성부]
- 분류: {metrics.get('voice_type','-')} ({metrics.get('voice_type_en','-')})
- 신뢰도: {metrics.get('voice_type_confidence',0)}%

[Parselmouth]
- HNR(성대건강): {metrics['hnr']}dB | 점수 {scores['clarity']}/100
- Jitter(음정떨림): {metrics['jitter_pct']}% | 점수 {scores['pitch_precision']}/100
- Shimmer(성량떨림): {metrics['shimmer_pct']}% | 점수 {scores['volume_consistency']}/100
- CPP(발성안정): {metrics['cpp_db']}dB | 점수 {scores['phonation_stability']}/100
- 비브라토: {metrics['vibrato_rate_hz']}Hz, 강도 {metrics['vibrato_strength_pct']}% | 점수 {scores['vibrato_quality']}/100
- Formant F1:{metrics['f1_hz']}Hz / F2:{metrics['f2_hz']}Hz / F3:{metrics['f3_hz']}Hz
- 공명 점수: {scores['resonance']}/100
- 녹음길이: {metrics['duration']}초{crepe_block}{dyn_block}

JSON만 출력 (다른텍스트 절대 금지):
{{"voice_type":"창의적 보컬 타입 이름","overall_grade":"S/A/B/C/D","summary":"성부와 측정값 기반 2문장 총평","strengths":["측정값 근거 강점1","강점2"],"improvements":["수치 기반 개선점1","개선점2"],"training_tip":"측정 결과 기반 구체적 훈련 1가지","celebrity_voice":"비슷한 가수"}}"""

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
        "voice_type": m.get("voice_type", "보이스 분석 완료"),
        "overall_grade": g,
        "summary": f"{m.get('voice_type','-')} 분류, HNR {m.get('hnr',0)}dB로 측정. 종합 {int(avg)}점.",
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
    return {"status": "MUZLY API v5", "crepe": CREPE_OK, "librosa": LIBROSA_OK,
            "features": ["성부 분류", "음역 옥타브", "cents 음정 정확도", "다이나믹 레인지",
                         "9축 보컬 인바디"]}

@app.get("/health")
def health():
    return {"ok": True, "crepe": CREPE_OK, "librosa": LIBROSA_OK}

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
        if not ok: raise HTTPException(500, "오디오 변환 실패")

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
