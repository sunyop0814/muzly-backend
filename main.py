from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import parselmouth
import parselmouth.praat as praat
import numpy as np
import tempfile
import os
import httpx
import json

app = FastAPI(title="MUZLY Voice Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

def analyze_voice(wav_path: str) -> dict:
    snd = parselmouth.Sound(wav_path)
    duration = snd.duration

    # ── HNR (성대 건강도) ──────────────────────
    try:
        harmonicity = praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = praat.call(harmonicity, "Get mean", 0, 0)
        hnr = round(float(hnr), 2) if not np.isnan(hnr) else 0.0
    except:
        hnr = 0.0

    # ── Jitter (음정 떨림) ─────────────────────
    try:
        pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = praat.call([snd, pp], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter = round(float(jitter) * 100, 3) if not np.isnan(jitter) else 0.0
    except:
        jitter = 0.0
        pp = None

    # ── Shimmer (성량 떨림) ────────────────────
    try:
        if pp is None:
            pp = praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        shimmer = praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer = round(float(shimmer) * 100, 3) if not np.isnan(shimmer) else 0.0
    except:
        shimmer = 0.0

    # ── F0 평균 (평균 음역) ────────────────────
    try:
        pitch = praat.call(snd, "To Pitch", 0, 75, 500)
        f0_mean = praat.call(pitch, "Get mean", 0, 0, "Hertz")
        f0_mean = round(float(f0_mean), 1) if not np.isnan(f0_mean) else 0.0
    except:
        f0_mean = 0.0

    # ── Formant F1/F2 (발성 위치) ─────────────
    try:
        formant = praat.call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        f1 = praat.call(formant, "Get mean", 1, 0, 0, "Hertz")
        f2 = praat.call(formant, "Get mean", 2, 0, 0, "Hertz")
        f1 = round(float(f1), 1) if not np.isnan(f1) else 0.0
        f2 = round(float(f2), 1) if not np.isnan(f2) else 0.0
    except:
        f1, f2 = 0.0, 0.0

    return {
        "duration": round(duration, 1),
        "hnr": hnr,
        "jitter": jitter,
        "shimmer": shimmer,
        "f0_mean": f0_mean,
        "f1": f1,
        "f2": f2,
    }


def score(val, low_good, high_good, higher_is_better=True):
    """0~100점 변환"""
    if higher_is_better:
        if val >= high_good: return 95
        if val <= low_good: return 30
        return int(30 + (val - low_good) / (high_good - low_good) * 65)
    else:
        if val <= low_good: return 95
        if val >= high_good: return 30
        return int(95 - (val - low_good) / (high_good - low_good) * 65)


def compute_scores(m: dict) -> dict:
    return {
        "hnr_score":     score(m["hnr"],     7,  22, True),
        "jitter_score":  score(m["jitter"],  0,  1.0, False),
        "shimmer_score": score(m["shimmer"], 0,  6.0, False),
        "f0_score":      min(99, max(30, 75)) if m["f0_mean"] > 50 else 40,
    }


async def claude_report(metrics: dict, scores: dict) -> str:
    if not CLAUDE_API_KEY:
        return generate_rule_report(metrics, scores)

    prompt = f"""당신은 전문 보컬 트레이너입니다. 아래 음성 분석 수치를 보고 뮤즐리 앱의 '보이스 인바디' 리포트를 작성하세요.

[측정 수치]
- HNR(성대 건강도): {metrics['hnr']}dB (점수: {scores['hnr_score']}/100)
- Jitter(음정 떨림): {metrics['jitter']}% (점수: {scores['jitter_score']}/100)
- Shimmer(성량 떨림): {metrics['shimmer']}% (점수: {scores['shimmer_score']}/100)
- 평균 음역(F0): {metrics['f0_mean']}Hz
- 포먼트 F1: {metrics['f1']}Hz / F2: {metrics['f2']}Hz
- 녹음 길이: {metrics['duration']}초

[출력 형식 - 반드시 JSON으로만 응답]
{{
  "voice_type": "보이스 타입 이름 (예: 파워 보컬, 섬세한 리릭 등 창의적으로)",
  "overall_grade": "S/A/B/C/D 중 하나",
  "summary": "2문장 이내 전체 총평",
  "strengths": ["강점1", "강점2"],
  "improvements": ["개선점1", "개선점2"],
  "training_tip": "오늘 당장 할 수 있는 연습 팁 1가지",
  "celebrity_voice": "비슷한 보컬 스타일의 가수 이름"
}}"""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 800,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = resp.json()
        raw = data["content"][0]["text"].strip()
        # JSON 파싱
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw


def generate_rule_report(metrics: dict, scores: dict) -> str:
    avg = (scores["hnr_score"] + scores["jitter_score"] + scores["shimmer_score"]) / 3
    grade = "S" if avg >= 88 else "A" if avg >= 75 else "B" if avg >= 60 else "C" if avg >= 45 else "D"
    return json.dumps({
        "voice_type": "분석 완료 보컬",
        "overall_grade": grade,
        "summary": f"HNR {metrics['hnr']}dB, Jitter {metrics['jitter']}%로 측정됐습니다. 전체 점수 {int(avg)}점입니다.",
        "strengths": ["녹음 분석 완료", "수치 측정 성공"],
        "improvements": ["Claude API키를 연결하면 더 자세한 분석이 가능합니다"],
        "training_tip": "매일 15분 허밍 연습으로 성대 건강을 유지하세요.",
        "celebrity_voice": "분석 중"
    }, ensure_ascii=False)


@app.get("/")
def root():
    return {"status": "MUZLY API running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 파일 형식 확인
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".webm")):
        raise HTTPException(400, "wav, mp3, m4a, ogg, webm 파일만 지원합니다.")

    # 임시 파일로 저장
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    wav_path = tmp_path
    try:
        # webm/mp3/m4a → wav 변환 (ffmpeg 있을 때)
        if not suffix == ".wav":
            wav_path = tmp_path.replace(suffix, ".wav")
            ret = os.system(f"ffmpeg -y -i {tmp_path} {wav_path} -loglevel error")
            if ret != 0:
                wav_path = tmp_path  # 변환 실패 시 원본 시도

        # 음성 분석
        metrics = analyze_voice(wav_path)
        scores = compute_scores(metrics)

        # Claude 리포트
        report_raw = await claude_report(metrics, scores)
        try:
            report = json.loads(report_raw)
        except:
            report = {"voice_type": "분석완료", "overall_grade": "B",
                      "summary": report_raw[:200], "strengths": [], "improvements": [],
                      "training_tip": "", "celebrity_voice": ""}

        return {
            "success": True,
            "metrics": metrics,
            "scores": scores,
            "report": report
        }

    except Exception as e:
        raise HTTPException(500, f"분석 오류: {str(e)}")
    finally:
        for p in [tmp_path, wav_path]:
            try: os.unlink(p)
            except: pass
