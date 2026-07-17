"""
app.py
=======
Demo Gradio: Speech → Text (ASR) + Emotion (SER) → Phân tích (LLM qua Groq)

Cài đặt:
    pip install -r requirements.txt

API key (Windows PowerShell):
    $env:GROQ_API_KEY="xxxxxxxx"

Chạy:
    python app.py
"""

import os
import re

import gradio as gr
import torch
from faster_whisper import WhisperModel
from groq import Groq

from inference import load_emotion_model, predict_emotion

USE_CUDA = torch.cuda.is_available()
print(f"Đang chạy trên: {'GPU (CUDA)' if USE_CUDA else 'CPU'}")

# --- 1. ASR: Whisper ---
asr_model = WhisperModel(
    "small",
    device="cuda" if USE_CUDA else "cpu",
    compute_type="float16" if USE_CUDA else "int8",
)

# --- 2. SER: EmotionClassifier đã fine-tune ---
emo_model, emo_processor, id2emo, emo2vi, emo_device = load_emotion_model()

# --- 3. LLM qua Groq ---
if not os.environ.get("GROQ_API_KEY"):
    raise RuntimeError(
        "Chưa có GROQ_API_KEY. "
        "PowerShell: $env:GROQ_API_KEY='...'"
    )

groq_client = Groq()
LLM_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "Bạn là trợ lý phân tích tâm lý. Dựa trên lời nói của người dùng (đã chuyển thành văn bản) "
    "và cảm xúc phát hiện từ giọng nói, hãy viết MỘT đoạn ngắn (3-5 câu) bằng tiếng Việt: "
    "tóm tắt vấn đề người dùng đang gặp và nhận định về trạng thái cảm xúc của họ. "
    "Viết tự nhiên, đồng cảm, KHÔNG gạch đầu dòng."
)


def _strip_reasoning(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_analysis(text: str, emotion_label: str) -> str:
    user_msg = (
        f'Lời nói của người dùng: "{text}"\n'
        f"Cảm xúc phát hiện từ giọng nói: {emotion_label}"
    )
    resp = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.6,
        reasoning_effort="low",
    )
    return _strip_reasoning(resp.choices[0].message.content)


def analyze(audio_path):
    if audio_path is None:
        return "Chưa có audio.", "-", "-"

    segments, _ = asr_model.transcribe(audio_path, beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    if not text:
        return "(không nghe ra lời)", "-", "-"

    result = predict_emotion(
        audio_path,
        emo_model,
        emo_processor,
        emo_device,
        id2emo=id2emo,
        emo2vi=emo2vi,
    )
    emotion_str = result["scores_str"]
    analysis = generate_analysis(text, result["label_vi"])

    return text, emotion_str, analysis


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="Ghi âm bằng mic hoặc tải file audio",
    ),
    outputs=[
        gr.Textbox(label="Văn bản (ASR)"),
        gr.Textbox(label="Cảm xúc (SER)"),
        gr.Textbox(label="Phân tích (LLM qua Groq)"),
    ],
    title="ViPsyEmo — Speech → Text + Emotion → Phân tích",
    description=(
        "Nói hoặc tải một đoạn audio. Hệ thống chuyển thành văn bản, "
        "dự đoán cảm xúc từ giọng nói (model ViPsyEmo), "
        "và viết một đoạn phân tích vấn đề + trạng thái cảm xúc."
    ),
)

if __name__ == "__main__":
    demo.launch()
