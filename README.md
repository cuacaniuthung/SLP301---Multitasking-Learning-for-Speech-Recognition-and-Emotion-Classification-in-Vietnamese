# ViPsyEmo

Nhận diện cảm xúc từ giọng nói (SER) theo Russell's Circumplex Model, kết hợp ASR + LLM phân tích.

## Cấu trúc

```
ViPsyEmo/
├── app.py              # Demo Gradio: ASR + SER + LLM
├── train_emotion.py    # Huấn luyện EmotionClassifier
├── inference.py        # Load checkpoint & predict
├── model.py            # Wav2Vec2 + Statistical Pooling + MLP
├── dataset.py          # EmotionDataset + pad_collate
├── config.py           # Config + emotion mapping
├── utils.py            # Training helpers
├── requirements.txt
├── data/               # metadata.csv + audio
└── checkpoints/        # best_model.pt
```

## Emotion labels

| ID | EN       | VI        | Quadrant               |
|----|----------|-----------|------------------------|
| 0  | anxious  | Lo lắng   | low valence, high arousal |
| 1  | happy    | Vui vẻ    | high valence, high arousal |
| 2  | sad      | Buồn bã   | low valence, low arousal  |
| 3  | calm     | Bình tĩnh | high valence, low arousal |

## Setup

```bash
pip install -r requirements.txt
```

Đặt `data/metadata.csv` với các cột: `path`, `valence`, `arousal`.

## Train

```bash
python train_emotion.py
```

Checkpoint lưu tại `checkpoints/best_model.pt`.

## Demo

```powershell
$env:GROQ_API_KEY="your_key"
python app.py
```

## Module roles

- **config.py** — hyperparams, `to_label()`, ID ↔ emotion
- **dataset.py** — load/resample/augment audio
- **model.py** — `EmotionClassifier` (frozen Wav2Vec2 + head)
- **utils.py** — class weights, epoch loop, report
- **train_emotion.py** — full training pipeline
- **inference.py** — production predict API
- **app.py** — Gradio UI (Whisper + trained SER + Groq)
