from collections import defaultdict, Counter
from threading import Thread, Event, Lock
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import time

app = FastAPI(title="Sherlock AI Training Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Small starter corpus. Later, users can provide their own permitted training text.
TRAIN_TEXT = (
    "sherlock is learning one character at a time. "
    "a small model learns patterns from examples. "
    "training means counting which characters tend to follow other characters. "
    "this first engine is an experiment, not a thinking assistant. "
) * 12
TEST_TEXT = (
    "a model can learn patterns from text. "
    "sherlock is a small character learning experiment. "
)

lock = Lock()
stop_event = Event()
model = defaultdict(Counter)
training = False
epochs = 0
score = 0.0
worker = None

def build_counts(text):
    counts = defaultdict(Counter)
    for left, right in zip(text, text[1:]):
        counts[left][right] += 1
    return counts

def accuracy(counts, text):
    correct = 0
    total = 0
    for left, right in zip(text, text[1:]):
        choices = counts.get(left)
        if choices:
            prediction = choices.most_common(1)[0][0]
            correct += prediction == right
            total += 1
    return round((correct / total) * 100, 1) if total else 0.0

def train_loop():
    global training, epochs, score, model
    # Build the model once from training data. Further epochs measure it repeatedly
    # rather than pretending duplicate passes create new knowledge.
    counts = build_counts(TRAIN_TEXT)
    with lock:
        model = counts
        score = accuracy(model, TEST_TEXT)
    while not stop_event.wait(0.5):
        with lock:
            epochs += 1
            score = accuracy(model, TEST_TEXT)
    with lock:
        training = False

class StartRequest(BaseModel):
    reset: bool = True

@app.get("/health")
def health():
    return {"ok": True, "engine": "python-character-markov"}

@app.post("/train/start")
def start_training(request: StartRequest):
    global training, epochs, score, model, worker
    with lock:
        if training:
            return {"status": "already_training"}
        if request.reset:
            model = defaultdict(Counter)
            epochs = 0
            score = 0.0
        stop_event.clear()
        training = True
        worker = Thread(target=train_loop, daemon=True)
        worker.start()
    return {"status": "training_started"}

@app.post("/train/stop")
def stop_training():
    global training
    stop_event.set()
    return {"status": "stopping"}

@app.get("/status")
def status():
    with lock:
        return {
            "training": training,
            "epochs": epochs,
            "score": score,
            "model_ready": bool(model),
        }

@app.get("/generate")
def generate(seed: str = "hi", length: int = 40):
    length = max(1, min(length, 200))
    with lock:
        if not model:
            return {"error": "Train the model first."}
        local_model = {key: value.copy() for key, value in model.items()}
    output = seed[:20]
    current = output[-1:] or " "
    for _ in range(length):
        choices = local_model.get(current)
        if not choices:
            current = random.choice(list(local_model.keys()))
            output += current
            continue
        letters, weights = zip(*choices.items())
        current = random.choices(letters, weights=weights, k=1)[0]
        output += current
    return {"text": output}
