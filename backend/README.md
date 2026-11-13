# Sherlock AI Python training engine

This folder contains the Python API for the character-level starter model.

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

The API provides:
- `GET /health`
- `POST /train/start`
- `POST /train/stop`
- `GET /status`
- `GET /generate?seed=hi&length=40`

This is an experimental character-transition model, not a general-purpose language model. Its score is a simple next-character prediction check on a small held-out text sample.
