# Sherlock AI

Sherlock AI is a small character-by-character learning experiment.

## Where the code lives

- `index.html` — the browser interface published with GitHub Pages.
- `backend/main.py` — the Python FastAPI training engine.
- `backend/requirements.txt` — Python server dependencies.
- `backend/README.md` — notes for running the Python API.

The Python engine is in the repository, but the GitHub Pages website does not run Python by itself. The next deployment step is to host `backend/main.py` as a web service (for example, on Render) and then connect the page to its API.

## Accuracy

The Accuracy bar reports the model's held-out next-character prediction accuracy. It is a small experiment metric, not a measure of how intelligent the model is.
