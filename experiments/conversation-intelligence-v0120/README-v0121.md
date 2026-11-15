# Sherlock AI v0.12.1 curated curriculum experiment

This is an isolated continuation of the v0.12.0 conversation-intelligence experiment. It keeps the same neural architecture and adds a small, deterministic, locally generated assistant curriculum to the existing public dataset mix.

## What changed

- Added 430 short training conversation examples across basic arithmetic, elementary factual questions, instruction following, one-step reasoning, and short multi-turn recall.
- Added 20 separate held-out curriculum examples. These are not included in training.
- Added deterministic prompt variations for instruction examples and unique values for multi-turn recall examples.
- Kept the existing DailyDialog, UltraChat, No Robots, and GSM8K sources.
- Changed the dataset revision and checkpoint keys so v0.12.1 does not overwrite the v0.12.0 experiment checkpoint.
- Kept the model architecture unchanged: 1,024-token word vocabulary, context 32, embedding 8, hidden 48, and 70,704 parameters.

## Data checks

The curriculum generator was checked independently:

- 430 training examples generated.
- 20 held-out examples generated.
- 0 exact duplicate training blocks.
- 0 exact train/validation overlaps.
- Every generated block uses the user: and sherlock: format.
- The combined curriculum text is approximately 31.8 KB before UTF-8 encoding.

The page's inline JavaScript syntax was checked after the change.

## Important limitations

This is a controlled data experiment, not a claim that the model is now intelligent or that quality has improved. The curriculum is deliberately small because the model has limited capacity. The existing data sources are still broad and may contain examples that are too long or noisy for this model.

The worker lowercases input text, so it cannot reliably learn case-sensitive output requirements such as returning uppercase words exactly. This is a tokenizer limitation, not something that more examples alone will fix.

The next evaluation should use the same five-question benchmark as v0.12.0, plus the page's held-out validation metrics. Keep the stable model unchanged unless the new experiment shows repeatable improvements.