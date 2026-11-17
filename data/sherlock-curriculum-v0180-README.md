# Sherlock Balanced Learning curriculum v0.18.0

## Design

This is a locally stored, synthetic curriculum made specifically for Sherlock AI. It does not download or redistribute examples from external datasets.

- Training set: **12,000** single-turn examples.
- Held-out validation set: **1,200** independent single-turn examples.
- Six task families: **2,000 training / 200 validation examples each**.
- Each family contains five equally sized skill subtypes: **400 training / 40 validation examples per subtype**.
- Each subtype has four wording variants: **100 training / 10 validation examples per wording variant**.
- Training and validation use the same prompt-template mix and the same numeric ranges, but are generated from independent deterministic seeds. Prompts are unique across both splits.
- Numbers are tokenised digit by digit by the existing neural worker, allowing values not seen in training to reuse digit tokens.

## Task-family coverage

1. Arithmetic: addition, subtraction, multiplication, exact division, and two-step addition/multiplication.
2. Percentages: percentage of an amount, percentage increase, percentage decrease, finding a percentage, and discounted amount remaining.
3. Unit conversions: centimetres/millimetres, metres/centimetres, kilometres/metres, kilograms/grams, and time conversions.
4. Word problems: addition, subtraction, multiplication, equal sharing, and two-step contexts.
5. Number sequences: increasing arithmetic sequences, decreasing arithmetic sequences, doubling, alternating increments, and square sequences.
6. Letter/instruction tasks: ascending order, descending order, finding a letter within a word, moving through the alphabet, and alphabetic sorting.

## Quality checks run before commit

| Check | Result |
|---|---:|
| Training examples | 12,000 |
| Validation examples | 1,200 |
| Training family balance | 2,000 each |
| Validation family balance | 200 each |
| Training subtype balance | 400 each |
| Validation subtype balance | 40 each |
| Wording-variant balance | 500 training / 50 validation each |
| Duplicate training prompts | 0 |
| Duplicate validation prompts | 0 |
| Train/validation prompt overlap | 0 |
| Validation tokens absent from training vocabulary | 0 of 46,748 |
| Training file size | 948,651 bytes |
| Validation file size | 94,937 bytes (limit: 131,072 bytes) |
| Maximum validation conversation size | 117 bytes |
| Training vocabulary tokens | 275 |

All conversations were checked for exactly one non-empty `user:` turn and one non-empty `sherlock:` turn. All generated answer strings matched their calculated expected values, and every prompt was assigned to the intended family by the same family-detection rules used by Neural Lab.

## Important limitations

This is a **synthetic skills curriculum**, not a broad natural-language conversation corpus. It deliberately measures performance on these six task families. Matching validation templates and numeric ranges to training makes validation more representative of this curriculum, but does not prove general language ability. Results must still be checked with separate human-written tests.

The v0.17.0 files and browser model/checkpoint keys are left untouched. v0.18.0 uses a new dataset revision and separate latest, best, and archive checkpoint keys, so the old browser checkpoints are not overwritten or automatically loaded into the new curriculum.
