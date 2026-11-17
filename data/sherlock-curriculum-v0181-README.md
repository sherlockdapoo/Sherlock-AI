# Sherlock Balanced Learning curriculum v0.18.1

## Purpose of this revision

This revision preserves the six-family curriculum and repairs the number-sequence subtype mix. In v0.18.0, some alternating-increment rows used equal consecutive increments, making them indistinguishable from ordinary arithmetic sequences. v0.18.1 regenerates all sequence rows with five distinct patterns and exact subtype/template counts.

## Dataset design

- Training set: **12,000** single-turn examples.
- Held-out validation set: **1,200** separate single-turn examples.
- Six task families: **2,000 training / 200 validation examples each**.
- Five skill subtypes per family: **400 training / 40 validation examples each**.
- Four wording variants per subtype: **100 training / 10 validation examples per wording variant**.
- Training and validation prompts are unique within and across splits.
- Non-sequence examples are byte-for-byte unchanged from v0.18.0.

### Number-sequence subtype balance

| Pattern | Training | Validation |
|---|---:|---:|
| Increasing arithmetic progression | 400 | 40 |
| Decreasing arithmetic progression | 400 | 40 |
| Doubling | 400 | 40 |
| Alternating increments (first and third increments equal; middle increment different) | 400 | 40 |
| Consecutive square numbers | 400 | 40 |
| **Total** | **2,000** | **200** |

Each of the four sequence wording templates appears **500 times in training** and **50 times in validation**. Within each subtype, every wording template appears **100 times in training** and **10 times in validation**.

## Quality checks performed

| Check | Result |
|---|---:|
| Training examples | 12,000 |
| Validation examples | 1,200 |
| Training family balance | 2,000 per family |
| Validation family balance | 200 per family |
| Sequence subtype balance | 400 / 40 per subtype |
| Sequence wording-template balance | 100 / 10 per subtype-template pair |
| Duplicate training prompts | 0 |
| Duplicate validation prompts | 0 |
| Train/validation prompt overlap | 0 |
| Duplicate sequence number-tuples across splits | 0 |
| Ambiguous or misclassified generated sequence patterns | 0 |
| Incorrect generated sequence answers | 0 |
| Malformed or empty user/assistant turns | 0 |
| Changed non-sequence examples | 0 |
| Validation file size | 95,426 bytes (app limit: 131,072) |
| Maximum conversation size | 117 characters |

All generated sequence patterns were classified from their four displayed terms, and their next terms were recalculated from those terms. Every dataset block was checked for one non-empty user turn followed by one non-empty Sherlock answer. Family totals use the same six prompt-family detection expressions as the Neural Lab loader.

## Training and checkpoint handling

The active website is v0.20.1 and uses Balanced Learning v0.18.1. This revision has a new dataset cache revision and separate latest, best, and archive checkpoint keys. Existing v0.18.0 dataset files and browser checkpoints are not overwritten or deleted. The new revision starts with its own checkpoint namespace so a model trained on v0.18.0 is not silently treated as trained on v0.18.1.

## Scope and limitations

This is a synthetic skills curriculum covering six deliberately selected task families, not a broad conversational corpus. Balanced counts improve coverage of the designed patterns; they do not guarantee the model has learned them. Validation scores measure next-token prediction and should be supplemented with fresh, human-written tests.
