# Sherlock Balanced Learning curriculum v0.18.5

## Purpose
This revision adds 3,000 training examples and 60 held-out examples, with a focused increase in exact letter preservation and alphabetical sorting.

## Dataset composition
- Previous curriculum v0.18.4: 21,000 training / 1,620 validation conversations.
- New supplement: 3,000 training / 60 held-out validation conversations.
- Combined total: **24,000 training / 1,680 held-out validation conversations**.
- New training supplement: 1,500 letter/instruction examples and 300 examples in each of the other five families.
- New validation supplement: 30 held-out letter/instruction examples and 6 examples in each other family.

## Letter-ordering practice
- Five distinct lowercase letters per task.
- Four equivalent prompt phrasings, including the benchmark wording.
- Output contains exactly the same five letters, sorted alphabetically.
- The new supplement prompts are checked for uniqueness against the previous v0.18.2, v0.18.3, and v0.18.4 files and between training and validation.

## Expected combined family counts
Training: arithmetic 3,900; percentages 3,700; unit conversions 3,800; word problems 3,800; number sequences 3,800; letter/instruction tasks 5,000.
Validation: arithmetic 286; percentages 266; unit conversions 276; word problems 276; number sequences 276; letter/instruction tasks 300.

## Model and limitations
The model architecture, 70,704 parameters, tokenizer design, optimizer, and deterministic maths path are unchanged. The extra examples give the small word-level model more practice; they do not guarantee algorithmic sorting or broad language understanding. Evaluate with fresh, unseen prompts after training.
