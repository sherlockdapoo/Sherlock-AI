# Sherlock Balanced Learning curriculum v0.18.3

## Purpose

This revision preserves the six-family balanced curriculum and adds more supervised practice for the weak letter/instruction task. The current answer-quality benchmark scored 0/6 on alphabetic ordering and 6/6 on maths. This is a data-only learning revision; the neural architecture, parameter count, optimizer defaults, deterministic maths path, and graph code are unchanged.

## Dataset composition

- Base training set: 15,000 examples from v0.18.2.
- Additional training set: 3,000 examples.
- Total training set: **18,000 examples**.
- Base held-out validation set: 1,500 examples from v0.18.2.
- Additional held-out validation set: 60 examples.
- Total held-out validation set: **1,560 examples**.
- Six task families: **3,000 training / 260 validation examples each**.
- The supplement adds 500 training examples and 10 held-out validation examples to every family, preserving equal family totals.

## Targeted letter/instruction improvement

All 500 new letter/instruction training examples and all 10 new held-out letter/instruction examples practise ordering five distinct lowercase letters alphabetically. Inputs use the same instruction wording as the answer-quality benchmark, with independently generated letter sets. The exact benchmark letter sets are not copied into the supplemental examples.

## Other supplementary families

The other five families each receive 500 new training examples and 10 held-out validation examples:
1. Arithmetic (addition).
2. Percentages (percentage of an amount).
3. Unit conversions.
4. Word problems (addition, multiplication, subtraction, and equal sharing).
5. Number sequences (increasing and decreasing arithmetic sequences).

## Quality checks

- 3,000 supplementary training examples.
- 60 supplementary validation examples.
- 500 training examples per family in the supplement.
- 10 validation examples per family in the supplement.
- No duplicate examples within the generated supplement.
- No overlap between the generated training and validation supplements.
- Validation supplement size: 4,659 bytes, within the existing 128 KiB validation limit.
- Existing v0.18.2 data files remain unchanged.
- Existing v0.18.2 checkpoints remain in their original storage namespace; v0.18.3 uses separate checkpoint keys and dataset revision.

## Limitations

The supplemental data is synthetic and narrowly targets the current observed weakness. A higher validation score does not prove broad language understanding. The answer-quality benchmark and additional unseen instruction-following tests must be rerun after training.