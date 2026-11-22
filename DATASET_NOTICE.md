# Sherlock AI training data notice

## Active Neural Lab dataset: Sherlock knowledge curriculum v0.19.1

The active curriculum contains **24,751 training conversations** and **1,795 held-out validation conversations** across seven task families:

- Arithmetic: 3,900 training / 286 validation
- Percentages: 3,700 / 266
- Unit conversions: 3,800 / 276
- Word problems: 3,800 / 276
- Number sequences: 3,800 / 276
- Letter/instruction tasks: 5,000 / 300
- Conversation and factual knowledge: 751 / 115

The v0.18.7 conversation supplement replaces the earlier v0.18.6 set because that set reused identical user prompts with different assistant responses. The replacement keeps the same 700 training and 100 held-out examples, but gives every example a unique user prompt. The loader now checks prompt uniqueness within each split and rejects any prompt shared between training and validation.

The v0.19.0 factual knowledge supplement remains included (51 training / 15 validation examples). Training and validation contain no exact conversation overlap.

The curriculum is original synthetic content. No external dataset samples are downloaded or included.
