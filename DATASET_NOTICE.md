# Sherlock GPT training data notice

## Active Neural Lab dataset: Balanced Learning v0.18.0

The active main Neural Lab uses an original synthetic curriculum generated specifically for Sherlock AI. The two standalone files are:

- `data/sherlock-curriculum-v0180-train.txt`: 12,000 training examples.
- `data/sherlock-curriculum-v0180-validation.txt`: 1,200 held-out validation examples.

No external dataset samples are downloaded or included in this active curriculum.

The curriculum contains six task families, with exactly 2,000 training and 200 validation examples in each family. Each family has five equally sized subtypes and four wording variants per subtype. The training and validation splits have the same template distribution and use the same numeric ranges, but are generated independently and have no exact prompt overlap.

See [the v0.18.0 quality report](data/sherlock-curriculum-v0180-README.md) for the detailed family/subtype breakdown, byte sizes, and checks.

## Historical data references

Earlier code revisions and isolated experiments may refer to DailyDialog, UltraChat 200k, No Robots, and GSM8K. These sources are not fetched by the active v0.20.0 standalone curriculum. Their original licence terms still apply to any archived copies or derivatives that someone chooses to reuse; consult the original dataset cards and licences before redistribution.

The active curriculum is synthetic and does not carry those external dataset licence conditions. This notice is a practical summary, not legal advice.
