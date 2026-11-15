# Sherlock GPT training data notice

Sherlock GPT uses bounded samples from four public English-language datasets. Dataset rows are downloaded through the Hugging Face Dataset Viewer API. Training runs locally in the browser's neural worker.

## 1. DailyDialog

- Original dataset: https://huggingface.co/datasets/li2017dailydialog/daily_dialog
- Browser-readable mirror: https://huggingface.co/datasets/agentlans/li2017dailydialog
- Splits: `train` for training and `validation` for held-out evaluation.
- Sample caps: up to 3,500 training rows and 500 validation rows.
- Licence label: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- Paper: Li et al., “DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset,” IJCNLP 2017.
- Full licence: https://creativecommons.org/licenses/by-nc-sa/4.0/

## 2. UltraChat 200k

- Dataset: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- Splits: `train_sft` for training and `test_sft` for held-out evaluation.
- Sample caps: up to 1,600 training conversations and 250 evaluation conversations.
- Licence label in the dataset card: MIT.
- Sherlock reads the `messages` column and includes user and assistant turns. System messages and other roles are ignored.

## 3. No Robots

- Dataset: https://huggingface.co/datasets/HuggingFaceH4/no_robots
- Splits: `train` for training and `test` for held-out evaluation.
- Sample caps: up to 3,500 training conversations and 500 evaluation conversations.
- Licence label in the dataset card: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- Sherlock reads the `messages` column and includes user and assistant turns. System messages and other roles are ignored.
- Dataset card: https://huggingface.co/datasets/HuggingFaceH4/no_robots

## 4. GSM8K

- Dataset: https://huggingface.co/datasets/openai/gsm8k
- Configuration: `main`.
- Splits: `train` for training and `test` for held-out evaluation.
- Sample caps: up to 1,000 training maths problems and 250 evaluation problems.
- The `question` and `answer` columns are formatted as user/Sherlock turns. Inline calculation annotations are converted to plain equation text, and the final-answer marker is converted to readable text.
- Licence label in the dataset card: MIT.
- Paper: Cobbe et al., “Training Verifiers to Solve Math Word Problems,” 2021.

## How Sherlock combines the sources

- The application samples across the available rows in each selected split. It does not simply take the first rows.
- Training examples and held-out evaluation examples remain separate. GSM8K uses its `test` split only for validation; it is not included in training.
- The combined training text is limited to 9 MiB, matching the existing local engine buffer. The combined held-out text is limited to 128 KiB.
- Each optional source can fail independently. The application reports which sources were available and continues with the others when enough valid data remains.
- Cleaned examples are flattened into a text stream and split into word, punctuation, space, and newline tokens. The tokenizer keeps the 1,021 most frequent training tokens plus three reserved IDs; rare or unseen tokens are mapped to an unknown-token ID. The active model predicts the next token from the preceding eight tokens; this is not full instruction fine-tuning.
- The new dataset revision is cached in browser storage. A model checkpoint from the earlier dataset mix is archived in this browser and a fresh model is trained against the expanded data.

## Licence reminder

DailyDialog is labelled CC BY-NC-SA 4.0 and No Robots CC BY-NC 4.0. These non-commercial conditions matter when reusing or redistributing source data or adaptations. UltraChat 200k and GSM8K are labelled MIT in their dataset cards. Review the original dataset cards and full licences before redistribution. This notice is a practical summary, not legal advice.
