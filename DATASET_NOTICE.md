# Sherlock GPT training data notice

Sherlock GPT's browser-based training experiment can use two public conversation datasets: DailyDialog and a bounded sample of UltraChat 200k. The app downloads dataset rows through the Hugging Face Dataset Viewer API; the neural network's training calculations run locally in the user's browser.

## 1. DailyDialog

- Original dataset: https://huggingface.co/datasets/li2017dailydialog/daily_dialog
- Browser-readable mirror used by Sherlock: https://huggingface.co/datasets/agentlans/li2017dailydialog
- Paper: https://aclanthology.org/I17-1099/
- Licence label: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- Splits used: `train` for training and `validation` for held-out evaluation. The DailyDialog test split is not fetched.

Please cite the original paper:

Li, Yanran, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. “DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset.” Proceedings of the 8th International Joint Conference on Natural Language Processing (IJCNLP 2017), 2017.

Full licence: https://creativecommons.org/licenses/by-nc-sa/4.0/

## 2. UltraChat 200k

- Dataset: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- Dataset card: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- Licence label in the dataset card: MIT
- Splits used: `train_sft` for training and `test_sft` for held-out evaluation.
- Download cap: up to 1,200 training conversations and 250 evaluation conversations. The full dataset is not downloaded.

The dataset card describes UltraChat 200k as a filtered collection of multi-turn conversations. The app reads the `messages` column and includes user and assistant turns. System messages and other roles are ignored.

## How Sherlock combines the sources

- Training and evaluation splits stay separate. UltraChat's `test_sft` sample is not included in the training stream.
- The training stream stays within the existing 2 MB local engine buffer. When UltraChat loads, up to 1.2 MB is reserved for DailyDialog and about 0.8 MB for UltraChat.
- The combined held-out evaluation text is capped at 120 KB, with up to about 60 KB reserved for each source.
- If the UltraChat split metadata or row endpoint is unavailable, the app reports that condition and falls back to the original DailyDialog-only dataset budget. It does not silently claim UltraChat was loaded.
- Text is cleaned and flattened into a character stream. This is character-level next-character training, not token-level supervised instruction fine-tuning.

## Licence reminder

DailyDialog is labelled CC BY-NC-SA 4.0, including non-commercial and ShareAlike conditions. UltraChat's dataset card labels that dataset MIT. When using or redistributing outputs, adapted material, or a combined dataset, review the source terms and obtain appropriate permission where needed. This notice is a practical summary, not legal advice.
