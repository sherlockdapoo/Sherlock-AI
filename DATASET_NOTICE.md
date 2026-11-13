# DailyDialog dataset notice

Sherlock AI's browser training experiment uses examples from **DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset** by Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu (IJCNLP 2017).

## Source

- Original dataset page: https://huggingface.co/datasets/li2017dailydialog/daily_dialog
- Browser-readable mirror used by Sherlock AI: https://huggingface.co/datasets/agentlans/li2017dailydialog
- Paper: https://aclanthology.org/I17-1099/

## Attribution

Please cite the original paper:

Li, Yanran, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. “DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset.” Proceedings of the 8th International Joint Conference on Natural Language Processing (IJCNLP 2017), 2017.

## Licence

The DailyDialog dataset is labelled **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**:

https://creativecommons.org/licenses/by-nc-sa/4.0/

In practical terms, the dataset must be attributed, it must not be used commercially under this licence, and adaptations that are distributed must follow the same licence. This notice is a summary, not legal advice. Review the full licence before redistributing any dataset material or adapted dataset.

## How Sherlock AI uses it

- The training split supplies examples for learning.
- The validation split is kept separate and is used to calculate the displayed next-character prediction score.
- The test split is not fetched or used by the page.
- The browser-readable mirror stores turns with speaker labels. Sherlock uses human and assistant turns and ignores system prompts.
- Text is cleaned and converted into a character stream before being copied into the local C++ WebAssembly engine.
- The browser requests dataset rows from the Hugging Face Dataset Viewer API in pages of up to 100 rows. The model's training calculations run on the user's device; the dataset API is only used to deliver the text.

The current model is a small character trigram counter. Training on this dataset gives it more realistic character patterns, but does not make it a general-purpose conversational model.
