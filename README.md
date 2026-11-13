# Sherlock AI

Sherlock AI is a small, browser-based character-language experiment. The C++ model is compiled to WebAssembly and runs locally in the browser.

## Try it

Open the GitHub Pages site:

https://sherlockdapoo.github.io/Sherlock-AI/

On the Training tab, press **Start training**. The page downloads DailyDialog training and validation examples in small pages, prepares the text, and copies it into the WebAssembly engine. The model training itself runs locally in the browser. Stop training when you want to unlock Chat.

The first data load needs an internet connection. The prepared model data stays in page memory and is not saved as a permanent trained model. Refreshing the page resets the training progress and may require downloading the examples again.

## Training data

- Dataset: [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)
- Browser-readable mirror: [agentlans/li2017dailydialog](https://huggingface.co/datasets/agentlans/li2017dailydialog)
- Train and validation splits are used separately.
- The test split is not downloaded or used.
- Only human and assistant turns are included. Dataset system prompts are ignored.
- Attribution and licensing details are in [DATASET_NOTICE.md](DATASET_NOTICE.md).

**Licence restriction:** DailyDialog is licensed under CC BY-NC-SA 4.0. It is for non-commercial use. If you redistribute adapted dataset material, the applicable share-alike terms also matter. Do not use the dataset or this trained-data setup for a commercial product without separate permission.

## What the score means

The Accuracy bar measures held-out next-character prediction accuracy on a separate validation split. It is not a measure of intelligence, factual knowledge, or conversational quality.

- **Current accuracy** is the latest validation score, checked after each complete pass through the training text.
- **Best accuracy this run** is the highest completed-pass score reached since the current run started.
- **Current training pass** shows how far the model has progressed through the training text.
- Accuracy is displayed to three decimal places. The validation calculation samples up to 10,000 positions to keep checking responsive on iPad.

The current score can still rise or fall between completed passes. This small count-based model changes its most-common next-character predictions as it sees more examples, so more training does not guarantee a higher validation score. The best score is shown separately, and when training is stopped the engine restores the model checkpoint that achieved the best held-out score in that run.

The current model is a small character-level 4-gram counter with backoff. It first uses the previous three characters when it has enough examples, then falls back to the previous two characters. Its counters are 32-bit, so an individual next-character count no longer stops at 65,535. It still learns character patterns rather than meaning, and it is not a modern language model.

## Technology

- Front end: HTML, CSS, and JavaScript.
- Learning engine: C++ compiled to WebAssembly.
- Model: sparse three-character context table with two-character backoff and 32-bit counters.
- Best checkpoint: saves and restores the best validation-scoring model within the current page session.
- Training location: user's browser and device.
- Dataset delivery: Hugging Face Dataset Viewer API.

The old Python/Render backend files may remain in the repository for historical experiments, but the GitHub Pages training path uses the local WebAssembly engine.
