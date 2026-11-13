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

The current model is a small character trigram counter. It learns which character tends to follow the previous two characters. It does not understand meaning like a modern language model.

## Technology

- Front end: HTML, CSS, and JavaScript.
- Learning engine: C++ compiled to WebAssembly.
- Training location: user's browser and device.
- Dataset delivery: Hugging Face Dataset Viewer API.

The old Python/Render backend files may remain in the repository for historical experiments, but the GitHub Pages training path uses the local WebAssembly engine.
