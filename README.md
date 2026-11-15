# Sherlock GPT

Sherlock GPT is a small browser-based neural language experiment. The workspace has two tabs: **Chat** and **Neural Lab**. Training runs locally in a dedicated browser worker, and the latest word-model checkpoint is saved in this browser.

## Version 0.9.2: validation-loss history graph

Sherlock GPT now combines four bounded public datasets to give its compact word-level model more varied text:

- **DailyDialog:** up to 3,500 training conversations and 500 held-out validation conversations.
- **UltraChat 200k:** up to 1,600 training conversations from `train_sft` and 250 held-out conversations from `test_sft`.
- **No Robots:** up to 3,500 training conversations from `train` and 500 held-out conversations from `test`.
- **GSM8K:** up to 1,000 maths word problems and worked answers from the `main/train` split, with 250 separate problems from `main/test` held out for evaluation.

The loader samples across each available split rather than taking only its first rows. It formats GSM8K question/answer pairs as user/Sherlock turns, while leaving each source's evaluation examples outside the training stream. Optional sources can fail independently; the interface reports which sources loaded.

The combined training text can now use up to **9 MiB**, matching the existing local engine buffer, and held-out text can use up to **128 KiB**. Downloads are cached in this browser. The new dataset revision causes Sherlock to start a fresh model for the new data mix; the previous compatible checkpoint is archived in this browser before it is replaced.

Because the validation mix has changed, scores from this revision should not be compared directly with scores from earlier dataset versions. These metrics still measure next-token prediction, not mathematical reasoning or general intelligence.

## Open Sherlock GPT

https://sherlockdapoo.github.io/Sherlock-AI/

Open **Neural Lab** and press **Start neural training**. The first run downloads the bounded training and evaluation samples. The neural network itself trains locally in a background browser worker. Press **Stop** to save the current word-level checkpoint and unlock Chat.

The C++ character-counter training screen and its separate Live View have been removed from the visible workspace. The earlier counter remains only as an archived baseline in the project code; the active Chat path uses the neural model.

## Neural network experiment

- **Architecture:** compact word-level multilayer perceptron (MLP), eight-token context, 8-dimensional learned embeddings, one 48-neuron ReLU hidden layer, and a 1,024-way softmax output.
- **Size:** 61,488 trainable parameters.
- **Learning task:** predict the next word-or-punctuation token from the previous eight tokens. Common words and punctuation receive vocabulary IDs; rare or unseen tokens map to an unknown-token ID.
- **Training:** cross-entropy loss and backpropagation, using mini-batches of 32 tokens and AdamW (default learning rate 0.001; decoupled weight decay 0.0001). Two mini-batches are processed per worker slice before yielding to the browser event loop.
- **Validation/evaluation:** every 50 optimizer steps, the worker evaluates up to 320 positions from the combined held-out token stream. It reports top-1 accuracy, mean cross-entropy (negative log-likelihood), and perplexity (`exp(cross-entropy)`). Accuracy is the share of next tokens predicted exactly; lower cross-entropy and lower perplexity indicate better probability estimates. These are next-token metrics, not measures of intelligence, factual knowledge, or conversation quality.
- **Network visualiser:** the Neural Lab draws a live sampled view of token-embedding inputs, 16 hidden neurons, and 12 output-token neurons. The worker sends real activation values and sampled connection weights during progress updates. This is a readable sample of the 61,488-parameter network, not every neuron or connection.
- **Checkpoint:** after each validation check, the page stores model weights, AdamW optimiser memory, training counters, model version, and both graph histories in this browser.
- **Validation-loss history:** Neural Lab now plots held-out cross-entropy at each completed validation check, alongside the existing training-loss and validation-accuracy graphs. The new history is saved in the browser checkpoint and continues accumulating after refresh. The previous character-level checkpoint remains under its old browser storage key and is not overwritten by the word-level upgrade.
- **Stop behaviour:** stopping performs a final evaluation and saves the current word-level model state for Chat and future training.

## Training data and licences

- [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog), accessed through the [browser-readable mirror](https://huggingface.co/datasets/agentlans/li2017dailydialog). The dataset is labelled **CC BY-NC-SA 4.0**.
- [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). Its dataset card lists **MIT**.
- [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots). Its dataset card lists **CC BY-NC 4.0**.
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k). Its dataset card lists **MIT**.

Only the selected training split is included in the training text. Each held-out validation/test sample is assembled separately and is not passed to the training buffer. Text is cleaned, converted to a simple user/Sherlock conversation format, and split into word, punctuation, space, and newline tokens. The tokenizer keeps the 1,021 most frequent training tokens plus three reserved IDs; rare or unseen tokens become `<unk>`. The active model predicts the next token from the preceding eight tokens; this is not full instruction fine-tuning.

Review each dataset's card and licence before redistributing data or model outputs. In particular, DailyDialog and No Robots have non-commercial terms. This README is a practical summary, not legal advice.

## Technology

- Front end: HTML, CSS, and JavaScript.
- Neural training: JavaScript in a dedicated Web Worker.
- Optimiser: AdamW with backpropagation (learning rate 0.001 by default).
- Model storage: browser local storage for the current full training checkpoint and graph history.
- Training location: the user's browser and device.
- Dataset delivery: Hugging Face Dataset Viewer API.
- Static hosting: GitHub Pages.

Legacy C++/WebAssembly baseline files and old Python/Render experiments may remain in the repository, but the active GitHub Pages path uses the neural worker.
