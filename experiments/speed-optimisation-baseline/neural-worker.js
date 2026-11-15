/* Sherlock GPT neural worker: a compact word-token MLP trained with backpropagation. */
"use strict";

const VOCAB = 1024;
const CONTEXT = 8;
const EMBED = 8;
const INPUT = CONTEXT * EMBED;
const HIDDEN = 48;
const UNK_ID = 0;
const SPACE_ID = 1;
const NEWLINE_ID = 2;
const EMBED_OFFSET = 0;
const W1_OFFSET = VOCAB * EMBED;
const B1_OFFSET = W1_OFFSET + HIDDEN * INPUT;
const W2_OFFSET = B1_OFFSET + HIDDEN;
const B2_OFFSET = W2_OFFSET + VOCAB * HIDDEN;
const PARAMS = B2_OFFSET + VOCAB;

const DEFAULT_BATCH_SIZE = 32;
const PERMANENT_MAX_STEPS = 0;
let activeBatchSize = DEFAULT_BATCH_SIZE;
let activeMaxSteps = 0;
let runStartSteps = 0;
const DEFAULT_LEARNING_RATE = 0.001;
let activeLearningRate = DEFAULT_LEARNING_RATE;
const ADAMW_WEIGHT_DECAY = 0.0001;
let activeOptimizer = "adamw";
const EVAL_EVERY = 50;
const EVAL_SAMPLES = 320;
const BATCHES_PER_SLICE = 8;

let trainData = null;
let validationData = null;
let vocabulary = new Array(VOCAB).fill(null);
let tokenToId = new Map();
let vocabularySize = 0;
let trainingUnknownRate = 0;
let validationUnknownRate = 0;
let mathUnknownRate = 0;
let parameters = new Float32Array(PARAMS);
let gradients = new Float32Array(PARAMS);
let firstMoment = new Float32Array(PARAMS);
let secondMoment = new Float32Array(PARAMS);
let bestParameters = null;
let bestAccuracy = -1;
let latestAccuracy = null;
let latestLoss = 0;
let latestValidationLoss = null;
let latestPerplexity = null;
let bestValidationLoss = Infinity;
let evaluationCount = 0;
let validationSamples = 0;
let running = false;
let initialized = false;
let steps = 0;
let examplesSeen = 0;
let optimizerStep = 0;
let randomState = 0x5A17C0DE;
let reportAt = 0;
let reportExamples = 0;
let lastReportedSpeed = 0;
let movingLoss = 0;
let lossInitialized = false;

const inputVector = new Float32Array(INPUT);
const hiddenVector = new Float32Array(HIDDEN);
const probabilities = new Float32Array(VOCAB);
const outputDelta = new Float32Array(VOCAB);
const hiddenDelta = new Float32Array(HIDDEN);
const contextCodes = new Uint16Array(CONTEXT);

function randomUnit() {
  let x = randomState >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  randomState = x >>> 0;
  return randomState / 4294967296;
}
function randomInt(max) {
  return Math.floor(randomUnit() * max);
}
function tokenAt(data, index) {
  return index >= 0 && index < data.length ? data[index] : SPACE_ID;
}
function normaliseToken(token) {
  if (token[0] === "\n") return "\n\n";
  if (token[0] === " " || token[0] === "\t") return " ";
  return token;
}
function walkTokens(text, callback) {
  const pattern = /[a-z0-9]+(?:'[a-z0-9]+)?|[ \t]+|\n+|[^a-z0-9\s]/g;
  let match;
  while ((match = pattern.exec(text)) !== null) callback(normaliseToken(match[0]));
}
function buildTokenizer(trainingText) {
  const counts = new Map();
  walkTokens(trainingText, token => counts.set(token, (counts.get(token) || 0) + 1));
  const ranked = Array.from(counts.entries())
    .filter(([token]) => token !== " " && token !== "\n\n")
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .slice(0, VOCAB - 3);

  vocabulary = new Array(VOCAB).fill(null);
  vocabulary[UNK_ID] = "<unk>";
  vocabulary[SPACE_ID] = " ";
  vocabulary[NEWLINE_ID] = "\n\n";
  tokenToId = new Map([
    ["<unk>", UNK_ID],
    [" ", SPACE_ID],
    ["\n\n", NEWLINE_ID]
  ]);
  vocabularySize = 3;
  for (const [token] of ranked) {
    if (tokenToId.has(token) || vocabularySize >= VOCAB) continue;
    tokenToId.set(token, vocabularySize);
    vocabulary[vocabularySize] = token;
    vocabularySize++;
  }
}
function encodeText(text) {
  const ids = [];
  let unknown = 0;
  let total = 0;
  walkTokens(text, token => {
    const id = tokenToId.get(token);
    if (id === undefined) {
      ids.push(UNK_ID);
      unknown++;
    } else {
      ids.push(id);
    }
    total++;
  });
  return {
    data: Uint16Array.from(ids),
    unknownRate: total ? unknown * 100 / total : 0,
    totalTokens: total
  };
}
function initialiseParameters() {
  parameters.fill(0);
  gradients.fill(0);
  firstMoment.fill(0);
  secondMoment.fill(0);
  for (let i = EMBED_OFFSET; i < W1_OFFSET; i++)
    parameters[i] = (randomUnit() * 2 - 1) * 0.08;
  const w1Scale = Math.sqrt(2 / INPUT);
  for (let i = W1_OFFSET; i < B1_OFFSET; i++)
    parameters[i] = (randomUnit() * 2 - 1) * w1Scale;
  const w2Scale = Math.sqrt(2 / HIDDEN);
  for (let i = W2_OFFSET; i < B2_OFFSET; i++)
    parameters[i] = (randomUnit() * 2 - 1) * w2Scale;
  bestParameters = null;
  bestAccuracy = -1;
  latestAccuracy = null;
  latestLoss = 0;
  latestValidationLoss = null;
  latestPerplexity = null;
  bestValidationLoss = Infinity;
  evaluationCount = 0;
  validationSamples = 0;
  steps = 0;
  examplesSeen = 0;
  optimizerStep = 0;
  movingLoss = 0;
  lossInitialized = false;
}
function forwardContext() {
  for (let slot = 0; slot < CONTEXT; slot++) {
    const code = contextCodes[slot] < vocabularySize ? contextCodes[slot] : UNK_ID;
    const source = EMBED_OFFSET + code * EMBED;
    const destination = slot * EMBED;
    for (let j = 0; j < EMBED; j++)
      inputVector[destination + j] = parameters[source + j];
  }

  for (let k = 0; k < HIDDEN; k++) {
    let sum = parameters[B1_OFFSET + k];
    const row = W1_OFFSET + k * INPUT;
    for (let j = 0; j < INPUT; j++)
      sum += parameters[row + j] * inputVector[j];
    hiddenVector[k] = sum > 0 ? sum : 0;
  }

  let maximum = -Infinity;
  for (let c = 0; c < VOCAB; c++) {
    if (c >= vocabularySize) {
      probabilities[c] = -Infinity;
      continue;
    }
    let sum = parameters[B2_OFFSET + c];
    const row = W2_OFFSET + c * HIDDEN;
    for (let k = 0; k < HIDDEN; k++)
      sum += parameters[row + k] * hiddenVector[k];
    probabilities[c] = sum;
    if (sum > maximum) maximum = sum;
  }
  let denominator = 0;
  for (let c = 0; c < VOCAB; c++) {
    if (c >= vocabularySize) {
      probabilities[c] = 0;
      continue;
    }
    const value = Math.exp(probabilities[c] - maximum);
    probabilities[c] = value;
    denominator += value;
  }
  const inverse = 1 / Math.max(denominator, 1e-20);
  for (let c = 0; c < vocabularySize; c++) probabilities[c] *= inverse;
  return probabilities;
}
function forward(data, position) {
  for (let slot = 0; slot < CONTEXT; slot++)
    contextCodes[slot] = tokenAt(data, position - CONTEXT + slot);
  return forwardContext();
}
function trainExample(position) {
  const target = tokenAt(trainData, position);
  const probs = forward(trainData, position);
  const targetProbability = Math.max(probs[target], 1e-9);
  const loss = -Math.log(targetProbability);

  hiddenDelta.fill(0);
  for (let c = 0; c < VOCAB; c++) {
    const delta = c < vocabularySize
      ? probs[c] - (c === target ? 1 : 0)
      : 0;
    outputDelta[c] = delta;
    if (delta === 0) continue;
    gradients[B2_OFFSET + c] += delta;
    const row = W2_OFFSET + c * HIDDEN;
    for (let k = 0; k < HIDDEN; k++) {
      gradients[row + k] += delta * hiddenVector[k];
      hiddenDelta[k] += delta * parameters[row + k];
    }
  }

  for (let k = 0; k < HIDDEN; k++) {
    const delta = hiddenVector[k] > 0 ? hiddenDelta[k] : 0;
    hiddenDelta[k] = delta;
    gradients[B1_OFFSET + k] += delta;
    const row = W1_OFFSET + k * INPUT;
    for (let j = 0; j < INPUT; j++)
      gradients[row + j] += delta * inputVector[j];
  }

  for (let slot = 0; slot < CONTEXT; slot++) {
    const code = contextCodes[slot];
    const embeddingRow = EMBED_OFFSET + code * EMBED;
    const inputStart = slot * EMBED;
    for (let j = 0; j < EMBED; j++) {
      let contribution = 0;
      const inputIndex = inputStart + j;
      for (let k = 0; k < HIDDEN; k++)
        contribution += hiddenDelta[k] * parameters[W1_OFFSET + k * INPUT + inputIndex];
      gradients[embeddingRow + j] += contribution;
    }
  }
  return loss;
}
function applyOptimizer(batchCount) {
  let squaredNorm = 0;
  const inverseBatch = 1 / batchCount;
  for (let i = 0; i < PARAMS; i++) {
    gradients[i] *= inverseBatch;
    squaredNorm += gradients[i] * gradients[i];
  }
  const norm = Math.sqrt(squaredNorm);
  const clip = norm > 5 ? 5 / norm : 1;
  optimizerStep++;
  const correction1 = 1 - Math.pow(0.9, optimizerStep);
  const correction2 = 1 - Math.pow(0.999, optimizerStep);

  for (let i = 0; i < PARAMS; i++) {
    const gradient = gradients[i] * clip;
    const m = 0.9 * firstMoment[i] + 0.1 * gradient;
    const v = 0.999 * secondMoment[i] + 0.001 * gradient * gradient;
    firstMoment[i] = m;
    secondMoment[i] = v;
    const mHat = m / correction1;
    const vHat = v / correction2;
    const adaptiveUpdate = activeLearningRate * mHat / (Math.sqrt(vHat) + 1e-8);
    const decayParameter = (i >= EMBED_OFFSET && i < W1_OFFSET) ||
      (i >= W1_OFFSET && i < B1_OFFSET) ||
      (i >= W2_OFFSET && i < B2_OFFSET);
    const weightDecay = activeOptimizer === "adamw" && decayParameter
      ? activeLearningRate * ADAMW_WEIGHT_DECAY * parameters[i] : 0;
    parameters[i] -= adaptiveUpdate + weightDecay;
    gradients[i] = 0;
  }
}
function predict(data, position) {
  forward(data, position);
  let best = 0;
  let bestValue = probabilities[0];
  for (let c = 1; c < vocabularySize; c++) {
    if (probabilities[c] > bestValue) {
      best = c;
      bestValue = probabilities[c];
    }
  }
  return {token: best, loss: -Math.log(Math.max(bestValue, 1e-9))};
}
function evaluate() {
  if (!validationData || validationData.length <= CONTEXT + 2) return null;
  const available = validationData.length - CONTEXT - 1;
  const count = Math.min(EVAL_SAMPLES, available);
  let correct = 0;
  let totalLoss = 0;
  let total = 0;
  const stride = Math.max(1, Math.floor(available / count));
  for (let i = 0; i < count; i++) {
    const position = CONTEXT + i * stride;
    if (position >= validationData.length) break;
    const result = predict(validationData, position);
    const target = tokenAt(validationData, position);
    if (result.token === target) correct++;
    totalLoss += -Math.log(Math.max(probabilities[target], 1e-9));
    total++;
  }
  if (!total) return null;

  const score = correct * 100 / total;
  const validationLoss = totalLoss / total;
  const perplexity = Math.exp(validationLoss);
  latestAccuracy = score;
  latestValidationLoss = validationLoss;
  latestPerplexity = perplexity;
  evaluationCount++;
  validationSamples = total;

  bestValidationLoss = validationLoss;
  bestAccuracy = score;
  bestParameters = new Float32Array(parameters);
  const checkpointState = new Float32Array(PARAMS * 3);
  checkpointState.set(parameters, 0);
  checkpointState.set(firstMoment, PARAMS);
  checkpointState.set(secondMoment, PARAMS * 2);
  const checkpointBuffer = checkpointState.buffer;
  self.postMessage({
    type: "checkpoint",
    optimizer: activeOptimizer,
    learningRate: activeLearningRate,
    accuracy: score,
    bestAccuracy,
    validationLoss,
    bestValidationLoss,
    perplexity,
    evaluations: evaluationCount,
    validationSamples: total,
    steps,
    examples: examplesSeen,
    loss: movingLoss,
    batchSize: activeBatchSize,
    maxSteps: activeMaxSteps,
    runSteps: Math.max(0, steps - runStartSteps),
    optimizerStep,
    randomState: randomState >>> 0,
    parameters: PARAMS,
    modelBuffer: checkpointBuffer
  }, [checkpointBuffer]);
  return {accuracy: score, validationLoss, perplexity, samples: total, evaluations: evaluationCount};
}
function evaluateMathText(text) {
  const encoded = encodeText(text);
  const data = encoded.data;
  mathUnknownRate = encoded.unknownRate;
  if (!data || data.length <= CONTEXT + 2) throw new Error("The separate GSM8K held-out text is too short.");
  const available = data.length - CONTEXT - 1;
  const count = Math.min(EVAL_SAMPLES, available);
  const stride = Math.max(1, Math.floor(available / count));
  let correct = 0;
  let totalLoss = 0;
  let total = 0;
  for (let i = 0; i < count; i++) {
    const position = CONTEXT + i * stride;
    if (position >= data.length) break;
    const result = predict(data, position);
    const target = tokenAt(data, position);
    if (result.token === target) correct++;
    totalLoss += -Math.log(Math.max(probabilities[target], 1e-9));
    total++;
  }
  if (!total) throw new Error("No GSM8K tokens were available for evaluation.");
  const loss = totalLoss / total;
  return {
    accuracy: correct * 100 / total,
    loss,
    perplexity: Math.exp(loss),
    samples: total,
    tokens: data.length,
    unknownRate: mathUnknownRate,
    vocabularySize
  };
}
function networkSnapshot() {
  const inputIds = Array.from({length: 12}, (_, i) => Math.floor(i * (INPUT - 1) / 11));
  const hiddenIds = Array.from({length: 16}, (_, i) => Math.floor(i * (HIDDEN - 1) / 15));
  const outputIds = Array.from({length: 12}, (_, i) => Math.floor(i * (vocabularySize - 1) / 11));
  const input = inputIds.map(i => inputVector[i]);
  const hidden = hiddenIds.map(i => hiddenVector[i]);
  const output = outputIds.map(i => probabilities[i]);
  const w1 = [], w2 = [];
  for (const h of hiddenIds) for (const i of inputIds) w1.push(parameters[W1_OFFSET + h * INPUT + i]);
  for (const o of outputIds) for (const h of hiddenIds) w2.push(parameters[W2_OFFSET + o * HIDDEN + h]);
  let top = 0;
  for (let i = 1; i < vocabularySize; i++) if (probabilities[i] > probabilities[top]) top = i;
  const token = vocabulary[top] || "<unk>";
  const label = token.replace(/\n/g, "↵").slice(0, 14);
  return {input, hidden, output, w1, w2, topCharacter: JSON.stringify(label) + " · " + (probabilities[top] * 100).toFixed(2) + "%"};
}
function progressPayload() {
  const now = performance.now();
  if (reportAt === 0) {
    reportAt = now;
    reportExamples = examplesSeen;
  } else if (now - reportAt >= 500) {
    const elapsed = now - reportAt;
    const measured = Math.max(0, (examplesSeen - reportExamples) * 1000 / elapsed);
    lastReportedSpeed = lastReportedSpeed > 0
      ? lastReportedSpeed * 0.65 + measured * 0.35
      : measured;
    reportAt = now;
    reportExamples = examplesSeen;
  }
  return {
    network: networkSnapshot(),
    steps,
    examples: examplesSeen,
    epochs: trainData ? examplesSeen / Math.max(1, trainData.length - CONTEXT) : 0,
    accuracy: latestAccuracy,
    bestAccuracy,
    loss: movingLoss,
    validationLoss: latestValidationLoss,
    bestValidationLoss: Number.isFinite(bestValidationLoss) ? bestValidationLoss : null,
    perplexity: latestPerplexity,
    evaluations: evaluationCount,
    validationSamples,
    examplesPerSecond: lastReportedSpeed,
    parameters: PARAMS,
    batchSize: activeBatchSize,
    maxSteps: activeMaxSteps,
    runSteps: Math.max(0, steps - runStartSteps),
    optimizer: activeOptimizer,
    learningRate: activeLearningRate,
    vocabularySize,
    trainingUnknownRate,
    validationUnknownRate,
    trainTokens: trainData ? trainData.length : 0,
    validationTokens: validationData ? validationData.length : 0
  };
}
function runBatch() {
  if (!running) return;
  try {
    const maximumPosition = trainData.length - 1;
    for (let slice = 0; slice < BATCHES_PER_SLICE && running &&
      (activeMaxSteps === 0 || steps - runStartSteps < activeMaxSteps); slice++) {
      let batchLoss = 0;
      for (let i = 0; i < activeBatchSize; i++) {
        const position = CONTEXT + randomInt(maximumPosition - CONTEXT + 1);
        batchLoss += trainExample(position);
      }
      applyOptimizer(activeBatchSize);
      steps++;
      examplesSeen += activeBatchSize;
      const averageLoss = batchLoss / activeBatchSize;
      movingLoss = lossInitialized ? movingLoss * 0.92 + averageLoss * 0.08 : averageLoss;
      lossInitialized = true;

      if (steps % EVAL_EVERY === 0) evaluate();
      if (activeMaxSteps > 0 && steps - runStartSteps >= activeMaxSteps) {
        stopTraining("limit");
        return;
      }
    }
    if (running) self.postMessage({type: "progress", ...progressPayload()});
    if (running) setTimeout(runBatch, 0);
  } catch (error) {
    running = false;
    self.postMessage({type: "error", message: error && error.message ? error.message : String(error)});
  }
}
function stopTraining(reason = "manual") {
  if (!running) return;
  running = false;
  evaluate();
  self.postMessage({type: "stopped", ...progressPayload(), restoredBest: false, stopReason: reason});
}
function generate(prompt, requestedLength) {
  if (!initialized || examplesSeen === 0 || vocabularySize <= 3) return "Train the neural network first.";
  const rawPrompt = String(prompt || "").slice(0, 180);
  const promptEncoded = encodeText(rawPrompt.toLowerCase());
  const history = Array.from(promptEncoded.data);
  const maxLength = Math.max(1, Math.min(160, Number(requestedLength) || 80));
  let output = rawPrompt;

  for (let i = 0; i < maxLength; i++) {
    for (let slot = 0; slot < CONTEXT; slot++) {
      const sourceIndex = history.length - CONTEXT + slot;
      contextCodes[slot] = sourceIndex >= 0 ? history[sourceIndex] : SPACE_ID;
    }
    forwardContext();

    const candidates = [];
    let total = 0;
    const inverseTemperature = 1 / 0.85;
    const previousToken = history.length ? history[history.length - 1] : SPACE_ID;
    for (let c = 1; c < vocabularySize; c++) {
      // Training text normalises whitespace runs to one token. Avoid generating
      // impossible repeated whitespace tokens that create empty-looking output.
      if ((c === SPACE_ID || c === NEWLINE_ID) &&
          (previousToken === SPACE_ID || previousToken === NEWLINE_ID)) continue;
      const weight = Math.pow(Math.max(probabilities[c], 1e-20), inverseTemperature);
      probabilities[c] = weight;
      total += weight;
      candidates.push(c);
    }
    if (!candidates.length) candidates.push(SPACE_ID), total = 1, probabilities[SPACE_ID] = 1;
    candidates.sort((a, b) => probabilities[b] - probabilities[a]);

    let nucleusTotal = 0;
    let nucleusCount = 0;
    const nucleusTarget = total * 0.92;
    for (let j = 0; j < candidates.length; j++) {
      nucleusTotal += probabilities[candidates[j]];
      nucleusCount = j + 1;
      if (nucleusTotal >= nucleusTarget) break;
    }

    let pick = randomUnit() * nucleusTotal;
    let chosen = candidates[nucleusCount - 1] ?? SPACE_ID;
    for (let j = 0; j < nucleusCount; j++) {
      const id = candidates[j];
      pick -= probabilities[id];
      if (pick <= 0) {
        chosen = id;
        break;
      }
    }
    history.push(chosen);
    output += vocabulary[chosen] || "";
  }
  return output;
}

function setDatasetBuffers(trainBuffer, validationBuffer) {
  const decoder = new TextDecoder("utf-8");
  const trainingText = decoder.decode(new Uint8Array(trainBuffer)).toLowerCase();
  const validationText = decoder.decode(new Uint8Array(validationBuffer)).toLowerCase();
  buildTokenizer(trainingText);
  const encodedTraining = encodeText(trainingText);
  const encodedValidation = encodeText(validationText);
  trainData = encodedTraining.data;
  validationData = encodedValidation.data;
  trainingUnknownRate = encodedTraining.unknownRate;
  validationUnknownRate = encodedValidation.unknownRate;
  if (trainData.length <= CONTEXT + 10 || validationData.length <= CONTEXT + 10)
    throw new Error("The tokenized training or validation text is too short.");
  return {
    trainTokens: trainData.length,
    validationTokens: validationData.length,
    vocabularySize,
    trainingUnknownRate,
    validationUnknownRate
  };
}

self.addEventListener("message", event => {
  const message = event.data || {};
  try {
    if (message.type === "init") {
      const dataInfo = setDatasetBuffers(message.trainBuffer, message.validationBuffer);
      randomState = 0x5A17C0DE;
      initialiseParameters();
      initialized = true;
      self.postMessage({type: "ready", parameters: PARAMS, ...dataInfo});
    } else if (message.type === "load-model") {
      activeOptimizer = message.optimizer === "adamw" ? "adamw" : "adam";
      activeLearningRate = [0.001, 0.003, 0.005].includes(Number(message.learningRate))
        ? Number(message.learningRate) : DEFAULT_LEARNING_RATE;
      activeBatchSize = DEFAULT_BATCH_SIZE;
      activeMaxSteps = PERMANENT_MAX_STEPS;
      const restoredState = new Float32Array(message.modelBuffer);
      if (restoredState.length !== PARAMS * 3 || Number(message.parameterCount) !== PARAMS)
        throw new Error("The saved word-token training state does not match this model version.");
      parameters = restoredState.slice(0, PARAMS);
      firstMoment.set(restoredState.subarray(PARAMS, PARAMS * 2));
      secondMoment.set(restoredState.subarray(PARAMS * 2, PARAMS * 3));
      gradients.fill(0);
      bestParameters = new Float32Array(parameters);
      bestAccuracy = Number.isFinite(Number(message.accuracy)) ? Number(message.accuracy) : -1;
      latestAccuracy = bestAccuracy >= 0 ? bestAccuracy : null;
      latestValidationLoss = message.validationLoss !== null && message.validationLoss !== undefined && Number.isFinite(Number(message.validationLoss))
        ? Number(message.validationLoss) : null;
      bestValidationLoss = latestValidationLoss === null ? Infinity : latestValidationLoss;
      latestPerplexity = message.perplexity !== null && message.perplexity !== undefined && Number.isFinite(Number(message.perplexity))
        ? Number(message.perplexity) : null;
      evaluationCount = Math.max(0, Number(message.evaluations) || 0);
      validationSamples = Math.max(0, Number(message.validationSamples) || 0);
      steps = Math.max(0, Number(message.steps) || 0);
      examplesSeen = Math.max(0, Number(message.examples) || 0);
      optimizerStep = Math.max(0, Math.floor(Number(message.optimizerStep) || 0));
      randomState = (Number(message.randomState) >>> 0) || 0x5A17C0DE;
      movingLoss = Math.max(0, Number(message.loss) || 0);
      lossInitialized = movingLoss > 0;
      initialized = true;
      self.postMessage({
        type: "model-loaded",
        optimizer: activeOptimizer,
        parameters: PARAMS,
        accuracy: latestAccuracy,
        bestAccuracy,
        validationLoss: latestValidationLoss,
        bestValidationLoss: Number.isFinite(bestValidationLoss) ? bestValidationLoss : null,
        perplexity: latestPerplexity,
        evaluations: evaluationCount,
        validationSamples,
        steps,
        examples: examplesSeen,
        loss: movingLoss,
        learningRate: activeLearningRate,
        batchSize: activeBatchSize,
        maxSteps: activeMaxSteps,
        fullTrainingState: Number(message.checkpointVersion) >= 9
      });
    } else if (message.type === "set-data") {
      const dataInfo = setDatasetBuffers(message.trainBuffer, message.validationBuffer);
      self.postMessage({type: "data-ready", ...dataInfo});
    } else if (message.type === "reset-for-dataset") {
      if (running) throw new Error("Stop training before resetting the model.");
      const dataInfo = setDatasetBuffers(message.trainBuffer, message.validationBuffer);
      activeOptimizer = message.optimizer === "adam" ? "adam" : "adamw";
      randomState = 0x5A17C0DE;
      initialiseParameters();
      initialized = true;
      self.postMessage({
        type: "reset-ready",
        optimizer: activeOptimizer,
        parameters: PARAMS,
        ...dataInfo
      });
    } else if (message.type === "start") {
      if (!initialized) throw new Error("Load or restore the neural model first.");
      const requestedOptimizer = message.optimizer === "adamw" || (message.optimizer == null && activeOptimizer === "adamw") ? "adamw" : "adam";
      const requestedLearningRate = [0.001, 0.003, 0.005].includes(Number(message.learningRate))
        ? Number(message.learningRate) : DEFAULT_LEARNING_RATE;
      activeLearningRate = requestedLearningRate;
      activeBatchSize = DEFAULT_BATCH_SIZE;
      activeMaxSteps = PERMANENT_MAX_STEPS;
      const optimizerChanged = requestedOptimizer !== activeOptimizer;
      if (optimizerChanged) {
        activeOptimizer = requestedOptimizer;
        randomState = 0x5A17C0DE;
        initialiseParameters();
      }
      if (!trainData || !validationData) throw new Error("Load the training and validation datasets before training.");
      if (!running) {
        runStartSteps = steps;
        running = true;
        reportAt = performance.now();
        reportExamples = examplesSeen;
        lastReportedSpeed = 0;
        self.postMessage({
          type: "started",
          optimizer: activeOptimizer,
          learningRate: activeLearningRate,
          batchSize: activeBatchSize,
          maxSteps: activeMaxSteps,
          runSteps: 0,
          reset: optimizerChanged
        });
        setTimeout(runBatch, 0);
      }
    } else if (message.type === "stop") {
      stopTraining();
    } else if (message.type === "evaluate-math") {
      if (!initialized || vocabularySize <= 3) throw new Error("Load the training data before running the GSM8K benchmark.");
      const decoder = new TextDecoder("utf-8");
      const result = evaluateMathText(decoder.decode(message.mathBuffer).toLowerCase());
      self.postMessage({
        type: "math-evaluation",
        id: message.id,
        ...result,
        problems: Math.max(0, Number(message.problems) || 0),
        steps,
        examples: examplesSeen
      });
    } else if (message.type === "generate") {
      const savedRandomState = randomState;
      try {
        const requestedSeed = Number(message.seed);
        if (Number.isFinite(requestedSeed) && requestedSeed >= 0)
          randomState = (requestedSeed >>> 0) || 0x5A17C0DE;
        self.postMessage({type: "generated", id: message.id, text: generate(message.prompt, message.length)});
      } finally {
        randomState = savedRandomState;
      }
    }
  } catch (error) {
    running = false;
    self.postMessage({type: "error", id: message.id, message: error && error.message ? error.message : String(error)});
  }
});