/* Sherlock GPT neural worker: a small character-level MLP trained with backpropagation. */
"use strict";

const VOCAB = 128;
const CONTEXT = 4;
const EMBED = 12;
const INPUT = CONTEXT * EMBED;
const HIDDEN = 48;
const EMBED_OFFSET = 0;
const W1_OFFSET = VOCAB * EMBED;
const B1_OFFSET = W1_OFFSET + HIDDEN * INPUT;
const W2_OFFSET = B1_OFFSET + HIDDEN;
const B2_OFFSET = W2_OFFSET + VOCAB * HIDDEN;
const PARAMS = B2_OFFSET + VOCAB;

const BATCH_SIZE = 24;
const LEARNING_RATE = 0.003;
const EVAL_EVERY = 100;
const EVAL_SAMPLES = 256;
const BATCHES_PER_SLICE = 6;

let trainData = null;
let validationData = null;
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
let movingLoss = 0;
let lossInitialized = false;

const inputVector = new Float32Array(INPUT);
const hiddenVector = new Float32Array(HIDDEN);
const probabilities = new Float32Array(VOCAB);
const outputDelta = new Float32Array(VOCAB);
const hiddenDelta = new Float32Array(HIDDEN);
const contextCodes = new Uint8Array(CONTEXT);

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
function byteAt(data, index) {
  const value = data[index] || 0;
  return value < VOCAB ? value : 63;
}
function initializeParameters() {
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
function forward(data, position) {
  for (let slot = 0; slot < CONTEXT; slot++) {
    const code = byteAt(data, position - CONTEXT + slot);
    contextCodes[slot] = code;
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
    let sum = parameters[B2_OFFSET + c];
    const row = W2_OFFSET + c * HIDDEN;
    for (let k = 0; k < HIDDEN; k++)
      sum += parameters[row + k] * hiddenVector[k];
    probabilities[c] = sum;
    if (sum > maximum) maximum = sum;
  }
  let denominator = 0;
  for (let c = 0; c < VOCAB; c++) {
    const value = Math.exp(probabilities[c] - maximum);
    probabilities[c] = value;
    denominator += value;
  }
  const inverse = 1 / Math.max(denominator, 1e-20);
  for (let c = 0; c < VOCAB; c++) probabilities[c] *= inverse;
  return probabilities;
}
function trainExample(position) {
  const target = byteAt(trainData, position);
  const probs = forward(trainData, position);
  const targetProbability = Math.max(probs[target], 1e-9);
  const loss = -Math.log(targetProbability);

  hiddenDelta.fill(0);
  for (let c = 0; c < VOCAB; c++) {
    const delta = probs[c] - (c === target ? 1 : 0);
    outputDelta[c] = delta;
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
function applyAdam(batchCount) {
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
    parameters[i] -= LEARNING_RATE * mHat / (Math.sqrt(vHat) + 1e-8);
    gradients[i] = 0;
  }
}
function predict(data, position) {
  forward(data, position);
  let best = 0;
  let bestValue = probabilities[0];
  for (let c = 1; c < VOCAB; c++) {
    if (probabilities[c] > bestValue) {
      best = c;
      bestValue = probabilities[c];
    }
  }
  return { character: best, loss: -Math.log(Math.max(bestValue, 1e-9)) };
}
function evaluate() {
  if (!validationData || validationData.length <= CONTEXT + 2) return null;
  const available = validationData.length - CONTEXT - 1;
  const count = Math.min(EVAL_SAMPLES, available);
  let correct = 0;
  let totalLoss = 0;
  let total = 0;
  for (let i = 0; i < count; i++) {
    // Keep the same 256-position validation layout used by v0.3.0 for fair comparison.
    const stride = Math.max(1, Math.floor(available / count));
    const position = CONTEXT + i * stride;
    if (position >= validationData.length) break;
    const result = predict(validationData, position);
    const target = byteAt(validationData, position);
    if (result.character === target) correct++;
    const targetProbability = probabilities[target];
    totalLoss += -Math.log(Math.max(targetProbability, 1e-9));
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

  // Preserve the previous accuracy-first checkpoint behaviour. Validation loss breaks ties.
  const improved = score > bestAccuracy + 1e-7 ||
    (Math.abs(score - bestAccuracy) <= 1e-7 && validationLoss < bestValidationLoss - 1e-7);
  if (improved) {
    bestValidationLoss = validationLoss;
    bestAccuracy = score;
    bestParameters = new Float32Array(parameters);
    const checkpointBuffer = bestParameters.slice().buffer;
    self.postMessage({
      type: "checkpoint",
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
      modelBuffer: checkpointBuffer
    }, [checkpointBuffer]);
  }
  return {accuracy: score, validationLoss, perplexity, samples: total, evaluations: evaluationCount};
}
function progressPayload() {
  const now = performance.now();
  let speed = 0;
  if (reportAt > 0 && now > reportAt) {
    speed = (examplesSeen - reportExamples) * 1000 / (now - reportAt);
  }
  reportAt = now;
  reportExamples = examplesSeen;
  return {
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
    examplesPerSecond: speed,
    parameters: PARAMS,
    batchSize: BATCH_SIZE
  };
}
function runBatch() {
  if (!running) return;
  try {
    const maximumPosition = trainData.length - 1;
    // Process several mini-batches per worker slice to reduce timer overhead.
    // Yield after each slice so the page can still send Stop and other messages.
    for (let slice = 0; slice < BATCHES_PER_SLICE && running; slice++) {
      let batchLoss = 0;
      for (let i = 0; i < BATCH_SIZE; i++) {
        const position = CONTEXT + randomInt(maximumPosition - CONTEXT + 1);
        batchLoss += trainExample(position);
      }
      applyAdam(BATCH_SIZE);
      steps++;
      examplesSeen += BATCH_SIZE;
      const averageLoss = batchLoss / BATCH_SIZE;
      movingLoss = lossInitialized ? movingLoss * 0.92 + averageLoss * 0.08 : averageLoss;
      lossInitialized = true;

      if (steps % EVAL_EVERY === 0) evaluate();
      if (steps % 8 === 0) self.postMessage({type: "progress", ...progressPayload()});
    }
    if (running) setTimeout(runBatch, 0);
  } catch (error) {
    running = false;
    self.postMessage({type: "error", message: error && error.message ? error.message : String(error)});
  }
}
function stopTraining() {
  if (!running) return;
  running = false;
  evaluate();
  if (bestParameters) {
    parameters.set(bestParameters);
    // Report metrics for the exact weights that will remain active after Stop.
    evaluate();
  }
  self.postMessage({type: "stopped", ...progressPayload(), restoredBest: Boolean(bestParameters)});
}
function generate(prompt, requestedLength) {
  if (!initialized || examplesSeen === 0) return "Train the neural network first.";
  const maxLength = Math.max(1, Math.min(160, Number(requestedLength) || 100));
  const output = Array.from(String(prompt || "").slice(0, 180));
  const history = output.map(character => {
    const code = character.charCodeAt(0);
    return code < VOCAB ? code : 63;
  });
  for (let i = 0; i < maxLength; i++) {
    const position = history.length;
    for (let slot = 0; slot < CONTEXT; slot++) {
      const sourceIndex = position - CONTEXT + slot;
      contextCodes[slot] = sourceIndex >= 0 ? history[sourceIndex] : 32;
      const source = EMBED_OFFSET + contextCodes[slot] * EMBED;
      for (let j = 0; j < EMBED; j++)
        inputVector[slot * EMBED + j] = parameters[source + j];
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
      if ((c < 32 && c !== 10 && c !== 13) || c === 127) {
        probabilities[c] = -100;
        continue;
      }
      let sum = parameters[B2_OFFSET + c];
      const row = W2_OFFSET + c * HIDDEN;
      for (let k = 0; k < HIDDEN; k++)
        sum += parameters[row + k] * hiddenVector[k];
      probabilities[c] = sum;
      if (sum > maximum) maximum = sum;
    }
    let total = 0;
    for (let c = 0; c < VOCAB; c++) {
      const probability = Math.exp((probabilities[c] - maximum) / 0.85);
      probabilities[c] = probability;
      total += probability;
    }
    let pick = randomUnit() * total;
    let chosen = 32;
    for (let c = 0; c < VOCAB; c++) {
      pick -= probabilities[c];
      if (pick <= 0) {
        chosen = c;
        break;
      }
    }
    history.push(chosen);
    output.push(String.fromCharCode(chosen));
  }
  return output.join("");
}

function setDatasetBuffers(trainBuffer, validationBuffer) {
  trainData = new Uint8Array(trainBuffer);
  validationData = new Uint8Array(validationBuffer);
  if (trainData.length <= CONTEXT + 10 || validationData.length <= CONTEXT + 10)
    throw new Error("The training or validation text is too short.");
}

self.addEventListener("message", event => {
  const message = event.data || {};
  try {
    if (message.type === "init") {
      setDatasetBuffers(message.trainBuffer, message.validationBuffer);
      randomState = 0x5A17C0DE;
      initializeParameters();
      initialized = true;
      self.postMessage({type: "ready", parameters: PARAMS});
    } else if (message.type === "load-model") {
      const restored = new Float32Array(message.modelBuffer);
      if (restored.length !== PARAMS) throw new Error("The saved neural model has the wrong number of weights.");
      parameters = new Float32Array(restored);
      bestParameters = new Float32Array(restored);
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
      optimizerStep = 0;
      firstMoment.fill(0);
      secondMoment.fill(0);
      movingLoss = Math.max(0, Number(message.loss) || 0);
      lossInitialized = movingLoss > 0;
      initialized = true;
      self.postMessage({
        type: "model-loaded",
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
        loss: movingLoss
      });
    } else if (message.type === "set-data") {
      setDatasetBuffers(message.trainBuffer, message.validationBuffer);
      self.postMessage({type: "data-ready", trainBytes: trainData.length, validationBytes: validationData.length});
    } else if (message.type === "start") {
      if (!initialized) throw new Error("Load or restore the neural model first.");
      if (!trainData || !validationData) throw new Error("Load the DailyDialog dataset before training.");
      if (!running) {
        running = true;
        reportAt = performance.now();
        reportExamples = examplesSeen;
        self.postMessage({type: "started"});
        setTimeout(runBatch, 0);
      }
    } else if (message.type === "stop") {
      stopTraining();
    } else if (message.type === "generate") {
      self.postMessage({type: "generated", id: message.id, text: generate(message.prompt, message.length)});
    }
  } catch (error) {
    running = false;
    self.postMessage({type: "error", id: message.id, message: error && error.message ? error.message : String(error)});
  }
});
