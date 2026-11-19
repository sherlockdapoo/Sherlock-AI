/* Sherlock GPT v0.18.1: balanced learning with responsive manual stopping. */
"use strict";

const VOCAB = 1024;
const CONTEXT = 32;
const EMBED = 8;
const INPUT = CONTEXT * EMBED;
const HIDDEN = 48;
const UNK_ID = 0;
const SPACE_ID = 1;
const NEWLINE_ID = 2;
const EOS_ID = 3;
const EOS_TOKEN = "\uE000";
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
const EVAL_SAMPLES = 640;
const EVAL_CHUNK_SIZE = 40; // Yield between validation slices so Stop can be handled.
const MIN_VALIDATION_IMPROVEMENT = 0.001;
const BATCHES_PER_SLICE = 1; // Yield to the worker event queue after every batch so Stop and other messages are handled promptly.
const PROGRESS_INTERVAL_MS = 500;
const NETWORK_SNAPSHOT_INTERVAL_MS = 1000;
let lastProgressMessageAt = -Infinity;
let lastNetworkSnapshotAt = -Infinity;
let profilingBatchMs = 0;
let profilingSelectionMs = 0;
let profilingExampleMs = 0;
let profilingOptimizerMs = 0;
let profilingBatches = 0;
let detailedProfiling = false;
let detailedProfileActive = false;
let profileForwardMs = 0;
let profileOutputMs = 0;
let profileBackpropMs = 0;
let profileEmbeddingMs = 0;
let detailedProfileExamples = 0;

let trainData = null;
let validationData = null;
let trainTargetPositions = new Uint32Array(0);
let trainTargetPositionsByFamily = [];
let validationTargetPositions = new Uint32Array(0);
const TASK_FAMILY_NAMES = [
  "arithmetic",
  "percentages",
  "unit conversions",
  "word problems",
  "number sequences",
  "letter/instruction tasks"
];
let vocabulary = new Array(VOCAB).fill(null);
let tokenToId = new Map();
let vocabularySize = 0;
let activeOptimizerRanges = [];
let trainingUnknownRate = 0;
let validationUnknownRate = 0;
let mathUnknownRate = 0;
let parameters = new Float32Array(PARAMS);
let gradients = new Float32Array(PARAMS);
let firstMoment = new Float32Array(PARAMS);
let secondMoment = new Float32Array(PARAMS);
let bestParameters = null;
let bestFirstMoment = new Float32Array(PARAMS);
let bestSecondMoment = new Float32Array(PARAMS);
let bestOptimizerStep = 0;
let bestRandomState = 0x5A17C0DE;
let bestSteps = 0;
let bestExamples = 0;
let bestMovingLoss = 0;
let evaluationInProgress = false;
let stopRequested = false;
let stopRequestReason = "manual";
let bestCheckpointStepsSent = -1;
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
const embeddingInputGradient = new Float32Array(INPUT);
const hiddenVector = new Float32Array(HIDDEN);
const probabilities = new Float32Array(VOCAB);
const hiddenDelta = new Float32Array(HIDDEN);
const contextCodes = new Uint16Array(CONTEXT);
const activeHiddenIndices = new Uint8Array(HIDDEN);
const batchPositions = new Uint32Array(DEFAULT_BATCH_SIZE);
let activeHiddenCount = 0;

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
  const pattern = /[a-z]+(?:'[a-z]+)?|[0-9]|[ \t]+|\n+|[^a-z0-9\s]/g;
  let match;
  let tokenIndex = 0;
  while ((match = pattern.exec(text)) !== null) {
    callback(normaliseToken(match[0]), match.index, tokenIndex);
    tokenIndex++;
  }
}

// Insert a dedicated target at the end of each Sherlock response.
function markAssistantEndTokens(text) {
  const rolePattern = /(?:^|\n)(user|sherlock):[ \t]*/g;
  let output = "";
  let cursor = 0;
  let activeRole = null;
  let match;
  while ((match = rolePattern.exec(text)) !== null) {
    output += text.slice(cursor, match.index);
    if (activeRole === "sherlock") output += EOS_TOKEN;
    output += match[0];
    cursor = rolePattern.lastIndex;
    activeRole = match[1];
  }
  output += text.slice(cursor);
  if (activeRole === "sherlock") output += EOS_TOKEN;
  return output;
}

// Return token positions whose targets fall inside Sherlock responses.
// The preceding context is still taken from the full conversation stream.
function collectAssistantTargetPositions(text) {
  const ranges = [];
  const rolePattern = /(?:^|\n)(user|sherlock):[ \t]*/g;
  let activeRole = null;
  let bodyStart = 0;
  let match;
  while ((match = rolePattern.exec(text)) !== null) {
    if (activeRole === "sherlock" && bodyStart < match.index)
      ranges.push([bodyStart, match.index]);
    activeRole = match[1];
    bodyStart = rolePattern.lastIndex;
  }
  if (activeRole === "sherlock" && bodyStart < text.length)
    ranges.push([bodyStart, text.length]);

  const positions = [];
  let rangeIndex = 0;
  let tokenIndex = 0;
  walkTokens(text, (_token, offset) => {
    while (rangeIndex < ranges.length && offset >= ranges[rangeIndex][1])
      rangeIndex++;
    if (tokenIndex >= CONTEXT && rangeIndex < ranges.length &&
        offset >= ranges[rangeIndex][0] && offset < ranges[rangeIndex][1]) {
      positions.push(tokenIndex);
    }
    tokenIndex++;
  });
  return Uint32Array.from(positions);
}

function taskFamilyIndex(prompt) {
  const value = String(prompt || "").toLowerCase();
  if (/%|\bpercent(?:age)?\b/.test(value)) return 1;
  if (/\bconvert\b|\bare equal to\b|\bchange\b.*\binto\b/.test(value)) return 2;
  if (/\bsequence\b/.test(value)) return 4;
  if (/\balphabet\b|\bletter\b|\bascending order\b|\bdescending order\b/.test(value)) return 5;
  if (/\b(?:box|shop|bags?|groups?|shared|added|sells?|remain|altogether|marbles|stickers|pencils|books|cards)\b/.test(value)) return 3;
  if (/\b(?:plus|minus|multiplied|divide)\b/.test(value)) return 0;
  return -1;
}

function collectAssistantTargetPositionsByFamily(text) {
  const ranges = [];
  const rolePattern = /(?:^|\n)(user|sherlock):[ \t]*/g;
  let activeRole = null;
  let bodyStart = 0;
  let activeFamily = -1;
  let match;
  while ((match = rolePattern.exec(text)) !== null) {
    if (activeRole === "user") {
      activeFamily = taskFamilyIndex(text.slice(bodyStart, match.index));
    } else if (activeRole === "sherlock" && bodyStart < match.index) {
      ranges.push([bodyStart, match.index, activeFamily]);
    }
    activeRole = match[1];
    bodyStart = rolePattern.lastIndex;
  }
  if (activeRole === "sherlock" && bodyStart < text.length)
    ranges.push([bodyStart, text.length, activeFamily]);

  const positions = TASK_FAMILY_NAMES.map(() => []);
  let rangeIndex = 0;
  let tokenIndex = 0;
  walkTokens(text, (_token, offset) => {
    while (rangeIndex < ranges.length && offset >= ranges[rangeIndex][1])
      rangeIndex++;
    const range = ranges[rangeIndex];
    if (tokenIndex >= CONTEXT && range && offset >= range[0] && offset < range[1] &&
        range[2] >= 0 && range[2] < positions.length) {
      positions[range[2]].push(tokenIndex);
    }
    tokenIndex++;
  });
  return positions.map(group => Uint32Array.from(group));
}

function sampleBalancedTrainingPosition() {
  const familyIndex = randomInt(TASK_FAMILY_NAMES.length);
  const familyPositions = trainTargetPositionsByFamily[familyIndex];
  if (!familyPositions || !familyPositions.length)
    throw new Error("Balanced Learning has no training targets for " + TASK_FAMILY_NAMES[familyIndex] + ".");
  return familyPositions[randomInt(familyPositions.length)];
}
function refreshOptimizerRanges() {
  activeOptimizerRanges = [
    [EMBED_OFFSET, vocabularySize * EMBED],
    [W1_OFFSET, B1_OFFSET],
    [B1_OFFSET, W2_OFFSET],
    [W2_OFFSET, W2_OFFSET + vocabularySize * HIDDEN],
    [B2_OFFSET, B2_OFFSET + vocabularySize]
  ];
}
function buildTokenizer(trainingText) {
  const counts = new Map();
  walkTokens(trainingText, token => counts.set(token, (counts.get(token) || 0) + 1));
  const ranked = Array.from(counts.entries())
    .filter(([token]) => token !== " " && token !== "\n\n" && token !== EOS_TOKEN)
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .slice(0, VOCAB - 4);

  vocabulary = new Array(VOCAB).fill(null);
  vocabulary[UNK_ID] = "<unk>";
  vocabulary[SPACE_ID] = " ";
  vocabulary[NEWLINE_ID] = "\n\n";
  vocabulary[EOS_ID] = EOS_TOKEN;
  tokenToId = new Map([
    ["<unk>", UNK_ID],
    [" ", SPACE_ID],
    ["\n\n", NEWLINE_ID],
    [EOS_TOKEN, EOS_ID]
  ]);
  vocabularySize = 4;
  for (const [token] of ranked) {
    if (tokenToId.has(token) || vocabularySize >= VOCAB) continue;
    tokenToId.set(token, vocabularySize);
    vocabulary[vocabularySize] = token;
    vocabularySize++;
  }
  refreshOptimizerRanges();
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

function diagnoseUnknownTokens(trainBuffer, validationBuffer) {
  const decoder = new TextDecoder("utf-8");
  const trainingText = markAssistantEndTokens(decoder.decode(new Uint8Array(trainBuffer)).toLowerCase());
  const validationText = markAssistantEndTokens(decoder.decode(new Uint8Array(validationBuffer)).toLowerCase());
  const counts = new Map();
  walkTokens(trainingText, token => counts.set(token, (counts.get(token) || 0) + 1));
  const ranked = Array.from(counts.entries())
    .filter(([token]) => token !== " " && token !== "\n\n" && token !== EOS_TOKEN)
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .slice(0, VOCAB - 4);
  const knownTokens = new Set(["<unk>", " ", "\n\n", EOS_TOKEN]);
  for (const [token] of ranked) knownTokens.add(token);

  let totalTokens = 0;
  let unknownTokens = 0;
  const missingCounts = new Map();
  walkTokens(validationText, token => {
    totalTokens++;
    if (!knownTokens.has(token)) {
      unknownTokens++;
      missingCounts.set(token, (missingCounts.get(token) || 0) + 1);
    }
  });
  const topUnknownTokens = Array.from(missingCounts.entries())
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .slice(0, 15)
    .map(([token, count]) => ({token, count}));
  return {
    vocabularySize: knownTokens.size,
    totalTokens,
    unknownTokens,
    unknownRate: totalTokens ? unknownTokens * 100 / totalTokens : 0,
    uniqueUnknownTokens: missingCounts.size,
    topUnknownTokens
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
  bestFirstMoment.fill(0);
  bestSecondMoment.fill(0);
  bestOptimizerStep = 0;
  bestRandomState = randomState >>> 0;
  bestSteps = 0;
  bestExamples = 0;
  bestMovingLoss = 0;
  evaluationInProgress = false;
  stopRequested = false;
  stopRequestReason = "manual";
  bestCheckpointStepsSent = -1;
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
  const profileStart = detailedProfileActive ? performance.now() : 0;
  for (let slot = 0; slot < CONTEXT; slot++) {
    const code = contextCodes[slot] < vocabularySize ? contextCodes[slot] : UNK_ID;
    const source = EMBED_OFFSET + code * EMBED;
    const destination = slot * EMBED;
    for (let j = 0; j < EMBED; j++)
      inputVector[destination + j] = parameters[source + j];
  }

  const embeddingEnd = detailedProfileActive ? performance.now() : 0;
  // Collect active ReLU units as each hidden activation is computed. This
  // avoids a second pass over hiddenVector without changing activation values.
  activeHiddenCount = 0;
  for (let k = 0; k < HIDDEN; k++) {
    let sum = parameters[B1_OFFSET + k];
    const row = W1_OFFSET + k * INPUT;
    for (let j = 0; j < INPUT; j++)
      sum += parameters[row + j] * inputVector[j];
    const activation = sum > 0 ? sum : 0;
    hiddenVector[k] = activation;
    if (activation > 0) activeHiddenIndices[activeHiddenCount++] = k;
  }

  const hiddenEnd = detailedProfileActive ? performance.now() : 0;
  let maximum = -Infinity;
  for (let c = 0; c < vocabularySize; c++) {
    let sum = parameters[B2_OFFSET + c];
    const row = W2_OFFSET + c * HIDDEN;
    // ReLU makes many hidden activations exactly zero. Reuse a compact index
    // list so output projection avoids multiplying by inactive neurons.
    for (let n = 0; n < activeHiddenCount; n++) {
      const k = activeHiddenIndices[n];
      sum += parameters[row + k] * hiddenVector[k];
    }
    probabilities[c] = sum;
    if (sum > maximum) maximum = sum;
  }
  const outputEnd = detailedProfileActive ? performance.now() : 0;
  let denominator = 0;
  for (let c = 0; c < vocabularySize; c++) {
    const value = Math.exp(probabilities[c] - maximum);
    probabilities[c] = value;
    denominator += value;
  }
  const inverse = 1 / Math.max(denominator, 1e-20);
  for (let c = 0; c < vocabularySize; c++) probabilities[c] *= inverse;
  if (detailedProfileActive) {
    profileForwardMs += hiddenEnd - profileStart;
    profileOutputMs += (outputEnd - hiddenEnd) + (performance.now() - outputEnd);
  }
  return probabilities;
}
function forward(data, position) {
  for (let slot = 0; slot < CONTEXT; slot++)
    contextCodes[slot] = tokenAt(data, position - CONTEXT + slot);
  return forwardContext();
}
function trainExample(position) {
  detailedProfileActive = detailedProfiling;
  const target = tokenAt(trainData, position);
  const probs = forward(trainData, position);
  const targetProbability = Math.max(probs[target], 1e-9);
  const loss = -Math.log(targetProbability);
  const backpropStart = detailedProfileActive ? performance.now() : 0;

  hiddenDelta.fill(0);
  for (let c = 0; c < vocabularySize; c++) {
    const delta = probs[c] - (c === target ? 1 : 0);
    if (delta === 0) continue;
    gradients[B2_OFFSET + c] += delta;
    const row = W2_OFFSET + c * HIDDEN;
    // Inactive ReLU units have zero activation and zero backpropagated delta.
    // Visit only active units to avoid zero-gradient reads and writes.
    for (let n = 0; n < activeHiddenCount; n++) {
      const k = activeHiddenIndices[n];
      gradients[row + k] += delta * hiddenVector[k];
      hiddenDelta[k] += delta * parameters[row + k];
    }
  }

  // The forward pass already records active ReLU units in ascending index order.
  // Only those units can have non-zero hidden deltas, so skip the full-width scan.
  for (let n = 0; n < activeHiddenCount; n++) {
    const k = activeHiddenIndices[n];
    const delta = hiddenDelta[k];
    if (delta === 0) continue;
    gradients[B1_OFFSET + k] += delta;
    const row = W1_OFFSET + k * INPUT;
    for (let j = 0; j < INPUT; j++)
      gradients[row + j] += delta * inputVector[j];
  }

  const backpropEnd = detailedProfileActive ? performance.now() : 0;
  // Compute embedding gradients in W1's contiguous row-major layout first.
  // This improves memory locality compared with repeatedly jumping between rows
  // for each individual embedding dimension.
  embeddingInputGradient.fill(0);
  for (let k = 0; k < HIDDEN; k++) {
    const delta = hiddenDelta[k];
    if (delta === 0) continue;
    const row = W1_OFFSET + k * INPUT;
    for (let inputIndex = 0; inputIndex < INPUT; inputIndex++)
      embeddingInputGradient[inputIndex] += delta * parameters[row + inputIndex];
  }
  for (let slot = 0; slot < CONTEXT; slot++) {
    const code = contextCodes[slot];
    const embeddingRow = EMBED_OFFSET + code * EMBED;
    const inputStart = slot * EMBED;
    for (let j = 0; j < EMBED; j++)
      gradients[embeddingRow + j] += embeddingInputGradient[inputStart + j];
  }
  if (detailedProfileActive) {
    profileBackpropMs += backpropEnd - backpropStart;
    profileEmbeddingMs += performance.now() - backpropEnd;
    detailedProfileExamples++;
  }
  detailedProfileActive = false;
  return loss;
}
function applyOptimizer(batchCount) {
  // Only active token rows can receive gradients. Keeping the full checkpoint
  // arrays intact while processing active rows avoids work on unused vocab slots.
  // Cache typed arrays and scalar settings locally in this hot path.
  const activeRanges = activeOptimizerRanges;
  const gradientValues = gradients;
  const parameterValues = parameters;
  const firstMoments = firstMoment;
  const secondMoments = secondMoment;
  let squaredNorm = 0;
  const inverseBatch = 1 / batchCount;
  for (let rangeIndex = 0; rangeIndex < activeRanges.length; rangeIndex++) {
    const range = activeRanges[rangeIndex];
    const start = range[0];
    const end = range[1];
    for (let i = start; i < end; i++) {
      gradientValues[i] *= inverseBatch;
      squaredNorm += gradientValues[i] * gradientValues[i];
    }
  }
  const norm = Math.sqrt(squaredNorm);
  const clip = norm > 5 ? 5 / norm : 1;
  optimizerStep++;
  const correction1 = 1 - Math.pow(0.9, optimizerStep);
  const correction2 = 1 - Math.pow(0.999, optimizerStep);
  const learningRate = activeLearningRate;
  // Algebraically fold Adam's bias corrections into batch-level constants.
  // This removes two divisions per parameter from the optimiser hot loop.
  const sqrtCorrection2 = Math.sqrt(correction2);
  const updateScale = learningRate * sqrtCorrection2 / correction1;
  const scaledEpsilon = 1e-8 * sqrtCorrection2;
  const decay = activeOptimizer === "adamw" ? learningRate * ADAMW_WEIGHT_DECAY : 0;

  for (let rangeIndex = 0; rangeIndex < activeRanges.length; rangeIndex++) {
    const range = activeRanges[rangeIndex];
    const start = range[0];
    const end = range[1];
    const useWeightDecay = decay !== 0 && (
      start === EMBED_OFFSET ||
      start === W1_OFFSET ||
      start === W2_OFFSET
    );
    for (let i = start; i < end; i++) {
      const gradient = gradientValues[i] * clip;
      const gradientSquared = gradient * gradient;
      const m = 0.9 * firstMoments[i] + 0.1 * gradient;
      const v = 0.999 * secondMoments[i] + 0.001 * gradientSquared;
      firstMoments[i] = m;
      secondMoments[i] = v;
      // Equivalent to lr * (m / correction1) /
      // (sqrt(v / correction2) + epsilon), with constants precomputed.
      const adaptiveUpdate = updateScale * m / (Math.sqrt(v) + scaledEpsilon);
      const weightDecay = useWeightDecay ? decay * parameterValues[i] : 0;
      parameterValues[i] -= adaptiveUpdate + weightDecay;
      gradientValues[i] = 0;
    }
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
function postLatestCheckpoint() {
  if (!(parameters instanceof Float32Array)) return false;
  const checkpointState = new Float32Array(PARAMS * 3);
  checkpointState.set(parameters, 0);
  checkpointState.set(firstMoment, PARAMS);
  checkpointState.set(secondMoment, PARAMS * 2);
  const checkpointBuffer = checkpointState.buffer;
  self.postMessage({
    type: 'latest-checkpoint', optimizer: activeOptimizer, learningRate: activeLearningRate,
    accuracy: latestAccuracy, bestAccuracy,
    validationLoss: latestValidationLoss,
    bestValidationLoss: Number.isFinite(bestValidationLoss) ? bestValidationLoss : null,
    perplexity: latestPerplexity, evaluations: evaluationCount, validationSamples,
    steps, examples: examplesSeen, loss: movingLoss, batchSize: activeBatchSize,
    maxSteps: activeMaxSteps, runSteps: Math.max(0, steps - runStartSteps),
    optimizerStep, randomState: randomState >>> 0, parameters: PARAMS,
    assistantTargetMetrics: true, trainingTargetTokens: trainTargetPositions.length,
    validationTargetTokens: validationTargetPositions.length, modelBuffer: checkpointBuffer
  }, [checkpointBuffer]);
  return true;
}
function postBestCheckpoint() {
  if (!(bestParameters instanceof Float32Array) || bestCheckpointStepsSent === bestSteps)
    return false;
  const checkpointState = new Float32Array(PARAMS * 3);
  checkpointState.set(bestParameters, 0);
  checkpointState.set(bestFirstMoment, PARAMS);
  checkpointState.set(bestSecondMoment, PARAMS * 2);
  const checkpointBuffer = checkpointState.buffer;
  self.postMessage({
    type: "checkpoint",
    optimizer: activeOptimizer,
    learningRate: activeLearningRate,
    accuracy: bestAccuracy,
    bestAccuracy,
    validationLoss: bestValidationLoss,
    bestValidationLoss,
    perplexity: Math.exp(bestValidationLoss),
    evaluations: evaluationCount,
    validationSamples,
    steps: bestSteps,
    examples: bestExamples,
    loss: bestMovingLoss,
    batchSize: activeBatchSize,
    maxSteps: activeMaxSteps,
    runSteps: Math.max(0, bestSteps - runStartSteps),
    optimizerStep: bestOptimizerStep,
    randomState: bestRandomState >>> 0,
    parameters: PARAMS,
    assistantTargetMetrics: true,
    trainingTargetTokens: trainTargetPositions.length,
    validationTargetTokens: validationTargetPositions.length,
    modelBuffer: checkpointBuffer
  }, [checkpointBuffer]);
  bestCheckpointStepsSent = bestSteps;
  return true;
}
function evaluateAsync() {
  if (!validationData || validationTargetPositions.length === 0)
    return Promise.resolve(null);
  if (evaluationInProgress)
    return Promise.reject(new Error("A validation pass is already running."));
  evaluationInProgress = true;
  const count = Math.min(EVAL_SAMPLES, validationTargetPositions.length);
  const stride = Math.max(1, Math.floor(validationTargetPositions.length / count));
  let cursor = 0;
  let correct = 0;
  let totalLoss = 0;
  let total = 0;
  const evaluationNumber = evaluationCount + 1;

  return new Promise((resolve, reject) => {
    const processChunk = () => {
      // A manual Stop exits at the next validation slice so the UI stays responsive.
      // The runBatch continuation then restores the best checkpoint and reports "stopped".
      if (!running || stopRequested) {
        evaluationInProgress = false;
        resolve(null);
        return;
      }
      try {
        const end = Math.min(count, cursor + EVAL_CHUNK_SIZE);
        for (; cursor < end; cursor++) {
          const position = validationTargetPositions[Math.min(cursor * stride, validationTargetPositions.length - 1)];
          const result = predict(validationData, position);
          const target = tokenAt(validationData, position);
          if (result.token === target) correct++;
          totalLoss += -Math.log(Math.max(probabilities[target], 1e-9));
          total++;
        }
        self.postMessage({
          type: "validation-progress",
          evaluation: evaluationNumber,
          completed: cursor,
          total: count,
          steps,
          stopRequested
        });
        if (cursor < count) {
          setTimeout(processChunk, 0);
          return;
        }
        if (!total) {
          evaluationInProgress = false;
          resolve(null);
          return;
        }

        const score = correct * 100 / total;
        const validationLoss = totalLoss / total;
        const perplexity = Math.exp(validationLoss);
        latestAccuracy = score;
        latestValidationLoss = validationLoss;
        latestPerplexity = perplexity;
        evaluationCount++;
        validationSamples = total;

        const improved = validationLoss < bestValidationLoss - MIN_VALIDATION_IMPROVEMENT;
        if (improved) {
          bestValidationLoss = validationLoss;
          bestAccuracy = score;
          bestParameters = new Float32Array(parameters);
          bestFirstMoment.set(firstMoment);
          bestSecondMoment.set(secondMoment);
          bestOptimizerStep = optimizerStep;
          bestRandomState = randomState >>> 0;
          bestSteps = steps;
          bestExamples = examplesSeen;
          bestMovingLoss = movingLoss;
        }

        // Preserve the best model separately and always persist the newest state.
        if (improved) postBestCheckpoint();
        postLatestCheckpoint();
        // Always report the completed check, including when a new best checkpoint was saved.
        self.postMessage({
          type: "evaluation",
          optimizer: activeOptimizer,
          learningRate: activeLearningRate,
          accuracy: score,
          bestAccuracy,
          validationLoss,
          bestValidationLoss,
          perplexity,
          evaluations: evaluationCount,
          improved,
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
          assistantTargetMetrics: true,
          trainingTargetTokens: trainTargetPositions.length,
          validationTargetTokens: validationTargetPositions.length
        });
        evaluationInProgress = false;
        resolve({accuracy: score, validationLoss, perplexity, samples: total, evaluations: evaluationCount});
      } catch (error) {
        evaluationInProgress = false;
        reject(error);
      }
    };
    self.postMessage({
      type: "validation-progress",
      evaluation: evaluationNumber,
      completed: 0,
      total: count,
      steps,
      stopRequested
    });
    setTimeout(processChunk, 0);
  });
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
  const payload = {
    steps,
    examples: examplesSeen,
    epochs: trainTargetPositions.length ? examplesSeen / trainTargetPositions.length : 0,
    trainingTargetTokens: trainTargetPositions.length,
    validationTargetTokens: validationTargetPositions.length,
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
    validationTokens: validationData ? validationData.length : 0,
    runtimeProfile: { batches: profilingBatches, totalBatchMs: Math.round(profilingBatchMs), selectionMs: Math.round(profilingSelectionMs), exampleComputeMs: Math.round(profilingExampleMs), optimizerMs: Math.round(profilingOptimizerMs), detailed: detailedProfiling, detailedExamples: detailedProfileExamples, forwardMs: Math.round(profileForwardMs), outputMs: Math.round(profileOutputMs), backpropMs: Math.round(profileBackpropMs), embeddingMs: Math.round(profileEmbeddingMs) }
  };
  if (now - lastNetworkSnapshotAt >= NETWORK_SNAPSHOT_INTERVAL_MS) {
    payload.network = networkSnapshot();
    lastNetworkSnapshotAt = now;
  }
  return payload;
}
function runBatch() {
  if (!running) return;
  const batchProfileStart = performance.now();
  try {
    if (!trainTargetPositions.length) throw new Error("No assistant-response training targets were found.");
    let batchLoss = 0;
    // Select the same balanced examples in the same RNG order, then time model
    // computation separately. Reuse the fixed buffer to avoid per-batch allocation.
    const selectionStart = performance.now();
    for (let i = 0; i < activeBatchSize; i++)
      batchPositions[i] = sampleBalancedTrainingPosition();
    const selectionEnd = performance.now();
    const exampleProfileStart = selectionEnd;
    for (let i = 0; i < activeBatchSize; i++)
      batchLoss += trainExample(batchPositions[i]);
    const exampleProfileEnd = performance.now();
    applyOptimizer(activeBatchSize);
    const optimizerProfileEnd = performance.now();
    profilingSelectionMs += selectionEnd - selectionStart;
    profilingExampleMs += exampleProfileEnd - exampleProfileStart;
    profilingOptimizerMs += optimizerProfileEnd - exampleProfileEnd;
    profilingBatchMs += optimizerProfileEnd - batchProfileStart;
    profilingBatches++;
    steps++;
    examplesSeen += activeBatchSize;
    const averageLoss = batchLoss / activeBatchSize;
    movingLoss = lossInitialized ? movingLoss * 0.92 + averageLoss * 0.08 : averageLoss;
    lossInitialized = true;

    if (steps % EVAL_EVERY === 0) {
      evaluateAsync().then(() => {
        if (stopRequested) {
          finishStop(stopRequestReason);
          return;
        }
        if (!running) return;
        setTimeout(runBatch, 0);
      }).catch(error => {
        evaluationInProgress = false;
        running = false;
        stopRequested = false;
        self.postMessage({type: "error", message: error && error.message ? error.message : String(error)});
      });
      return;
    }
    if (activeMaxSteps > 0 && steps - runStartSteps >= activeMaxSteps) {
      stopTraining("limit");
      return;
    }
    const now = performance.now();
    if (running && now - lastProgressMessageAt >= PROGRESS_INTERVAL_MS) {
      self.postMessage({type: "progress", ...progressPayload()});
      lastProgressMessageAt = now;
    }
    if (running) setTimeout(runBatch, 0);
  } catch (error) {
    running = false;
    self.postMessage({type: "error", message: error && error.message ? error.message : String(error)});
  }
}
function finishStop(reason = "manual") {
  running = false;
  stopRequested = false;

  // Capture the exact latest state before optionally restoring the best validation state.
  postLatestCheckpoint();
  const restoredBest = bestParameters instanceof Float32Array;
  if (restoredBest) {
    parameters.set(bestParameters);
    firstMoment.set(bestFirstMoment);
    secondMoment.set(bestSecondMoment);
    optimizerStep = bestOptimizerStep;
    randomState = bestRandomState >>> 0;
    steps = bestSteps;
    examplesSeen = bestExamples;
    movingLoss = bestMovingLoss;
    lossInitialized = true;
    latestAccuracy = bestAccuracy;
    latestValidationLoss = bestValidationLoss;
    latestPerplexity = Math.exp(bestValidationLoss);
    postBestCheckpoint();
  }
  self.postMessage({
    type: "stopped",
    ...progressPayload(),
    restoredBest,
    stopReason: reason,
    bestSteps: restoredBest ? bestSteps : steps
  });
}
function stopTraining(reason = "manual") {
  if (!running && !evaluationInProgress) return;
  if (evaluationInProgress) {
    stopRequested = true;
    stopRequestReason = reason;
    return;
  }
  finishStop(reason);
}
function generate(prompt, requestedLength, greedy = false) {
  if (!initialized || vocabularySize <= 3) return "Load or restore the neural model first.";
  const rawPrompt = String(prompt || "").trim().slice(0, 180);
  const modelPrompt = "user: " + rawPrompt.toLowerCase() + "\nsherlock: ";
  const promptEncoded = encodeText(modelPrompt);
  const history = Array.from(promptEncoded.data);
  const maxLength = Math.max(1, Math.min(160, Number(requestedLength) || 80));
  let output = "";

  for (let i = 0; i < maxLength; i++) {
    for (let slot = 0; slot < CONTEXT; slot++) {
      const sourceIndex = history.length - CONTEXT + slot;
      contextCodes[slot] = sourceIndex >= 0 ? history[sourceIndex] : SPACE_ID;
    }
    forwardContext();

    const previousToken = history.length ? history[history.length - 1] : SPACE_ID;
    let chosen = SPACE_ID;
    if (greedy) {
      // Greedy mode avoids sorting and sampling entirely: a single argmax scan.
      let bestProbability = -1;
      for (let c = 1; c < vocabularySize; c++) {
        if ((c === SPACE_ID || c === NEWLINE_ID) &&
            (previousToken === SPACE_ID || previousToken === NEWLINE_ID)) continue;
        if (probabilities[c] > bestProbability) {
          bestProbability = probabilities[c];
          chosen = c;
        }
      }
    } else {
      const candidates = [];
      let total = 0;
      const inverseTemperature = 1 / 0.85;
      for (let c = 1; c < vocabularySize; c++) {
        // Avoid repeated whitespace tokens.
        if ((c === SPACE_ID || c === NEWLINE_ID) &&
            (previousToken === SPACE_ID || previousToken === NEWLINE_ID)) continue;
        const weight = Math.pow(Math.max(probabilities[c], 1e-20), inverseTemperature);
        probabilities[c] = weight;
        total += weight;
        candidates.push(c);
      }
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
      chosen = candidates[nucleusCount - 1] ?? SPACE_ID;
      for (let j = 0; j < nucleusCount; j++) {
        const id = candidates[j];
        pick -= probabilities[id];
        if (pick <= 0) { chosen = id; break; }
      }
    }
    if (chosen === EOS_ID) break;
    history.push(chosen);
    output += vocabulary[chosen] || "";
  }
  return output;
}

function setDatasetBuffers(trainBuffer, validationBuffer) {
  const decoder = new TextDecoder("utf-8");
  const trainingText = markAssistantEndTokens(decoder.decode(new Uint8Array(trainBuffer)).toLowerCase());
  const validationText = markAssistantEndTokens(decoder.decode(new Uint8Array(validationBuffer)).toLowerCase());
  buildTokenizer(trainingText);
  const encodedTraining = encodeText(trainingText);
  const encodedValidation = encodeText(validationText);
  trainData = encodedTraining.data;
  validationData = encodedValidation.data;
  trainTargetPositions = collectAssistantTargetPositions(trainingText);
  trainTargetPositionsByFamily = collectAssistantTargetPositionsByFamily(trainingText);
  validationTargetPositions = collectAssistantTargetPositions(validationText);
  trainingUnknownRate = encodedTraining.unknownRate;
  validationUnknownRate = encodedValidation.unknownRate;
  if (trainData.length <= CONTEXT + 10 || validationData.length <= CONTEXT + 10)
    throw new Error("The tokenized training or validation text is too short.");
  if (!trainTargetPositions.length || !validationTargetPositions.length)
    throw new Error("Assistant-response targets were not found. Expected user: and sherlock: turns.");
  const missingFamilies = TASK_FAMILY_NAMES.filter((name, index) => !trainTargetPositionsByFamily[index] || !trainTargetPositionsByFamily[index].length);
  if (missingFamilies.length)
    throw new Error("Balanced Learning is missing training targets for: " + missingFamilies.join(", ") + ".");
  return {
    trainTokens: trainData.length,
    validationTokens: validationData.length,
    trainingTargetTokens: trainTargetPositions.length,
    validationTargetTokens: validationTargetPositions.length,
    balancedTaskFamilies: TASK_FAMILY_NAMES.length,
    balancedTargetTokensByFamily: trainTargetPositionsByFamily.map(group => group.length),
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
      bestFirstMoment.set(firstMoment);
      bestSecondMoment.set(secondMoment);
      bestOptimizerStep = Math.max(0, Math.floor(Number(message.optimizerStep) || 0));
      bestRandomState = (Number(message.randomState) >>> 0) || 0x5A17C0DE;
      bestSteps = Math.max(0, Number(message.steps) || 0);
      bestCheckpointStepsSent = bestSteps;
      bestExamples = Math.max(0, Number(message.examples) || 0);
      bestMovingLoss = Math.max(0, Number(message.loss) || 0);
      bestAccuracy = Number.isFinite(Number(message.bestAccuracy)) ? Number(message.bestAccuracy) :
        (Number.isFinite(Number(message.accuracy)) ? Number(message.accuracy) : -1);
      latestAccuracy = Number.isFinite(Number(message.accuracy)) ? Number(message.accuracy) : null;
      latestValidationLoss = message.validationLoss !== null && message.validationLoss !== undefined && Number.isFinite(Number(message.validationLoss))
        ? Number(message.validationLoss) : null;
      bestValidationLoss = message.bestValidationLoss !== null && message.bestValidationLoss !== undefined && Number.isFinite(Number(message.bestValidationLoss))
        ? Number(message.bestValidationLoss) : (latestValidationLoss === null ? Infinity : latestValidationLoss);
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
    } else if (message.type === "diagnose-unknown-tokens") {
      const report = diagnoseUnknownTokens(message.trainBuffer, message.validationBuffer);
      self.postMessage({type: "unknown-token-diagnosis", id: message.id, ...report});
    } else if (message.type === "start") {
      if (!initialized) throw new Error("Load or restore the neural model first.");
      const requestedOptimizer = message.optimizer === "adamw" || (message.optimizer == null && activeOptimizer === "adamw") ? "adamw" : "adam";
      const requestedLearningRate = [0.001, 0.003, 0.005].includes(Number(message.learningRate))
        ? Number(message.learningRate) : DEFAULT_LEARNING_RATE;
      activeLearningRate = requestedLearningRate;
      detailedProfiling = message.detailedProfiling === true;
      profileForwardMs = profileOutputMs = profileBackpropMs = profileEmbeddingMs = 0;
      detailedProfileExamples = 0;
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
        lastProgressMessageAt = -Infinity;
        lastNetworkSnapshotAt = -Infinity;
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
        stopRequested = false;
        stopRequestReason = "manual";
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
        self.postMessage({type: "generated", id: message.id, text: generate(message.prompt, message.length, !!message.greedy)});
      } finally {
        randomState = savedRandomState;
      }
    }
  } catch (error) {
    if (message.type !== "diagnose-unknown-tokens") running = false;
    self.postMessage({
      type: message.type === "diagnose-unknown-tokens" ? "unknown-token-diagnosis-error" : "error",
      id: message.id,
      message: error && error.message ? error.message : String(error)
    });
  }
});