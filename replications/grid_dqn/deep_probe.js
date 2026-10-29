const grid = require('./gridcore.js');
const mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
const model = grid.buildModel();
const testRandom = mulberry32(99), tests = Array.from({ length: 100 }, () => grid.randomBoard(testRandom));
const optimum = tests.reduce((sum, board) => sum + model.value(board), 0) / 100;
const sampleRandom = mulberry32(7), weightRandom = mulberry32(3);
function sampleState() { let board = grid.randomBoard(sampleRandom); const walkLength = Math.floor(sampleRandom() * 30); for (let i = 0; i < walkLength; i++) { const outcome = grid.gridStep(board, Math.floor(sampleRandom() * 4)); if (outcome.ended) break; board = outcome.board; } return board; }
const INPUT_SIZE = 60, HIDDEN_SIZE = 256;
const makeLayer = (inputs, outputs) => ({ weights: Float32Array.from({ length: inputs * outputs }, () => (weightRandom() - 0.5) * 2 * Math.sqrt(6 / (inputs + outputs))), biases: new Float32Array(outputs), inputs, outputs });
const layers = [makeLayer(INPUT_SIZE, HIDDEN_SIZE), makeLayer(HIDDEN_SIZE, HIDDEN_SIZE), makeLayer(HIDDEN_SIZE, 4)];
const adamState = layers.map(layer => ({ weightMean: new Float32Array(layer.weights.length), weightVariance: new Float32Array(layer.weights.length), biasMean: new Float32Array(layer.biases.length), biasVariance: new Float32Array(layer.biases.length) }));
const activeInputs = board => { const inputs = [board.bot]; for (let cell = 0; cell < 30; cell++) if (board.blocks >> cell & 1) inputs.push(30 + cell); return inputs; };
function forward(board) {
  const active = activeInputs(board), hidden1 = new Float32Array(HIDDEN_SIZE), hidden2 = new Float32Array(HIDDEN_SIZE), output = new Float32Array(4);
  for (let j = 0; j < HIDDEN_SIZE; j++) { let sum = layers[0].biases[j]; for (const i of active) sum += layers[0].weights[i * HIDDEN_SIZE + j]; hidden1[j] = sum > 0 ? sum : 0; }
  for (let j = 0; j < HIDDEN_SIZE; j++) { let sum = layers[1].biases[j]; for (let i = 0; i < HIDDEN_SIZE; i++) if (hidden1[i]) sum += hidden1[i] * layers[1].weights[i * HIDDEN_SIZE + j]; hidden2[j] = sum > 0 ? sum : 0; }
  for (let action = 0; action < 4; action++) { let sum = layers[2].biases[action]; for (let i = 0; i < HIDDEN_SIZE; i++) if (hidden2[i]) sum += hidden2[i] * layers[2].weights[i * 4 + action]; output[action] = sum; }
  return { active, hidden1, hidden2, output };
}
let step = 0;
function trainBatch(batch) {
  const grads = layers.map(layer => ({ weights: new Float32Array(layer.weights.length), biases: new Float32Array(layer.biases.length) }));
  for (const board of batch) {
    const { active, hidden1, hidden2, output } = forward(board), target = model.qStar(board);
    const outputGrad = [0, 1, 2, 3].map(action => (output[action] - target[action]) / batch.length);
    const hidden2Grad = new Float32Array(HIDDEN_SIZE), hidden1Grad = new Float32Array(HIDDEN_SIZE);
    for (let i = 0; i < HIDDEN_SIZE; i++) { if (!hidden2[i]) continue; let sum = 0; for (let action = 0; action < 4; action++) { grads[2].weights[i * 4 + action] += hidden2[i] * outputGrad[action]; sum += layers[2].weights[i * 4 + action] * outputGrad[action]; } hidden2Grad[i] = sum; }
    for (let action = 0; action < 4; action++) grads[2].biases[action] += outputGrad[action];
    for (let i = 0; i < HIDDEN_SIZE; i++) { if (!hidden1[i]) continue; let sum = 0; for (let j = 0; j < HIDDEN_SIZE; j++) { if (!hidden2Grad[j]) continue; grads[1].weights[i * HIDDEN_SIZE + j] += hidden1[i] * hidden2Grad[j]; sum += layers[1].weights[i * HIDDEN_SIZE + j] * hidden2Grad[j]; } hidden1Grad[i] = sum; }
    for (let j = 0; j < HIDDEN_SIZE; j++) grads[1].biases[j] += hidden2Grad[j];
    for (const i of active) for (let j = 0; j < HIDDEN_SIZE; j++) grads[0].weights[i * HIDDEN_SIZE + j] += hidden1Grad[j];
    for (let j = 0; j < HIDDEN_SIZE; j++) grads[0].biases[j] += hidden1Grad[j];
  }
  step++;
  const learningRate = 0.001, meanCorrection = 1 - 0.9 ** step, varianceCorrection = 1 - 0.999 ** step;
  layers.forEach((layer, layerIndex) => { for (const [params, grad, mean, variance] of [[layer.weights, grads[layerIndex].weights, adamState[layerIndex].weightMean, adamState[layerIndex].weightVariance], [layer.biases, grads[layerIndex].biases, adamState[layerIndex].biasMean, adamState[layerIndex].biasVariance]]) for (let i = 0; i < params.length; i++) { if (!grad[i]) continue; mean[i] = 0.9 * mean[i] + 0.1 * grad[i]; variance[i] = 0.999 * variance[i] + 0.001 * grad[i] * grad[i]; params[i] -= learningRate * (mean[i] / meanCorrection) / (Math.sqrt(variance[i] / varianceCorrection) + 1e-8); } });
}
const heldOut = Array.from({ length: 2000 }, sampleState);
const trainStart = Date.now();
for (let iteration = 1; iteration <= 6000; iteration++) {
  trainBatch(Array.from({ length: 64 }, sampleState));
  if (iteration % 1500 === 0) {
    let agreeCount = 0, squaredError = 0; for (const board of heldOut) { const exact = model.qStar(board), predicted = forward(board).output; for (let action = 0; action < 4; action++) squaredError += (predicted[action] - exact[action]) ** 2; if (exact[grid.argmax(predicted)] >= Math.max(...exact) - 1e-12) agreeCount++; }
    const runs = tests.map(start => grid.rollout(board => Array.from(forward(board).output), start));
    console.log(`${iteration * 64 / 1000}k samples: rmse ${Math.sqrt(squaredError / 8000).toFixed(4)}, picks best ${(agreeCount / 20).toFixed(1)}%, return ${(runs.reduce((sum, run) => sum + run.value, 0) / 100).toFixed(2)} of ${optimum.toFixed(2)}, scored ${runs.filter(run => run.entries).length}/100, hides ${runs.filter(run => run.firstHidden).length}  (${((Date.now() - trainStart) / 1000).toFixed(0)} s)`);
  }
}
