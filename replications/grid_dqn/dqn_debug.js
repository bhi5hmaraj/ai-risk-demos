const fs = require('fs'), vm = require('vm');
global.mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
vm.runInThisContext(fs.readFileSync(require('path').join(__dirname, '../gwern/rl.js'), 'utf8'));
const grid = require('./gridcore.js');
Math.random = mulberry32(1);
const model = grid.buildModel();
const onehot = board => { const features = new Array(61).fill(0); features[board.bot] = 1; for (let cell = 0; cell < 30; cell++) if (board.blocks >> cell & 1) features[30 + cell] = 1; return features; };
const sampleRandom = mulberry32(7);
function sampleState() { let board = grid.randomBoard(sampleRandom); const walkLength = Math.floor(sampleRandom() * 30); for (let i = 0; i < walkLength; i++) { const outcome = grid.gridStep(board, Math.floor(sampleRandom() * 4)); if (outcome.ended) break; board = outcome.board; } return board; }
const agent = new RL.DQNAgent({ getNumStates: () => 61, getMaxNumActions: () => 4 }, { num_hidden_units: 64, alpha: 0.01 });
const qValuesOf = board => { const inputMatrix = new R.Mat(61, 1); inputMatrix.setFrom(onehot(board)); return Array.from(agent.forwardQ(agent.net, inputMatrix, false).w); };
const heldOut = Array.from({ length: 3000 }, sampleState);
function report(tag) {
  let squaredError = 0, agreeCount = 0, count = 0;
  for (const board of heldOut) { const exact = model.qStar(board), predicted = qValuesOf(board); for (let action = 0; action < 4; action++) squaredError += (predicted[action] - exact[action]) ** 2; const best = Math.max(...exact); if (exact[grid.argmax(predicted)] >= best - 1e-12) agreeCount++; count++; }
  const example = heldOut[0]; console.log(tag, 'rmse', Math.sqrt(squaredError / count / 4).toFixed(4), 'picks an exact-best move', (100 * agreeCount / count).toFixed(1) + '%', 'example pred', qValuesOf(example).map(value => value.toFixed(3)).join(' '), 'exact', model.qStar(example).map(value => value.toFixed(3)).join(' '));
}
report('untrained');
for (let i = 1; i <= 200000; i++) {
  const board = sampleState(), inputMatrix = new R.Mat(61, 1); inputMatrix.setFrom(onehot(board));
  const target = model.qStar(board), prediction = agent.forwardQ(agent.net, inputMatrix, true);
  for (let action = 0; action < 4; action++) prediction.dw[action] = prediction.w[action] - target[action];
  agent.lastG.backward(); R.updateNet(agent.net, agent.alpha);
  if (i % 50000 === 0) report(`${i / 1000}k`);
}
