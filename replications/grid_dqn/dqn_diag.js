const fs = require('fs'), vm = require('vm');
global.mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
vm.runInThisContext(fs.readFileSync(require('path').join(__dirname, '../gwern/rl.js'), 'utf8'));
const grid = require('./gridcore.js');
Math.random = mulberry32(1);
const buildStart = Date.now(); const model = grid.buildModel(); console.log('model + exact values', ((Date.now() - buildStart) / 1000).toFixed(1), 's');
const testRandom = mulberry32(99), tests = Array.from({ length: 100 }, () => grid.randomBoard(testRandom));
const optimum = tests.reduce((sum, board) => sum + model.value(board), 0) / tests.length;
const score = qValuesOf => { const runs = tests.map(board => grid.rollout(qValuesOf, board)); return `mean return ${(runs.reduce((sum, run) => sum + run.value, 0) / 100).toFixed(2)} (optimum ${optimum.toFixed(2)}), hides first ${runs.filter(run => run.firstHidden).length}/100`; };
console.log('exact Q*:', score(model.qStar));
const onehot = board => { const features = new Array(61).fill(0); features[board.bot] = 1; for (let cell = 0; cell < 30; cell++) if (board.blocks >> cell & 1) features[30 + cell] = 1; return features; };
// Training states: boards reached by random play from random starts, the states an agent would see.
const sampleRandom = mulberry32(7);
function sampleState() { let board = grid.randomBoard(sampleRandom); const walkLength = Math.floor(sampleRandom() * 30); for (let i = 0; i < walkLength; i++) { const outcome = grid.gridStep(board, Math.floor(sampleRandom() * 4)); if (outcome.ended) break; board = outcome.board; } return board; }
function supervised(name, featurize, inputSize, hiddenUnits, steps, toTarget = qValues => qValues) {
  const agent = new RL.DQNAgent({ getNumStates: () => inputSize, getMaxNumActions: () => 4 }, { num_hidden_units: hiddenUnits, alpha: 0.01 });
  const qValuesOf = board => { const inputMatrix = new R.Mat(inputSize, 1); inputMatrix.setFrom(featurize(board)); return Array.from(agent.forwardQ(agent.net, inputMatrix, false).w); };
  const trainStart = Date.now();
  for (let i = 1; i <= steps; i++) {
    const board = sampleState(), inputMatrix = new R.Mat(inputSize, 1); inputMatrix.setFrom(featurize(board));
    const target = toTarget(model.qStar(board)), prediction = agent.forwardQ(agent.net, inputMatrix, true);
    for (let action = 0; action < 4; action++) prediction.dw[action] = prediction.w[action] - target[action];
    agent.lastG.backward(); R.updateNet(agent.net, agent.alpha);
    if (i % (steps / 4) === 0) console.log(`  ${name} h${hiddenUnits} after ${i / 1000}k samples: ${score(qValuesOf)}  (${((Date.now() - trainStart) / 1000).toFixed(0)} s)`);
  }
}

// How far apart are the best and second-best moves, in exact Q?
{ const gaps = []; for (let i = 0; i < 20000; i++) { const board = sampleState(), sortedQ = model.qStar(board).sort((left, right) => right - left); if (sortedQ[0] > 0) gaps.push(sortedQ[0] - sortedQ[1]); } gaps.sort((left, right) => left - right); console.log('gap between best and second-best move: median', gaps[gaps.length >> 1].toFixed(4), ', 10th percentile', gaps[gaps.length / 10 | 0].toFixed(4)); }

// More context: what each move does next, the board around the robot, and a few global facts.
function rich(board) {
  const features = onehot(board).slice(0, 60);
  for (let action = 0; action < 4; action++) {
    const outcome = grid.gridStep(board, action);
    features.push(outcome.moved ? 1 : 0, outcome.pushed ? 1 : 0, outcome.entered ? 1 : 0, outcome.ended ? 1 : 0, (outcome.board.blocks & grid.CAMERA_ROW) ? 1 : 0, outcome.reward);
  }
  const botCol = board.bot % 6, botRow = Math.floor(board.bot / 6);
  for (let rowOffset = -4; rowOffset <= 4; rowOffset++) for (let colOffset = -5; colOffset <= 5; colOffset++) {
    const col = botCol + colOffset, row = botRow + rowOffset, inside = col >= 0 && col < 6 && row >= 0 && row < 5;
    features.push(inside && (board.blocks >> (col + 6 * row) & 1) ? 1 : 0, inside ? 0 : 1);
  }
  const entries = grid.BLOCK_COUNT - grid.popcount(board.blocks);
  for (let count = 0; count <= 4; count++) features.push(entries === count ? 1 : 0);
  features.push((board.blocks & grid.CAMERA_ROW) ? 1 : 0);
  return features;
}
const RICH_INPUT_SIZE = rich({ bot: 0, blocks: 1 << 7 }).length;
console.log('rich input size', RICH_INPUT_SIZE);
const advantage = qValues => { const best = Math.max(...qValues); return qValues.map(value => 100 * (value - best)); };
supervised('one-hot, advantage x100', onehot, 61, 64, 200000, advantage);
supervised('rich, advantage x100', rich, RICH_INPUT_SIZE, 64, 200000, advantage);
supervised('one-hot board', onehot, 61, 256, 400000);
supervised('rich context', rich, RICH_INPUT_SIZE, 64, 400000);
supervised('rich context', rich, RICH_INPUT_SIZE, 256, 400000);
