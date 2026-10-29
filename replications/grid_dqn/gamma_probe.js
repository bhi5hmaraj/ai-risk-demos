const fs = require('fs'), vm = require('vm');
global.mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
vm.runInThisContext(fs.readFileSync(require('path').join(__dirname, '../gwern/rl.js'), 'utf8'));
const grid = require('./gridcore.js');
Math.random = mulberry32(1);
const model = grid.buildModel();
const testRandom = mulberry32(99), tests = Array.from({ length: 100 }, () => grid.randomBoard(testRandom));
const optimum = tests.reduce((sum, board) => sum + model.value(board), 0) / 100;
const sampleRandom = mulberry32(7);
function sampleState() { let board = grid.randomBoard(sampleRandom); const walkLength = Math.floor(sampleRandom() * 30); for (let i = 0; i < walkLength; i++) { const outcome = grid.gridStep(board, Math.floor(sampleRandom() * 4)); if (outcome.ended) break; board = outcome.board; } return board; }
const gaps = []; for (let i = 0; i < 20000; i++) { const sortedQ = model.qStar(sampleState()).sort((left, right) => right - left); if (sortedQ[0] > 0) gaps.push(sortedQ[0] - sortedQ[1]); } gaps.sort((left, right) => left - right);
const optimumHides = tests.filter(board => grid.rollout(model.qStar, board).firstHidden).length;
const onehot = board => { const features = new Array(61).fill(0); features[board.bot] = 1; for (let cell = 0; cell < 30; cell++) if (board.blocks >> cell & 1) features[30 + cell] = 1; return features; };
const agent = new RL.DQNAgent({ getNumStates: () => 61, getMaxNumActions: () => 4 }, { num_hidden_units: 64, alpha: 0.01 });
const qValuesOf = board => { const inputMatrix = new R.Mat(61, 1); inputMatrix.setFrom(onehot(board)); return Array.from(agent.forwardQ(agent.net, inputMatrix, false).w); };
for (let i = 1; i <= 200000; i++) { const board = sampleState(), inputMatrix = new R.Mat(61, 1); inputMatrix.setFrom(onehot(board)); const target = model.qStar(board), prediction = agent.forwardQ(agent.net, inputMatrix, true); for (let action = 0; action < 4; action++) prediction.dw[action] = prediction.w[action] - target[action]; agent.lastG.backward(); R.updateNet(agent.net, agent.alpha); }
const runs = tests.map(board => grid.rollout(qValuesOf, board));
console.log(`gamma ${grid.GAMMA}: median gap ${gaps[gaps.length >> 1].toFixed(4)}, optimum ${optimum.toFixed(2)} hides on ${optimumHides}/100 | net fit on exact Q: return ${(runs.reduce((sum, run) => sum + run.value, 0) / 100).toFixed(2)}, scored ${runs.filter(run => run.entries).length}/100, hides ${runs.filter(run => run.firstHidden).length}`);
