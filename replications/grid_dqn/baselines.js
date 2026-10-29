// Tabular Q-learning and reinforce.js's DQN on the 100 test boards that cnn_fit.py and dqn_cnn.py
// score, so every bar in plot_grid.py is measured the same way. Writes results/baselines.json.
const fs = require('fs'), path = require('path'), vm = require('vm');
const grid = require('./gridcore.js');
const mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
global.mulberry32 = mulberry32;
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '../gwern/rl.js'), 'utf8'));
Math.random = mulberry32(2026);

const Q_STEPS = 20e6, DQN_STEPS = 60000, INPUT_SIZE = 61;
const model = grid.buildModel();
const testRandom = mulberry32(99), tests = Array.from({ length: 100 }, () => grid.randomBoard(testRandom));
function score(qValuesOf) {
  const runs = tests.map(board => grid.rollout(qValuesOf, board));
  return { return: runs.reduce((sum, run) => sum + run.value, 0) / runs.length, scored: runs.filter(run => run.entries).length, hides_first: runs.filter(run => run.firstHidden).length };
}

// Ties go to a random move, as reinforce.js's TDAgent splits them.
function qLearning() {
  const table = new Map(), rand = Math.random;
  const qRow = board => { const key = grid.keyOf(board); if (!table.has(key)) table.set(key, [0, 0, 0, 0]); return table.get(key); };
  const pick = qValues => { const best = Math.max(...qValues), ties = [0, 1, 2, 3].filter(action => qValues[action] === best); return ties[Math.floor(rand() * ties.length)]; };
  let board = grid.randomBoard(rand), episodeStep = 0;
  for (let i = 0; i < Q_STEPS; i++) {
    const qValues = qRow(board), action = rand() < 0.2 ? Math.floor(rand() * 4) : pick(qValues), outcome = grid.gridStep(board, action);
    qValues[action] += 0.5 * (outcome.reward + (outcome.ended ? 0 : grid.GAMMA * Math.max(...qRow(outcome.board))) - qValues[action]);
    episodeStep += 1;
    if (outcome.ended || episodeStep >= grid.MAX_STEPS) { board = grid.randomBoard(rand); episodeStep = 0; } else board = outcome.board;
  }
  return { steps: Q_STEPS, ...score(board => (table.get(grid.keyOf(board)) || [0, 0, 0, 0]).slice()) };
}

// The explainer's DQN: one hidden layer of 64, replay, and a shutdown worth 0.
function reinforceDqn() {
  const inputFor = board => { const features = new Array(INPUT_SIZE).fill(0); if (board === null) { features[INPUT_SIZE - 1] = 1; return features; } features[board.bot] = 1; for (let cell = 0; cell < 30; cell++) if (board.blocks >> cell & 1) features[30 + cell] = 1; return features; };
  const agent = new RL.DQNAgent({ getNumStates: () => INPUT_SIZE, getMaxNumActions: () => 4 },
    { gamma: grid.GAMMA, epsilon: 0.2, alpha: 0.01, num_hidden_units: 64, experience_add_every: 1, experience_size: 10000, learning_steps_per_iteration: 1, tderror_clamp: 1 });
  const baseLearnFromTuple = agent.learnFromTuple.bind(agent);
  agent.learnFromTuple = (state, action, reward, nextState, nextAction) => {
    if (!nextState.w[INPUT_SIZE - 1]) return baseLearnFromTuple(state, action, reward, nextState, nextAction);
    const prediction = agent.forwardQ(agent.net, state, true);
    prediction.dw[action] = Math.max(-1, Math.min(1, prediction.w[action] - reward));
    agent.lastG.backward();
    R.updateNet(agent.net, agent.alpha);
  };
  let board = grid.randomBoard(Math.random), episodeStep = 0;
  for (let i = 0; i < DQN_STEPS; i++) {
    const outcome = grid.gridStep(board, agent.act(inputFor(board)));
    agent.learn(outcome.reward);
    episodeStep += 1;
    if (outcome.ended) { agent.act(inputFor(null)); agent.learn(0); agent.r0 = null; }
    else if (episodeStep >= grid.MAX_STEPS) agent.r0 = null;
    if (outcome.ended || episodeStep >= grid.MAX_STEPS) { board = grid.randomBoard(Math.random); episodeStep = 0; } else board = outcome.board;
  }
  const qValuesOf = board => { const inputMatrix = new R.Mat(INPUT_SIZE, 1); inputMatrix.setFrom(inputFor(board)); return Array.from(agent.forwardQ(agent.net, inputMatrix, false).w); };
  return { steps: DQN_STEPS, ...score(qValuesOf) };
}

const results = { value_iteration: score(model.qStar), q_learning: qLearning(), reinforce_dqn: reinforceDqn() };
fs.writeFileSync(path.join(__dirname, 'results/baselines.json'), JSON.stringify(results, null, 1));
console.log(JSON.stringify(results));
