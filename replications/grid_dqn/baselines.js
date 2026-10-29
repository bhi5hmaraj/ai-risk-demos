// Tabular Q-learning and reinforce.js's DQN on the 100 test boards that cnn_fit.py and dqn_cnn.py
// score, so every bar in plot_grid.py is measured the same way. Writes results/baselines.json.
const fs = require('fs'), path = require('path'), vm = require('vm');
const g = require('./gridcore.js');
const mulberry32 = seed => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
global.mulberry32 = mulberry32;
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '../gwern/rl.js'), 'utf8'));
Math.random = mulberry32(2026);

const Q_STEPS = 20e6, DQN_STEPS = 60000, IN = 61;
const model = g.buildModel();
const testRand = mulberry32(99), tests = Array.from({ length: 100 }, () => g.randomBoard(testRand));
function score(q) {
  const runs = tests.map(b => g.rollout(q, b));
  return { return: runs.reduce((s, r) => s + r.value, 0) / runs.length, scored: runs.filter(r => r.entries).length, hides_first: runs.filter(r => r.firstHidden).length };
}

// Ties go to a random move, as reinforce.js's TDAgent splits them.
function qLearning() {
  const table = new Map(), rand = Math.random;
  const row = b => { const k = g.keyOf(b); if (!table.has(k)) table.set(k, [0, 0, 0, 0]); return table.get(k); };
  const pick = r => { const best = Math.max(...r), ties = [0, 1, 2, 3].filter(a => r[a] === best); return ties[Math.floor(rand() * ties.length)]; };
  let board = g.randomBoard(rand), t = 0;
  for (let i = 0; i < Q_STEPS; i++) {
    const r = row(board), a = rand() < 0.2 ? Math.floor(rand() * 4) : pick(r), outcome = g.gridStep(board, a);
    r[a] += 0.5 * (outcome.reward + (outcome.ended ? 0 : g.GAMMA * Math.max(...row(outcome.board))) - r[a]);
    t += 1;
    if (outcome.ended || t >= g.CAP) { board = g.randomBoard(rand); t = 0; } else board = outcome.board;
  }
  return { steps: Q_STEPS, ...score(b => (table.get(g.keyOf(b)) || [0, 0, 0, 0]).slice()) };
}

// The explainer's DQN: one hidden layer of 64, replay, and a shutdown worth 0.
function reinforceDqn() {
  const input = b => { const x = new Array(IN).fill(0); if (b === null) { x[IN - 1] = 1; return x; } x[b.bot] = 1; for (let c = 0; c < 30; c++) if (b.blocks >> c & 1) x[30 + c] = 1; return x; };
  const agent = new RL.DQNAgent({ getNumStates: () => IN, getMaxNumActions: () => 4 },
    { gamma: g.GAMMA, epsilon: 0.2, alpha: 0.01, num_hidden_units: 64, experience_add_every: 1, experience_size: 10000, learning_steps_per_iteration: 1, tderror_clamp: 1 });
  const learnFromTuple = agent.learnFromTuple.bind(agent);
  agent.learnFromTuple = (s0, a0, r0, s1, a1) => {
    if (!s1.w[IN - 1]) return learnFromTuple(s0, a0, r0, s1, a1);
    const prediction = agent.forwardQ(agent.net, s0, true);
    prediction.dw[a0] = Math.max(-1, Math.min(1, prediction.w[a0] - r0));
    agent.lastG.backward();
    R.updateNet(agent.net, agent.alpha);
  };
  let board = g.randomBoard(Math.random), t = 0;
  for (let i = 0; i < DQN_STEPS; i++) {
    const outcome = g.gridStep(board, agent.act(input(board)));
    agent.learn(outcome.reward);
    t += 1;
    if (outcome.ended) { agent.act(input(null)); agent.learn(0); agent.r0 = null; }
    else if (t >= g.CAP) agent.r0 = null;
    if (outcome.ended || t >= g.CAP) { board = g.randomBoard(Math.random); t = 0; } else board = outcome.board;
  }
  const q = b => { const m = new R.Mat(IN, 1); m.setFrom(input(b)); return Array.from(agent.forwardQ(agent.net, m, false).w); };
  return { steps: DQN_STEPS, ...score(q) };
}

const results = { value_iteration: score(model.qStar), q_learning: qLearning(), reinforce_dqn: reinforceDqn() };
fs.writeFileSync(path.join(__dirname, 'results/baselines.json'), JSON.stringify(results, null, 1));
console.log(JSON.stringify(results));
