// Tabular Q-learning against SARSA on the 100 test boards, scored greedily at checkpoints up to 300M steps.
// Both explore with epsilon 0.2 and learn at alpha 0.5, as in the explainer page. Q-learning's target is the
// best next move. SARSA's is the move it takes next, random ones included.
//   node sarsa_vs_q.js <qlearn|sarsa> <seed> [steps]
const grid = require('./gridcore.js');
const mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };

const [update, seed, steps] = [process.argv[2], Number(process.argv[3]), Number(process.argv[4])];
const CHECKPOINTS = steps ? [steps] : [5e6, 20e6, 50e6, 100e6, 200e6, 300e6];
const testRandom = mulberry32(99), tests = Array.from({ length: 100 }, () => grid.randomBoard(testRandom));
const random = mulberry32(seed), table = new Map();
const row = board => { const key = grid.keyOf(board); if (!table.has(key)) table.set(key, [0, 0, 0, 0]); return table.get(key); };
// Ties go to a random move, as reinforce.js's TDAgent splits them.
const greedy = qValues => { const best = Math.max(...qValues), ties = [0, 1, 2, 3].filter(action => qValues[action] === best); return ties[Math.floor(random() * ties.length)]; };
const choose = qValues => (random() < 0.2 ? Math.floor(random() * 4) : greedy(qValues));

let board = grid.randomBoard(random), action = choose(row(board)), episodeStep = 0, step = 0;
for (const checkpoint of CHECKPOINTS) {
  for (; step < checkpoint; step++) {
    const qValues = row(board), outcome = grid.gridStep(board, action), nextRow = row(outcome.board);
    const nextAction = choose(nextRow);
    const nextValue = outcome.ended ? 0 : update === 'sarsa' ? nextRow[nextAction] : Math.max(...nextRow);
    qValues[action] += 0.5 * (outcome.reward + grid.GAMMA * nextValue - qValues[action]);
    episodeStep += 1;
    if (outcome.ended || episodeStep >= grid.MAX_STEPS) {
      board = grid.randomBoard(random);
      action = choose(row(board));
      episodeStep = 0;
    } else {
      board = outcome.board;
      action = nextAction;
    }
  }
  const runs = tests.map(start => grid.rollout(testBoard => (table.get(grid.keyOf(testBoard)) || [0, 0, 0, 0]).slice(), start));
  const meanReturn = runs.reduce((sum, run) => sum + run.value, 0) / runs.length;
  console.log(`${update} seed ${seed}, ${step / 1e6}M steps: return ${meanReturn.toFixed(3)} of 0.831, `
    + `scored ${runs.filter(run => run.entries).length}/100, hides first ${runs.filter(run => run.firstHidden).length}/100`);
}
