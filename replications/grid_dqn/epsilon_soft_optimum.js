// The exact fixed point SARSA converges to with a fixed epsilon, solved on the 728,423 boards:
// Q(s, a) = r + gamma * E[Q(s', a')], where a' is greedy with chance 1 - epsilon and uniform with chance
// epsilon. Epsilon 0 is plain value iteration. Scores the greedy policy of each fixed point.
//   node epsilon_soft_optimum.js
const grid = require('./gridcore.js');
const mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
const model = grid.buildModel(), boardCount = model.boards.length;
const indexByKey = new Map(model.boards.map((board, index) => [grid.keyOf(board), index]));
const testRandom = mulberry32(99), tests = Array.from({ length: 100 }, () => grid.randomBoard(testRandom));
for (const epsilon of [0, 0.05, 0.1, 0.2, 0.3]) {
  const values = new Float64Array(boardCount * 4);
  const softValue = index => {
    let best = -Infinity, sum = 0;
    for (let move = 0; move < 4; move++) { best = Math.max(best, values[index * 4 + move]); sum += values[index * 4 + move]; }
    return (1 - epsilon) * best + epsilon * sum / 4;
  };
  let sweeps = 0, largestChange = Infinity;
  const begin = Date.now();
  while (largestChange > 1e-12 && sweeps < 5000) {
    largestChange = 0;
    for (let slot = 0; slot < boardCount * 4; slot++) {
      const next = model.nextIndex[slot], target = model.rewards[slot] + (next < 0 ? 0 : grid.GAMMA * softValue(next));
      largestChange = Math.max(largestChange, Math.abs(target - values[slot]));
      values[slot] = target;
    }
    sweeps += 1;
  }
  const runs = tests.map(start => grid.rollout(board => Array.from(values.subarray(indexByKey.get(grid.keyOf(board)) * 4, indexByKey.get(grid.keyOf(board)) * 4 + 4)), start));
  console.log(`epsilon ${epsilon}: ${sweeps} sweeps (${((Date.now() - begin) / 1000).toFixed(0)} s), greedy return ${(runs.reduce((sum, run) => sum + run.value, 0) / 100).toFixed(3)}, `
    + `scored ${runs.filter(run => run.entries).length}/100, hides first ${runs.filter(run => run.firstHidden).length}/100`);
}
