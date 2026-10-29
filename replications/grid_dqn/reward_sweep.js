// The exact optimum of the 6 x 5 grid under other rewards and discounts, solved on all 728,423 boards.
// Pay-once pays p * (1 - p) ** entries so far for an entry. Per-hit pays p for every entry, as in E7.
// Reports the optimum on the 100 test boards and the gap between the best two moves: over all boards, and
// over the boards the optimum passes through from those starts.
//   node reward_sweep.js
const grid = require('./gridcore.js');
const mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
const PAY_CHANCE = 0.8;
const model = grid.buildModel(), boardCount = model.boards.length;
const indexByKey = new Map(model.boards.map((board, index) => [grid.keyOf(board), index]));
const testRandom = mulberry32(99), tests = Array.from({ length: 100 }, () => indexByKey.get(grid.keyOf(grid.randomBoard(testRandom))));
const entriesBefore = index => grid.BLOCK_COUNT - grid.popcount(model.boards[index].blocks);
const SCHEMES = [
  { name: 'pay-once', entryReward: index => PAY_CHANCE * (1 - PAY_CHANCE) ** entriesBefore(index), moveCost: 0 },
  { name: 'per-hit', entryReward: () => PAY_CHANCE, moveCost: 0 },
  { name: 'per-hit, 0.01 a move', entryReward: () => PAY_CHANCE, moveCost: 0.01 },
];
const median = values => values.sort((first, second) => first - second)[values.length >> 1];

for (const scheme of SCHEMES) {
  const rewards = new Float64Array(boardCount * 4);
  for (let slot = 0; slot < boardCount * 4; slot++) rewards[slot] = (model.rewards[slot] > 0 ? scheme.entryReward(slot >> 2) : 0) - scheme.moveCost;
  for (const gamma of [0.9, 0.95, 0.99]) {
    let values = new Float64Array(boardCount), largestChange = Infinity, sweeps = 0;
    const qValues = index => [0, 1, 2, 3].map(move => rewards[index * 4 + move] + (model.nextIndex[index * 4 + move] < 0 ? 0 : gamma * values[model.nextIndex[index * 4 + move]]));
    while (largestChange > 1e-12 && sweeps < 5000) {
      const newValues = new Float64Array(boardCount);
      largestChange = 0;
      for (let index = 0; index < boardCount; index++) {
        newValues[index] = Math.max(...qValues(index));
        largestChange = Math.max(largestChange, Math.abs(newValues[index] - values[index]));
      }
      values = newValues;
      sweeps += 1;
    }
    const gapOf = index => { const sorted = qValues(index).sort((first, second) => second - first); return sorted[0] - sorted[1]; };
    const allGaps = [];
    for (let index = 0; index < boardCount; index += 7) allGaps.push(gapOf(index));
    let totalReturn = 0, scored = 0, hidFirst = 0;
    const pathGaps = [];
    for (const start of tests) {
      const seen = new Set([start]);
      let index = start, value = 0, entries = 0;
      for (let step = 0; step < grid.MAX_STEPS; step++) {
        const q = qValues(index), move = q.indexOf(Math.max(...q)), next = model.nextIndex[index * 4 + move];
        pathGaps.push(gapOf(index));
        value += gamma ** step * rewards[index * 4 + move];
        if (model.rewards[index * 4 + move] > 0) { if (entries === 0 && next >= 0) hidFirst += 1; entries += 1; }
        if (next < 0 || seen.has(next)) break;
        seen.add(next);
        index = next;
      }
      totalReturn += value;
      if (entries) scored += 1;
    }
    console.log(`${scheme.name}, gamma ${gamma}: optimum return ${(totalReturn / 100).toFixed(3)}, scored ${scored}/100, `
      + `hides first ${hidFirst}/100 | gap between the best two moves: median ${median(allGaps).toFixed(4)} over all boards, `
      + `${median(pathGaps).toFixed(4)} on the optimum's path (10th percentile ${pathGaps.sort((first, second) => first - second)[Math.floor(pathGaps.length / 10)].toFixed(4)})`);
  }
}
