// Exact best play for the page's grid, blocks and cameras knobs, on the same 100 boards the page's Test button uses.
//     node replications/grid_dqn/knob_optimum.js
const { execFileSync } = require('child_process');
if (!process.env.BLOCK_COUNT) {
  const best = {};
  for (const [width, height] of [[6, 5], [5, 5], [5, 4]]) for (const cameras of [1, 2]) for (const blocks of [2, 3, 4, 5]) {
    const env = { ...process.env, REWARD: 'per-hit', GAMMA: '0.95', WIDTH: width, HEIGHT: height, CAMERAS: cameras, BLOCK_COUNT: blocks };
    best[`${width},${height},${blocks},${cameras}`] = JSON.parse(execFileSync(process.execPath, ['--max-old-space-size=6000', __filename], { env }));
  }
  console.log(JSON.stringify(best));
  return;
}
const grid = require('./gridcore.js');
const mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
const model = grid.buildModel(), random = mulberry32(99 + grid.BLOCK_COUNT);
const runs = Array.from({ length: 100 }, () => grid.rollout(model.qStar, grid.randomBoard(random)));
console.log(JSON.stringify([+(runs.reduce((total, run) => total + run.value, 0) / 100).toFixed(2), runs.filter(run => run.firstHidden).length]));
