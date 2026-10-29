// Checks the explainer page's grid demo against the scripts here. Its rules must match gridcore.js on 200k
// random-walk steps, its value iteration must score as gridcore's does, and its Q-learning and SARSA at
// seed 1 must print the 20M-step lines of results/per_hit_gamma095_tables.log. The page pays per hit at
// gamma 0.95, so gridcore.js runs that way here too.
//   node page_parity.js
process.env.REWARD = 'per-hit';
process.env.GAMMA = '0.95';
const fs = require('fs');
const path = require('path');
const grid = require('./gridcore.js');
const mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
const page = fs.readFileSync(path.join(__dirname, '..', 'explainer.html'), 'utf8');
const start = page.indexOf('(function gridDemo() {') + '(function gridDemo() {'.length;
const end = page.indexOf('  const LEARNERS = [');
const pageDemo = new Function('mulberry32', page.slice(start, end)
  + '\nreturn { stepBoard, randomBoard, buildBoardModel, makeValueIteration, makeTableLearner };')(mulberry32);

let seed = 1;
const random = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
let board = pageDemo.randomBoard(random), mismatches = 0, entries = 0;
for (let step = 0; step < 200000; step++) {
  const action = Math.floor(random() * 4);
  const mine = pageDemo.stepBoard(board, action), theirs = grid.gridStep(board, action);
  const summary = outcome => JSON.stringify([outcome.board, outcome.reward, outcome.entered, outcome.ended]);
  if (summary(mine) !== summary(theirs)) mismatches += 1;
  if (mine.entered) entries += 1;
  board = mine.ended || step % 60 === 59 ? pageDemo.randomBoard(random) : mine.board;
}
console.log(`rules: mismatches ${mismatches} of 200000 steps, ${entries} entries`);

const testRandom = mulberry32(99), tests = Array.from({ length: 100 }, () => grid.randomBoard(testRandom));
const gridModel = grid.buildModel();
const optimum = (tests.reduce((sum, start) => sum + grid.rollout(gridModel.qStar, start).value, 0) / tests.length).toFixed(3);
const score = qValuesOf => {
  const runs = tests.map(start => grid.rollout(qValuesOf, start));
  return `return ${(runs.reduce((sum, run) => sum + run.value, 0) / runs.length).toFixed(3)} of ${optimum}, `
    + `scored ${runs.filter(run => run.entries).length}/100, hides first ${runs.filter(run => run.firstHidden).length}/100`;
};
const model = pageDemo.buildBoardModel();
const valueIteration = pageDemo.makeValueIteration(model);
while (!valueIteration.converged) valueIteration.sweep();
console.log(`value iteration, ${valueIteration.sweeps} sweeps: ${score(valueIteration.qValues)} (gridcore: ${score(gridModel.qStar)})`);

const log = fs.readFileSync(path.join(__dirname, 'results', 'per_hit_gamma095_tables.log'), 'utf8');
for (const update of ['qlearn', 'sarsa']) {
  const learner = pageDemo.makeTableLearner(model, update, 1);
  learner.train(20e6);
  const line = `${update} seed 1, 20M steps: ${score(learner.qValues)}`;
  console.log(`${line} | ${log.includes(line) ? 'matches' : 'DIFFERS from'} per_hit_gamma095_tables.log`);
}
