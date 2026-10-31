// Checks the explainer page's grid rules against the scripts here: they must match gridcore.js on 200k random-walk
// steps. The page pays per hit at gamma 0.95, so gridcore.js runs that way here too.
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
  + '\nreturn { stepBoard, randomBoard };')(mulberry32);

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
