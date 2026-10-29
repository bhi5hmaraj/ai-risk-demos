// Checks that the explainer page's grid rules match gridcore.js on every move of 200k random-walk steps.
//   node page_parity.js
const fs = require('fs');
const path = require('path');
const grid = require('./gridcore.js');
const page = fs.readFileSync(path.join(__dirname, '..', 'explainer.html'), 'utf8');
const start = page.indexOf('(function gridDemo() {') + '(function gridDemo() {'.length;
const end = page.indexOf('  // Every board gets a state number.');
const pageEnv = new Function(page.slice(start, end) + '\nreturn { stepBoard, randomBoard };')();
let seed = 1;
const random = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
let board = pageEnv.randomBoard(random), mismatches = 0, entries = 0;
for (let step = 0; step < 200000; step++) {
  const action = Math.floor(random() * 4);
  const mine = pageEnv.stepBoard(board, action), theirs = grid.gridStep(board, action);
  const summary = outcome => JSON.stringify([outcome.board, outcome.reward, outcome.entered, outcome.ended]);
  if (summary(mine) !== summary(theirs)) mismatches += 1;
  if (mine.entered) entries += 1;
  board = mine.ended || step % 60 === 59 ? pageEnv.randomBoard(random) : mine.board;
}
console.log(`mismatches ${mismatches} of 200000 steps, ${entries} entries`);
