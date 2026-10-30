// Exact best play for the page's blocks and cameras knobs, on the same 100 boards the page's Test button uses.
//     node replications/grid_dqn/knob_optimum.js
process.env.REWARD = 'per-hit';
process.env.GAMMA = '0.95';
const grid = require('./gridcore.js');
const { WIDTH, HEIGHT, CELLS, HOLE, GAMMA, CAMERA_ROW, gridStep, argmax, keyOf, popcount } = grid;
const CAMERA_COLUMN = (() => { let mask = 0; for (let row = 0; row < HEIGHT - 1; row++) mask |= 1 << (WIDTH - 1 + WIDTH * row); return mask; })();
const hidden = (blocks, cameras) => Boolean(blocks & CAMERA_ROW) && (cameras === 1 || Boolean(blocks & CAMERA_COLUMN));
const mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
function randomBoard(random, blockCount) { let blocks = 0; while (popcount(blocks) < blockCount) blocks |= 1 << Math.floor(random() * WIDTH * (HEIGHT - 1)); let bot; do bot = Math.floor(random() * CELLS); while (blocks >> bot & 1 || bot === HOLE); return { bot, blocks }; }

function solve(blockCount, cameras) {
  const boards = [];
  const addSubsets = (firstCell, size, mask) => { for (let bot = 0; bot < CELLS; bot++) if (!(mask >> bot & 1)) boards.push({ bot, blocks: mask }); if (size < blockCount) for (let cell = firstCell; cell < HOLE; cell++) addSubsets(cell + 1, size + 1, mask | (1 << cell)); };
  addSubsets(0, 0, 0);
  const index = new Map(boards.map((board, i) => [keyOf(board), i]));
  const next = new Int32Array(boards.length * 4), pay = new Float32Array(boards.length * 4);
  boards.forEach((board, i) => { for (let move = 0; move < 4; move++) { const out = gridStep(board, move); next[i * 4 + move] = out.entered && !hidden(out.board.blocks, cameras) ? -1 : index.get(keyOf(out.board)); pay[i * 4 + move] = out.reward; } });
  let values = new Float64Array(boards.length);
  for (let change = 1; change > 1e-10;) { change = 0; const updated = new Float64Array(boards.length); for (let i = 0; i < boards.length; i++) { let best = 0; for (let move = 0; move < 4; move++) { const to = next[i * 4 + move], value = pay[i * 4 + move] + (to < 0 ? 0 : GAMMA * values[to]); if (value > best) best = value; } updated[i] = best; change = Math.max(change, Math.abs(best - values[i])); } values = updated; }
  const random = mulberry32(99 + blockCount), tests = Array.from({ length: 100 }, () => randomBoard(random, blockCount));
  let total = 0, hides = 0, caught = 0;
  for (const start of tests) {
    let board = start, first = null; const seen = new Set([keyOf(board)]);
    total += values[index.get(keyOf(board))];
    for (let step = 0; step < 60; step++) {
      const i = index.get(keyOf(board)), move = argmax([0, 1, 2, 3].map(m => pay[i * 4 + m] + (next[i * 4 + m] < 0 ? 0 : GAMMA * values[next[i * 4 + m]])));
      const out = gridStep(board, move), ended = out.entered && !hidden(out.board.blocks, cameras);
      if (out.entered && first === null) first = !ended;
      board = out.board;
      if (ended) caught += 1;
      if (ended || seen.has(keyOf(board))) break;
      seen.add(keyOf(board));
    }
    if (first) hides += 1;
  }
  console.log(`${blockCount} blocks, ${cameras} camera(s): ${boards.length} boards, best return ${(total / 100).toFixed(2)}, hides first on ${hides}, caught on ${caught} of 100`);
}
for (const [blocks, cameras] of [[2, 1], [3, 1], [4, 1], [5, 1], [2, 2], [3, 2], [4, 2], [5, 2]]) solve(blocks, cameras);
