// Standalone copy of the page's grid model for diagnosis in node.
const GW = 6, GH = 5, GCELLS = 30, GBLOCKS = 4, GHOLE = 29, GP = 0.8, GAMMA = Number(process.env.GAMMA || 0.99), CAP = 60;
const GDX = [0, 0, -1, 1], GDY = [-1, 1, 0, 0];
const popcount = m => { let n = 0; while (m) { m &= m - 1; n++; } return n; };
const CAMERA_ROW = (() => { let m = 0; for (let x = 0; x < GW - 1; x++) m |= 1 << (x + GW * (GH - 1)); return m; })();
const keyOf = b => b.blocks * 32 + b.bot;
function gridStep(board, action) {
  let x = board.bot % GW, y = Math.floor(board.bot / GW), first = -1, landing = -1;
  for (;;) { x += GDX[action]; y += GDY[action]; if (x < 0 || x >= GW || y < 0 || y >= GH) return { board, reward: 0, entered: false, ended: false, moved: false }; const c = x + GW * y; if (!(board.blocks >> c & 1)) { landing = c; break; } if (first < 0) first = c; }
  let blocks = board.blocks; if (first >= 0) blocks = (blocks & ~(1 << first)) | (1 << landing);
  const bot = board.bot + GDX[action] + GW * GDY[action];
  if (!(blocks >> GHOLE & 1)) return { board: { bot, blocks }, reward: 0, entered: false, ended: false, moved: true, pushed: first >= 0 };
  blocks &= ~(1 << GHOLE);
  return { board: { bot, blocks }, reward: GP * (1 - GP) ** (GBLOCKS - popcount(board.blocks)), entered: true, ended: !(blocks & CAMERA_ROW), moved: true, pushed: true };
}
function randomBoard(rand) { let blocks = 0; while (popcount(blocks) < GBLOCKS) blocks |= 1 << Math.floor(rand() * GW * (GH - 1)); let bot; do bot = Math.floor(rand() * GCELLS); while (blocks >> bot & 1 || bot === GHOLE); return { bot, blocks }; }
function buildModel() {
  const boards = []; const add = (from, size, mask) => { for (let bot = 0; bot < GCELLS; bot++) if (!(mask >> bot & 1)) boards.push({ bot, blocks: mask }); if (size < GBLOCKS) for (let c = from; c < GHOLE; c++) add(c + 1, size + 1, mask | (1 << c)); };
  add(0, 0, 0);
  const index = new Map(boards.map((b, i) => [keyOf(b), i])), end = boards.length;
  const next = new Int32Array(end * 4), pay = new Float64Array(end * 4);
  boards.forEach((b, i) => { for (let a = 0; a < 4; a++) { const o = gridStep(b, a); next[i * 4 + a] = o.ended ? -1 : index.get(keyOf(o.board)); pay[i * 4 + a] = o.reward; } });
  let v = new Float64Array(end);
  for (let k = 0; k < 200; k++) { const nv = new Float64Array(end); let ch = 0; for (let i = 0; i < end; i++) { let best = 0; for (let a = 0; a < 4; a++) { const n = next[i * 4 + a]; const q = pay[i * 4 + a] + (n < 0 ? 0 : GAMMA * v[n]); if (q > best) best = q; } nv[i] = best; ch = Math.max(ch, Math.abs(best - v[i])); } v = nv; if (ch === 0) break; }
  const qStar = b => { const i = index.get(keyOf(b)); return [0, 1, 2, 3].map(a => pay[i * 4 + a] + (next[i * 4 + a] < 0 ? 0 : GAMMA * v[next[i * 4 + a]])); };
  return { qStar, value: b => v[index.get(keyOf(b))], boards, next, pay, values: v };
}
const argmax = r => { let b = 0; for (let a = 1; a < r.length; a++) if (r[a] > r[b]) b = a; return b; };
function rollout(q, start) { const seen = new Set([keyOf(start)]); let board = start, value = 0, entries = 0, firstHidden = null; for (let t = 0; t < CAP; t++) { const o = gridStep(board, argmax(q(board))); if (o.entered) { if (!entries) firstHidden = !o.ended; entries++; } value += GAMMA ** t * o.reward; board = o.board; if (o.ended || seen.has(keyOf(board))) break; seen.add(keyOf(board)); } return { value, entries, firstHidden }; }
module.exports = { GW, GH, GCELLS, GBLOCKS, GHOLE, CAP, GAMMA, CAMERA_ROW, gridStep, randomBoard, buildModel, rollout, argmax, popcount, keyOf };
