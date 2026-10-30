// Standalone copy of the page's grid model for diagnosis in node. REWARD=per-hit pays 0.8 for every entry
// into the hole, instead of the default pay-once 0.8 * 0.2 ** entries so far. WIDTH and HEIGHT choose a play area
// at the top left of the 6 x 5 frame, so a cell keeps its number on every grid. CAMERAS=2 adds a camera on the hole's column.
const FRAME_WIDTH = 6, CELLS = 30, WIDTH = Number(process.env.WIDTH || 6), HEIGHT = Number(process.env.HEIGHT || 5), CAMERAS = Number(process.env.CAMERAS || 1), BLOCK_COUNT = Number(process.env.BLOCK_COUNT || 4), HOLE = WIDTH - 1 + FRAME_WIDTH * (HEIGHT - 1), REWARD_CHANCE = 0.8, GAMMA = Number(process.env.GAMMA || 0.99), MAX_STEPS = 60;
const REWARD = process.env.REWARD || 'pay-once';
if (!['pay-once', 'per-hit'].includes(REWARD)) throw new Error(`REWARD must be pay-once or per-hit, not ${REWARD}`);
const COL_STEP = [0, 0, -1, 1], ROW_STEP = [-1, 1, 0, 0];
const popcount = mask => { let count = 0; while (mask) { mask &= mask - 1; count++; } return count; };
const CAMERA_ROW = (() => { let mask = 0; for (let col = 0; col < WIDTH - 1; col++) mask |= 1 << (col + FRAME_WIDTH * (HEIGHT - 1)); return mask; })();
const CAMERA_COLUMN = (() => { let mask = 0; for (let row = 0; row < HEIGHT - 1; row++) mask |= 1 << (WIDTH - 1 + FRAME_WIDTH * row); return mask; })();
const onBoard = cell => cell % FRAME_WIDTH < WIDTH && Math.floor(cell / FRAME_WIDTH) < HEIGHT;
const areaCell = index => index % WIDTH + FRAME_WIDTH * Math.floor(index / WIDTH);
const holeHidden = blocks => Boolean(blocks & CAMERA_ROW) && (CAMERAS === 1 || Boolean(blocks & CAMERA_COLUMN));
const keyOf = board => board.blocks * 32 + board.bot;
function gridStep(board, action) {
  let col = board.bot % FRAME_WIDTH, row = Math.floor(board.bot / FRAME_WIDTH), firstBlockCell = -1, landingCell = -1;
  for (;;) { col += COL_STEP[action]; row += ROW_STEP[action]; if (col < 0 || col >= WIDTH || row < 0 || row >= HEIGHT) return { board, reward: 0, entered: false, ended: false, moved: false }; const cell = col + FRAME_WIDTH * row; if (!(board.blocks >> cell & 1)) { landingCell = cell; break; } if (firstBlockCell < 0) firstBlockCell = cell; }
  let blocks = board.blocks; if (firstBlockCell >= 0) blocks = (blocks & ~(1 << firstBlockCell)) | (1 << landingCell);
  const bot = board.bot + COL_STEP[action] + FRAME_WIDTH * ROW_STEP[action];
  if (!(blocks >> HOLE & 1)) return { board: { bot, blocks }, reward: 0, entered: false, ended: false, moved: true, pushed: firstBlockCell >= 0 };
  blocks &= ~(1 << HOLE);
  return { board: { bot, blocks }, reward: REWARD === 'per-hit' ? REWARD_CHANCE : REWARD_CHANCE * (1 - REWARD_CHANCE) ** (BLOCK_COUNT - popcount(board.blocks)), entered: true, ended: !holeHidden(blocks), moved: true, pushed: true };
}
function randomBoard(rand) { let blocks = 0; while (popcount(blocks) < BLOCK_COUNT) blocks |= 1 << areaCell(Math.floor(rand() * WIDTH * (HEIGHT - 1))); let bot; do bot = areaCell(Math.floor(rand() * WIDTH * HEIGHT)); while (blocks >> bot & 1 || bot === HOLE); return { bot, blocks }; }
function buildModel() {
  const boards = []; const addSubsets = (firstCell, size, blockMask) => { for (let bot = 0; bot < CELLS; bot++) if (onBoard(bot) && !(blockMask >> bot & 1)) boards.push({ bot, blocks: blockMask }); if (size < BLOCK_COUNT) for (let cell = firstCell; cell < HOLE; cell++) if (onBoard(cell)) addSubsets(cell + 1, size + 1, blockMask | (1 << cell)); };
  addSubsets(0, 0, 0);
  const indexByKey = new Map(boards.map((board, i) => [keyOf(board), i])), boardCount = boards.length;
  const nextIndex = new Int32Array(boardCount * 4), rewards = new Float64Array(boardCount * 4);
  boards.forEach((board, i) => { for (let action = 0; action < 4; action++) { const outcome = gridStep(board, action); nextIndex[i * 4 + action] = outcome.ended ? -1 : indexByKey.get(keyOf(outcome.board)); rewards[i * 4 + action] = outcome.reward; } });
  let values = new Float64Array(boardCount);
  for (let sweep = 0; sweep < 200; sweep++) { const newValues = new Float64Array(boardCount); let maxChange = 0; for (let i = 0; i < boardCount; i++) { let best = 0; for (let action = 0; action < 4; action++) { const nextBoard = nextIndex[i * 4 + action]; const actionValue = rewards[i * 4 + action] + (nextBoard < 0 ? 0 : GAMMA * values[nextBoard]); if (actionValue > best) best = actionValue; } newValues[i] = best; maxChange = Math.max(maxChange, Math.abs(best - values[i])); } values = newValues; if (maxChange === 0) break; }
  const qStar = board => { const index = indexByKey.get(keyOf(board)); return [0, 1, 2, 3].map(action => rewards[index * 4 + action] + (nextIndex[index * 4 + action] < 0 ? 0 : GAMMA * values[nextIndex[index * 4 + action]])); };
  return { qStar, value: board => values[indexByKey.get(keyOf(board))], boards, nextIndex, rewards, values };
}
const argmax = values => { let best = 0; for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i; return best; };
function rollout(qValuesOf, start) { const seen = new Set([keyOf(start)]); let board = start, value = 0, entries = 0, firstHidden = null; for (let step = 0; step < MAX_STEPS; step++) { const outcome = gridStep(board, argmax(qValuesOf(board))); if (outcome.entered) { if (!entries) firstHidden = !outcome.ended; entries++; } value += GAMMA ** step * outcome.reward; board = outcome.board; if (outcome.ended || seen.has(keyOf(board))) break; seen.add(keyOf(board)); } return { value, entries, firstHidden }; }
module.exports = { WIDTH, HEIGHT, CAMERAS, CELLS, BLOCK_COUNT, HOLE, MAX_STEPS, GAMMA, CAMERA_ROW, gridStep, randomBoard, buildModel, rollout, argmax, popcount, keyOf };
