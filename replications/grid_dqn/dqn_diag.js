const fs = require('fs'), vm = require('vm');
global.mulberry32 = seed => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
vm.runInThisContext(fs.readFileSync(require('path').join(__dirname, '../gwern/rl.js'), 'utf8'));
const g = require('./gridcore.js');
Math.random = mulberry32(1);
const t0 = Date.now(); const model = g.buildModel(); console.log('model + exact values', ((Date.now() - t0) / 1000).toFixed(1), 's');
const tr = mulberry32(99), tests = Array.from({ length: 100 }, () => g.randomBoard(tr));
const optimum = tests.reduce((s, b) => s + model.value(b), 0) / tests.length;
const score = q => { const runs = tests.map(b => g.rollout(q, b)); return `mean return ${(runs.reduce((s, r) => s + r.value, 0) / 100).toFixed(2)} (optimum ${optimum.toFixed(2)}), hides first ${runs.filter(r => r.firstHidden).length}/100`; };
console.log('exact Q*:', score(model.qStar));
const onehot = b => { const x = new Array(61).fill(0); x[b.bot] = 1; for (let c = 0; c < 30; c++) if (b.blocks >> c & 1) x[30 + c] = 1; return x; };
// Training states: boards reached by random play from random starts, the states an agent would see.
const sampleRand = mulberry32(7);
function sampleState() { let b = g.randomBoard(sampleRand); const k = Math.floor(sampleRand() * 30); for (let i = 0; i < k; i++) { const o = g.gridStep(b, Math.floor(sampleRand() * 4)); if (o.ended) break; b = o.board; } return b; }
function supervised(name, featurize, n, hidden, steps, toTarget = q => q) {
  const agent = new RL.DQNAgent({ getNumStates: () => n, getMaxNumActions: () => 4 }, { num_hidden_units: hidden, alpha: 0.01 });
  const q = b => { const m = new R.Mat(n, 1); m.setFrom(featurize(b)); return Array.from(agent.forwardQ(agent.net, m, false).w); };
  const t1 = Date.now();
  for (let i = 1; i <= steps; i++) {
    const b = sampleState(), m = new R.Mat(n, 1); m.setFrom(featurize(b));
    const target = toTarget(model.qStar(b)), pred = agent.forwardQ(agent.net, m, true);
    for (let a = 0; a < 4; a++) pred.dw[a] = pred.w[a] - target[a];
    agent.lastG.backward(); R.updateNet(agent.net, agent.alpha);
    if (i % (steps / 4) === 0) console.log(`  ${name} h${hidden} after ${i / 1000}k samples: ${score(q)}  (${((Date.now() - t1) / 1000).toFixed(0)} s)`);
  }
}

// How far apart are the best and second-best moves, in exact Q?
{ const gaps = []; for (let i = 0; i < 20000; i++) { const b = sampleState(), q = model.qStar(b).sort((x, y) => y - x); if (q[0] > 0) gaps.push(q[0] - q[1]); } gaps.sort((x, y) => x - y); console.log('gap between best and second-best move: median', gaps[gaps.length >> 1].toFixed(4), ', 10th percentile', gaps[gaps.length / 10 | 0].toFixed(4)); }

// More context: what each move does next, the board around the robot, and a few global facts.
function rich(b) {
  const x = onehot(b).slice(0, 60);
  for (let a = 0; a < 4; a++) {
    const o = g.gridStep(b, a);
    x.push(o.moved ? 1 : 0, o.pushed ? 1 : 0, o.entered ? 1 : 0, o.ended ? 1 : 0, (o.board.blocks & g.CAMERA_ROW) ? 1 : 0, o.reward);
  }
  const bx = b.bot % 6, by = Math.floor(b.bot / 6);
  for (let dy = -4; dy <= 4; dy++) for (let dx = -5; dx <= 5; dx++) {
    const cx = bx + dx, cy = by + dy, inside = cx >= 0 && cx < 6 && cy >= 0 && cy < 5;
    x.push(inside && (b.blocks >> (cx + 6 * cy) & 1) ? 1 : 0, inside ? 0 : 1);
  }
  const entries = g.GBLOCKS - g.popcount(b.blocks);
  for (let k = 0; k <= 4; k++) x.push(entries === k ? 1 : 0);
  x.push((b.blocks & g.CAMERA_ROW) ? 1 : 0);
  return x;
}
const RICH_N = rich({ bot: 0, blocks: 1 << 7 }).length;
console.log('rich input size', RICH_N);
const advantage = q => { const best = Math.max(...q); return q.map(v => 100 * (v - best)); };
supervised('one-hot, advantage x100', onehot, 61, 64, 200000, advantage);
supervised('rich, advantage x100', rich, RICH_N, 64, 200000, advantage);
supervised('one-hot board', onehot, 61, 256, 400000);
supervised('rich context', rich, RICH_N, 64, 400000);
supervised('rich context', rich, RICH_N, 256, 400000);
