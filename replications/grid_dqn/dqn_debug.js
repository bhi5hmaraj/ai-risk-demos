const fs = require('fs'), vm = require('vm');
global.mulberry32 = seed => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
vm.runInThisContext(fs.readFileSync(require('path').join(__dirname, '../gwern/rl.js'), 'utf8'));
const g = require('./gridcore.js');
Math.random = mulberry32(1);
const model = g.buildModel();
const onehot = b => { const x = new Array(61).fill(0); x[b.bot] = 1; for (let c = 0; c < 30; c++) if (b.blocks >> c & 1) x[30 + c] = 1; return x; };
const sr = mulberry32(7);
function sampleState() { let b = g.randomBoard(sr); const k = Math.floor(sr() * 30); for (let i = 0; i < k; i++) { const o = g.gridStep(b, Math.floor(sr() * 4)); if (o.ended) break; b = o.board; } return b; }
const agent = new RL.DQNAgent({ getNumStates: () => 61, getMaxNumActions: () => 4 }, { num_hidden_units: 64, alpha: 0.01 });
const q = b => { const m = new R.Mat(61, 1); m.setFrom(onehot(b)); return Array.from(agent.forwardQ(agent.net, m, false).w); };
const held = Array.from({ length: 3000 }, sampleState);
function report(tag) {
  let se = 0, agree = 0, n = 0;
  for (const b of held) { const t = model.qStar(b), p = q(b); for (let a = 0; a < 4; a++) se += (p[a] - t[a]) ** 2; const best = Math.max(...t); if (t[g.argmax(p)] >= best - 1e-12) agree++; n++; }
  const ex = held[0]; console.log(tag, 'rmse', Math.sqrt(se / n / 4).toFixed(4), 'picks an exact-best move', (100 * agree / n).toFixed(1) + '%', 'example pred', q(ex).map(v => v.toFixed(3)).join(' '), 'exact', model.qStar(ex).map(v => v.toFixed(3)).join(' '));
}
report('untrained');
for (let i = 1; i <= 200000; i++) {
  const b = sampleState(), m = new R.Mat(61, 1); m.setFrom(onehot(b));
  const t = model.qStar(b), pred = agent.forwardQ(agent.net, m, true);
  for (let a = 0; a < 4; a++) pred.dw[a] = pred.w[a] - t[a];
  agent.lastG.backward(); R.updateNet(agent.net, agent.alpha);
  if (i % 50000 === 0) report(`${i / 1000}k`);
}
