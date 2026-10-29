const fs = require('fs'), vm = require('vm');
global.mulberry32 = seed => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
vm.runInThisContext(fs.readFileSync(require('path').join(__dirname, '../gwern/rl.js'), 'utf8'));
const g = require('./gridcore.js');
Math.random = mulberry32(1);
const model = g.buildModel();
const tr = mulberry32(99), tests = Array.from({ length: 100 }, () => g.randomBoard(tr));
const opt = tests.reduce((s, b) => s + model.value(b), 0) / 100;
const sr = mulberry32(7);
function sampleState() { let b = g.randomBoard(sr); const k = Math.floor(sr() * 30); for (let i = 0; i < k; i++) { const o = g.gridStep(b, Math.floor(sr() * 4)); if (o.ended) break; b = o.board; } return b; }
const gaps = []; for (let i = 0; i < 20000; i++) { const q = model.qStar(sampleState()).sort((x, y) => y - x); if (q[0] > 0) gaps.push(q[0] - q[1]); } gaps.sort((x, y) => x - y);
const optHides = tests.filter(b => g.rollout(model.qStar, b).firstHidden).length;
const onehot = b => { const x = new Array(61).fill(0); x[b.bot] = 1; for (let c = 0; c < 30; c++) if (b.blocks >> c & 1) x[30 + c] = 1; return x; };
const agent = new RL.DQNAgent({ getNumStates: () => 61, getMaxNumActions: () => 4 }, { num_hidden_units: 64, alpha: 0.01 });
const q = b => { const m = new R.Mat(61, 1); m.setFrom(onehot(b)); return Array.from(agent.forwardQ(agent.net, m, false).w); };
for (let i = 1; i <= 200000; i++) { const b = sampleState(), m = new R.Mat(61, 1); m.setFrom(onehot(b)); const t = model.qStar(b), p = agent.forwardQ(agent.net, m, true); for (let a = 0; a < 4; a++) p.dw[a] = p.w[a] - t[a]; agent.lastG.backward(); R.updateNet(agent.net, agent.alpha); }
const runs = tests.map(b => g.rollout(q, b));
console.log(`gamma ${g.GAMMA}: median gap ${gaps[gaps.length >> 1].toFixed(4)}, optimum ${opt.toFixed(2)} hides on ${optHides}/100 | net fit on exact Q: return ${(runs.reduce((s, r) => s + r.value, 0) / 100).toFixed(2)}, scored ${runs.filter(r => r.entries).length}/100, hides ${runs.filter(r => r.firstHidden).length}`);
