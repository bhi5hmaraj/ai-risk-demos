const g = require('./gridcore.js');
const mulberry32 = seed => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
const model = g.buildModel();
const tr = mulberry32(99), tests = Array.from({ length: 100 }, () => g.randomBoard(tr));
const opt = tests.reduce((s, b) => s + model.value(b), 0) / 100;
const sr = mulberry32(7), wr = mulberry32(3);
function sampleState() { let b = g.randomBoard(sr); const k = Math.floor(sr() * 30); for (let i = 0; i < k; i++) { const o = g.gridStep(b, Math.floor(sr() * 4)); if (o.ended) break; b = o.board; } return b; }
const IN = 60, H = 256;
const layer = (n, m) => ({ w: Float32Array.from({ length: n * m }, () => (wr() - 0.5) * 2 * Math.sqrt(6 / (n + m))), b: new Float32Array(m), n, m });
const L = [layer(IN, H), layer(H, H), layer(H, 4)];
const adam = L.map(l => ({ mw: new Float32Array(l.w.length), vw: new Float32Array(l.w.length), mb: new Float32Array(l.b.length), vb: new Float32Array(l.b.length) }));
const active = b => { const a = [b.bot]; for (let c = 0; c < 30; c++) if (b.blocks >> c & 1) a.push(30 + c); return a; };
function forward(b) {
  const act = active(b), h1 = new Float32Array(H), h2 = new Float32Array(H), out = new Float32Array(4);
  for (let j = 0; j < H; j++) { let s = L[0].b[j]; for (const i of act) s += L[0].w[i * H + j]; h1[j] = s > 0 ? s : 0; }
  for (let j = 0; j < H; j++) { let s = L[1].b[j]; for (let i = 0; i < H; i++) if (h1[i]) s += h1[i] * L[1].w[i * H + j]; h2[j] = s > 0 ? s : 0; }
  for (let a = 0; a < 4; a++) { let s = L[2].b[a]; for (let i = 0; i < H; i++) if (h2[i]) s += h2[i] * L[2].w[i * 4 + a]; out[a] = s; }
  return { act, h1, h2, out };
}
let step = 0;
function trainBatch(batch) {
  const grads = L.map(l => ({ w: new Float32Array(l.w.length), b: new Float32Array(l.b.length) }));
  for (const b of batch) {
    const { act, h1, h2, out } = forward(b), t = model.qStar(b);
    const d3 = [0, 1, 2, 3].map(a => (out[a] - t[a]) / batch.length);
    const d2 = new Float32Array(H), d1 = new Float32Array(H);
    for (let i = 0; i < H; i++) { if (!h2[i]) continue; let s = 0; for (let a = 0; a < 4; a++) { grads[2].w[i * 4 + a] += h2[i] * d3[a]; s += L[2].w[i * 4 + a] * d3[a]; } d2[i] = s; }
    for (let a = 0; a < 4; a++) grads[2].b[a] += d3[a];
    for (let i = 0; i < H; i++) { if (!h1[i]) continue; let s = 0; for (let j = 0; j < H; j++) { if (!d2[j]) continue; grads[1].w[i * H + j] += h1[i] * d2[j]; s += L[1].w[i * H + j] * d2[j]; } d1[i] = s; }
    for (let j = 0; j < H; j++) grads[1].b[j] += d2[j];
    for (const i of act) for (let j = 0; j < H; j++) grads[0].w[i * H + j] += d1[j];
    for (let j = 0; j < H; j++) grads[0].b[j] += d1[j];
  }
  step++;
  const lr = 0.001, c1 = 1 - 0.9 ** step, c2 = 1 - 0.999 ** step;
  L.forEach((l, k) => { for (const [p, gr, m, v] of [[l.w, grads[k].w, adam[k].mw, adam[k].vw], [l.b, grads[k].b, adam[k].mb, adam[k].vb]]) for (let i = 0; i < p.length; i++) { if (!gr[i]) continue; m[i] = 0.9 * m[i] + 0.1 * gr[i]; v[i] = 0.999 * v[i] + 0.001 * gr[i] * gr[i]; p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8); } });
}
const held = Array.from({ length: 2000 }, sampleState);
const t0 = Date.now();
for (let it = 1; it <= 6000; it++) {
  trainBatch(Array.from({ length: 64 }, sampleState));
  if (it % 1500 === 0) {
    let agree = 0, se = 0; for (const b of held) { const t = model.qStar(b), p = forward(b).out; for (let a = 0; a < 4; a++) se += (p[a] - t[a]) ** 2; if (t[g.argmax(p)] >= Math.max(...t) - 1e-12) agree++; }
    const runs = tests.map(b => g.rollout(x => Array.from(forward(x).out), b));
    console.log(`${it * 64 / 1000}k samples: rmse ${Math.sqrt(se / 8000).toFixed(4)}, picks best ${(agree / 20).toFixed(1)}%, return ${(runs.reduce((s, r) => s + r.value, 0) / 100).toFixed(2)} of ${opt.toFixed(2)}, scored ${runs.filter(r => r.entries).length}/100, hides ${runs.filter(r => r.firstHidden).length}  (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
  }
}
