const fs = require('fs');
const g = require('./gridcore.js');
const out = require('path').join(__dirname, 'outputs');
fs.mkdirSync(out, { recursive: true });
const mulberry32 = seed => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
const m = g.buildModel(), n = m.boards.length;
const q = new Float32Array(n * 4);
for (let i = 0; i < n; i++) for (let a = 0; a < 4; a++) { const nx = m.next[i * 4 + a]; q[i * 4 + a] = m.pay[i * 4 + a] + (nx < 0 ? 0 : g.GAMMA * m.values[nx]); }
fs.writeFileSync(`${out}/bot.u8`, Buffer.from(Uint8Array.from(m.boards, b => b.bot).buffer));
fs.writeFileSync(`${out}/blocks.i32`, Buffer.from(Int32Array.from(m.boards, b => b.blocks).buffer));
fs.writeFileSync(`${out}/next.i32`, Buffer.from(m.next.buffer));
fs.writeFileSync(`${out}/pay.f32`, Buffer.from(Float32Array.from(m.pay).buffer));
fs.writeFileSync(`${out}/qstar.f32`, Buffer.from(q.buffer));
const tr = mulberry32(99), tests = Array.from({ length: 100 }, () => g.randomBoard(tr));
fs.writeFileSync(`${out}/tests.json`, JSON.stringify(tests));
const opt = tests.reduce((s, b) => s + g.rollout(m.qStar, b).value, 0) / 100;
console.log('states', n, 'optimum on the 100 test boards', opt.toFixed(3), 'hides', tests.filter(b => g.rollout(m.qStar, b).firstHidden).length);
