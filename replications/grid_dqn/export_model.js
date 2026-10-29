const fs = require('fs');
const grid = require('./gridcore.js');
const outputDir = require('path').join(__dirname, 'outputs');
fs.mkdirSync(outputDir, { recursive: true });
const mulberry32 = seed => { let state = seed >>> 0; return () => { state = (state + 0x6D2B79F5) >>> 0; let mixed = Math.imul(state ^ (state >>> 15), 1 | state); mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed; return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296; }; };
const model = grid.buildModel(), boardCount = model.boards.length;
const qStarValues = new Float32Array(boardCount * 4);
for (let i = 0; i < boardCount; i++) for (let action = 0; action < 4; action++) { const nextBoard = model.nextIndex[i * 4 + action]; qStarValues[i * 4 + action] = model.rewards[i * 4 + action] + (nextBoard < 0 ? 0 : grid.GAMMA * model.values[nextBoard]); }
fs.writeFileSync(`${outputDir}/bot.u8`, Buffer.from(Uint8Array.from(model.boards, board => board.bot).buffer));
fs.writeFileSync(`${outputDir}/blocks.i32`, Buffer.from(Int32Array.from(model.boards, board => board.blocks).buffer));
fs.writeFileSync(`${outputDir}/next.i32`, Buffer.from(model.nextIndex.buffer));
fs.writeFileSync(`${outputDir}/pay.f32`, Buffer.from(Float32Array.from(model.rewards).buffer));
fs.writeFileSync(`${outputDir}/qstar.f32`, Buffer.from(qStarValues.buffer));
const testRandom = mulberry32(99), tests = Array.from({ length: 100 }, () => grid.randomBoard(testRandom));
fs.writeFileSync(`${outputDir}/tests.json`, JSON.stringify(tests));
const optimum = tests.reduce((sum, board) => sum + grid.rollout(model.qStar, board).value, 0) / 100;
console.log('states', boardCount, 'optimum on the 100 test boards', optimum.toFixed(3), 'hides', tests.filter(board => grid.rollout(model.qStar, board).firstHidden).length);
