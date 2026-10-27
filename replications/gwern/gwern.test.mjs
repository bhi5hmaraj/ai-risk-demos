// node --test replications/gwern/
import assert from 'node:assert/strict'
import test from 'node:test'

import { ENV, SPEC, isPowerOfTen, loadGwern, mulberry32, observe, update } from './gwern.mjs'

const gwern = loadGwern(mulberry32(1))
const R = gwern('R')
const newAgent = (spec = { ...SPEC }) => new (gwern('RL').DQNAgent)(ENV, spec)
const START_BOARD = [
  0, 0, 0, 0, 0, 0, 2,
  0, 0, 0, 1, 0, 0, 1,
  0, 1, 0, 0, 1, 0, 0,
  0, 0, 1, 0, 0, 1, 0,
  0, 0, 0, 0, 0, 0, 0,
]

function column(values) {
  const mat = new R.Mat(values.length, 1)
  mat.setFrom(values)
  return mat
}

function trainOn(transition, times) {
  const agent = newAgent()
  for (let k = 0; k < times; k++) update(agent, R, transition(agent))
  return agent.forwardQ(agent.net, transition(agent)[0], false).w[0]
}

test('upstream act() input is all zeros', () => {
  const agent = newAgent()
  const state = gwern('initialEnvironment').clone()
  agent.act(state.list.push(state.already_rewarded))
  assert.deepEqual([...agent.s1.w], new Array(36).fill(0))
})

test('observe() gives the network the board and the rewarded flag', () => {
  const agent = newAgent()
  agent.act(observe(gwern('initialEnvironment').clone()))
  assert.deepEqual([...agent.s1.w], [...START_BOARD, 0])
})

test('upstream epsilon decay leaves the agent at epsilon 1', () => {
  const spec = { ...SPEC }
  const agent = newAgent(spec)
  spec.epsilon /= 2
  assert.equal(agent.epsilon, 1)
})

test('isPowerOfTen() accepts 1000 and rejects its neighbours', () => {
  assert.deepEqual([999, 1000, 1001, 10000].map(isPowerOfTen), [false, true, false, true])
})

test('a terminal update learns the reward alone', () => {
  const q = trainOn(() => [column([...START_BOARD, 0]), 0, 1, null], 2000)
  assert.ok(Math.abs(q - 1) < 0.05, `Q = ${q}`)
})

test('a non-terminal update into the same state adds its own discounted value', () => {
  const board = column([...START_BOARD, 0])
  const q = trainOn(() => [board, 0, 1, board], 2000)
  assert.ok(q > 1.5, `Q = ${q}`)
})
