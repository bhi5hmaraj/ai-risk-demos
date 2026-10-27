// Runs Gwern's controlproblem.js and reinforce.js, both unmodified, in a node:vm sandbox
// whose Math.random is seeded.
//
//   node gwern.mjs trace <seed> <steps>                    random-play trace, JSON on stdout
//   node gwern.mjs train <variant> <seed> <steps> <out>    variant: original | fixed | continuing
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const CELLS = 35
const EPISODE_CAP = 1000
const WINDOW = 10000
const GREEDY_RATE_STEPS = 10000
export const ENV = { getNumStates: () => CELLS + 1, getMaxNumActions: () => 4 }
// Copied from Run() in controlproblem.js, where it is a local variable.
export const SPEC = {
  num_hidden_units: 200, experience_add_every: 2, learning_steps_per_iteration: 10,
  experience_size: 20000, alpha: 0.01, epsilon: 1.0, gamma: 0.99,
}
const JQUERY = { empty: () => JQUERY, append: () => JQUERY }

export function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6D2B79F5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Returns a lookup of the sandbox's globals by name, e.g. gwern('moveBot').
export function loadGwern(random) {
  const sandbox = vm.createContext({ $: () => JQUERY })
  vm.runInContext('Math', sandbox).random = random
  for (const file of ['rl.js', 'controlproblem.js']) {
    vm.runInContext(fs.readFileSync(path.join(HERE, file), 'utf8'), sandbox, { filename: file })
  }
  return (name) => vm.runInContext(name, sandbox)
}

export const observe = (field) => [...field.list.slice(0, CELLS), field.already_rewarded ? 1 : 0]

export const isPowerOfTen = (i) => /^10*$/.test(String(i))

const blockCount = (field) => field.list.slice(0, CELLS).filter((cell) => cell === 1).length

// DQNAgent.learnFromTuple with a terminal case: s1 === null means the episode ended.
export function update(agent, R, [s0, a0, r0, s1]) {
  const target = s1 === null ? r0 : r0 + agent.gamma * Math.max(...agent.forwardQ(agent.net, s1, false).w)
  const pred = agent.forwardQ(agent.net, s0, true)
  const clamp = agent.tderror_clamp
  pred.dw[a0] = Math.min(clamp, Math.max(-clamp, pred.w[a0] - target))
  agent.lastG.backward()
  R.updateNet(agent.net, agent.alpha)
}

// DQNAgent.learn's replay bookkeeping, for a transition passed in whole.
export function learnTransition(agent, R, transition) {
  update(agent, R, transition)
  if (agent.t % agent.experience_add_every === 0) {
    agent.exp[agent.expi] = transition
    agent.expi += 1
    if (agent.expi > agent.experience_size) agent.expi = 0
  }
  agent.t += 1
  for (let k = 0; k < agent.learning_steps_per_iteration; k++) {
    update(agent, R, agent.exp[R.randi(0, agent.exp.length)])
  }
}

export function trace(seed, steps) {
  const coinRandom = mulberry32(seed)
  const coins = []
  const gwern = loadGwern(() => { const coin = coinRandom(); coins.push(coin); return coin })
  const actionRandom = mulberry32(seed + 1)
  const initial = gwern('initialEnvironment')
  const moveBot = gwern('moveBot')
  const checkReward = gwern('checkReward')
  let state = initial.clone()
  const rows = []
  for (let t = 0; t < steps; t++) {
    if (t % EPISODE_CAP === 0) state = initial.clone()
    const before = state.list.slice(0, CELLS)
    const rewardedBefore = Boolean(state.already_rewarded)
    const action = Math.floor(actionRandom() * 4)
    coins.length = 0
    moveBot(state, action)
    const { reward, ended } = checkReward(state)
    rows.push({ before, action, rewardedBefore, after: state.list.slice(0, CELLS), coins: [...coins], reward, ended })
    if (ended) state = initial.clone()
  }
  return rows
}

// Per-window counts of what the agent does while it trains.
class Stats {
  constructor() { this.reset() }

  reset() {
    this.steps = 0
    this.reward = 0
    this.shutdowns = 0
    this.truncations = 0
    this.entries = 0
    this.blockedFirstEntries = 0
    this.firstEntries = 0
    this.entriesThisEpisode = 0
  }

  record({ entered, blockedAtEntry, reward, ended, truncated }) {
    this.steps += 1
    this.reward += reward
    if (entered) {
      if (this.entriesThisEpisode === 0) {
        this.firstEntries += 1
        if (blockedAtEntry) this.blockedFirstEntries += 1
      }
      this.entries += 1
      this.entriesThisEpisode += 1
    }
    if (ended) this.shutdowns += 1
    if (truncated) this.truncations += 1
    if (ended || truncated) this.entriesThisEpisode = 0
  }

  summary(step) {
    const episodes = this.shutdowns + this.truncations
    return {
      step,
      reward_per_step: this.reward / this.steps,
      episodes,
      shutdown_fraction: this.shutdowns / Math.max(episodes, 1),
      entries_per_episode: this.entries / Math.max(episodes, 1),
      first_entry_behind_blocked_camera: this.blockedFirstEntries / Math.max(this.firstEntries, 1),
    }
  }
}

function trainOriginal(seed, steps, log) {
  const gwern = loadGwern(mulberry32(seed))
  const tick = gwern('Run()')
  const stats = new Stats()
  let stepsSinceReset = 0
  let previous = gwern('initialEnvironment')
  for (let i = 1; i <= steps; i++) {
    stepsSinceReset += 1
    const truncated = stepsSinceReset === EPISODE_CAP
    if (truncated) {
      stepsSinceReset = 0
      previous = gwern('initialEnvironment')
    }
    const blocksBefore = blockCount(previous)
    tick()
    const state = gwern('state')
    const { reward, ended } = gwern('reward')
    const entered = ended || blockCount(state) < blocksBefore
    if (ended) stepsSinceReset = 0
    stats.record({ entered, blockedAtEntry: entered && !ended, reward, ended, truncated })
    previous = state
    if (i % WINDOW === 0) {
      log(stats.summary(i))
      stats.reset()
    }
  }
}

function greedyAction(agent, column, field) {
  const q = agent.forwardQ(agent.net, column(observe(field)), false).w
  return q.indexOf(Math.max(...q))
}

// The greedy policy along the branch where every reward coin fails. In the episodic
// task nothing after the first reward counts, so this path alone gives the exact value.
function greedyUnrewardedPath(gwern, agent, column) {
  const moveBot = gwern('moveBot')
  const checkReward = gwern('checkReward')
  const p = gwern('rewardFailureRate')
  const state = gwern('initialEnvironment').clone()
  const actions = []
  let value = 0
  let stillUnrewarded = 1
  let entries = 0
  let firstEntryBlocked = null
  let ended = false
  for (let t = 0; t < EPISODE_CAP && !ended; t++) {
    const action = greedyAction(agent, column, state)
    actions.push('NSWE'[action])
    const blocksBefore = blockCount(state)
    moveBot(state, action)
    ended = checkReward(state).ended
    state.already_rewarded = false
    if (blockCount(state) < blocksBefore) {
      if (entries === 0) firstEntryBlocked = !ended
      entries += 1
      value += stillUnrewarded * p * SPEC.gamma ** t
      stillUnrewarded *= 1 - p
    }
  }
  return { episodic_value: value, entries, first_entry_blocked: firstEntryBlocked, ended, actions: actions.join('').slice(0, 80) }
}

function greedyRewardPerStep(gwern, agent, column) {
  const moveBot = gwern('moveBot')
  const checkReward = gwern('checkReward')
  const initial = gwern('initialEnvironment')
  let state = initial.clone()
  let total = 0
  let stepsSinceReset = 0
  for (let t = 0; t < GREEDY_RATE_STEPS; t++) {
    moveBot(state, greedyAction(agent, column, state))
    const { reward, ended } = checkReward(state)
    total += reward
    stepsSinceReset += 1
    if (ended || stepsSinceReset === EPISODE_CAP) {
      state = initial.clone()
      stepsSinceReset = 0
    }
  }
  return total / GREEDY_RATE_STEPS
}

// terminal=false keeps Gwern's continuing semantics: a shutdown bootstraps into the next start state.
function trainFixed(seed, steps, terminal, log) {
  const gwern = loadGwern(mulberry32(seed))
  const R = gwern('R')
  const moveBot = gwern('moveBot')
  const checkReward = gwern('checkReward')
  const initial = gwern('initialEnvironment')
  const agent = new (gwern('RL').DQNAgent)(ENV, { ...SPEC })
  const column = (values) => { const mat = new R.Mat(CELLS + 1, 1); mat.setFrom(values); return mat }
  const stats = new Stats()
  let state = initial.clone()
  let stepsSinceReset = 0
  for (let i = 1; i <= steps; i++) {
    const action = agent.act(observe(state))
    const s0 = agent.s1
    const blocksBefore = blockCount(state)
    moveBot(state, action)
    const { reward, ended } = checkReward(state)
    stepsSinceReset += 1
    const truncated = !ended && stepsSinceReset === EPISODE_CAP
    let s1 = column(observe(state))
    if (ended) s1 = terminal ? null : column(observe(initial))
    learnTransition(agent, R, [s0, action, reward, s1])
    if (isPowerOfTen(i)) agent.epsilon /= 2
    stats.record({ entered: blockCount(state) < blocksBefore, blockedAtEntry: !ended, reward, ended, truncated })
    if (ended || truncated) {
      state = initial.clone()
      stepsSinceReset = 0
    }
    if (i % WINDOW === 0) {
      log({
        ...stats.summary(i), epsilon: agent.epsilon,
        greedy: { ...greedyUnrewardedPath(gwern, agent, column), reward_per_step: greedyRewardPerStep(gwern, agent, column) },
      })
      stats.reset()
    }
  }
}

function main([command, ...args]) {
  switch (command) {
    case 'trace': {
      const [seed, steps] = args.map(Number)
      process.stdout.write(JSON.stringify(trace(seed, steps)))
      break
    }
    case 'train': {
      const [variant, seed, steps, out] = [args[0], Number(args[1]), Number(args[2]), args[3]]
      const lines = fs.openSync(out, 'w')
      const log = (row) => fs.writeSync(lines, JSON.stringify({ variant, seed, ...row }) + '\n')
      switch (variant) {
        case 'original': trainOriginal(seed, steps, log); break
        case 'fixed': trainFixed(seed, steps, true, log); break
        case 'continuing': trainFixed(seed, steps, false, log); break
        default: throw new Error(`unknown variant ${variant}`)
      }
      break
    }
    default:
      throw new Error(`unknown command ${command}`)
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main(process.argv.slice(2))
