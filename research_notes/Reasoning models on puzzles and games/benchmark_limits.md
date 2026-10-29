# Limits of reasoning language models on puzzle, planning and game benchmarks (notes as of 29 Oct 2026)

*Labels used below: "primary" means the paper or the benchmark owner's own site. "Secondary" means an aggregator, news story, mirror or social post. "Search excerpt" means I saw the claim only in a search-result excerpt and did not read the full source. Leaderboard snapshots name the date I read them.*

## 1. "The Illusion of Thinking", the rebuttals, and later analyses

### Takeaway
The Apple paper by Shojaee and colleagues (NeurIPS 2025) found three regimes and a complete accuracy collapse past puzzle-specific thresholds. Examples are Tower of Hanoi at N=7-8 disks and River Crossing at N=3. Thinking tokens also fell near the collapse, even though budget remained.

The rebuttals showed that part of the result was an artifact. Some River Crossing instances (N>=6 with boat capacity 3) cannot be solved. Full move lists press against output limits, and the grading could not tell truncation from failure. Replications and 2026 mechanistic work still find a real failure in long, exact, stateful execution. The models "build a world model, and then lose it" during the chain of thought.

### Cited Findings
**Original paper**
- arXiv 2506.06941, by Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio and Mehrdad Farajtabar (Apple). v1 was posted 7 Jun 2025. v3, the NeurIPS 2025 camera-ready version with extra appendix discussion, was posted 20 Nov 2025 - [arXiv abs](https://arxiv.org/abs/2506.06941)
- The puzzles are Tower of Hanoi, Checker Jumping, River Crossing and Blocks World, each with controllable complexity. The models are Claude 3.7 Sonnet with and without thinking, DeepSeek-R1 vs V3, o3-mini (medium/high) and DeepSeek-R1-Distill-Qwen-32B, with QwQ-32B vs Qwen2.5-32B in the appendix. Generation was capped at 64k tokens, with 25 samples per instance per complexity level - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)
- The three regimes, comparing reasoning and standard models at equal inference compute: (1) At low complexity, standard models do better. (2) At medium complexity, reasoning models have the edge. (3) At high complexity, both collapse. The authors write that LRMs "face a complete accuracy collapse beyond certain complexities" - [arXiv abs](https://arxiv.org/abs/2506.06941)
- Minimum moves ("compositional depth") are 2^N-1 for Hanoi, so N=8 needs 255 moves, and (N+1)^2-1 for Checker Jumping, so N=4 needs 24 moves. River Crossing at N=3 needs 11 moves - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)
- Collapse points: Hanoi at N=7-8. Checker Jumping at about N=4. River Crossing "mostly collapses from N=3". For QwQ-32B: Hanoi N=7, Checker Jumping N=2, Blocks World N=10, River Crossing N=3 - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)
- Failures come early in the move sequence. At N=8 (255 moves), the first error typically appears around move 40-50. At N=10 (~10^3 moves), it appears around move 10^2, about 10% of the way in. Failure is non-monotonic: some runs failed before move 50 at N=12 but got past move 100 at N=10 - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)
- Claude 3.7 Sonnet (thinking) makes up to ~100 correct Hanoi moves at N=10. On River Crossing at N=3 it stays valid only until about move 4 - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)
- Reasoning effort: thinking tokens rise with complexity, then fall past a model-specific threshold close to the accuracy collapse, even though the models are well below their length limits. The drop is largest for o3-mini and smallest for Claude 3.7 Sonnet (thinking) - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)
- Giving the solution algorithm in the prompt did not help on Hanoi or Checker Jumping. Collapse came at about the same N. The authors say the models "fail to use explicit algorithms and reason inconsistently across scales" - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3), [arXiv abs](https://arxiv.org/abs/2506.06941)
- Overthinking at low complexity: models often find the right answer early and then keep exploring wrong alternatives. On Hanoi N=1-4, accuracy declines or oscillates as thinking goes on - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)
- Camera-ready responses to critics (Appendix A.1):
  - The first failures fall well inside the context limit.
  - A temperature-0 ablation on DeepSeek-R1 still collapses. Hanoi collapses at N=8 at T=0. With sampling it keeps 18.2% at N=8 and collapses at N=9. Blocks World collapses at N=4 at T=0. With sampling it keeps 44.1% at N=4 and collapses at N=30.
  - River Crossing with N>=6 was dropped because it changes the problem structure (it needs boat capacity k=4).
  - N>12 was removed from Hanoi.
  - Allowing tools would not show reasoning, because the puzzles' algorithms are well known.
  - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)

**Rebuttals**
- "Comment on The Illusion of Thinking" by A. Lawsen (arXiv 2506.09250). v1 was posted 10 Jun 2025. v2, on 16 Jun 2025, removed Claude as co-author to follow arXiv policy and corrected sections 4 and 6. Its claims:
  - Hanoi outputs risk exceeding output-token limits, and models said so in their outputs.
  - The automated grading cannot tell reasoning failures from practical limits.
  - River Crossing is "mathematically impossible for N > 5" because the boat is too small, yet models were scored as failing.
  - When asked for a generating function instead of a move list, models reach high accuracy on Hanoi sizes that had been reported as complete failures.
  - [arXiv abs](https://arxiv.org/abs/2506.09250)
- Per search excerpts of the same comment, not checked against the full text: models wrote a working recursive Lua solution for 15 disks in under 5,000 tokens (small samples). The comment also argues that compositional depth (minimum move count) conflates mechanical execution with problem-solving difficulty - [arXiv PDF](https://arxiv.org/pdf/2506.09250)
- Opinion (blog): Sean Goedecke argues that Tower of Hanoi is a worse reasoning test than math or code. He suggests that past a certain size, models look for shortcuts rather than tracing every move, and that the token drop reflects models declining to continue, not proven inability - [seangoedecke.com](https://www.seangoedecke.com/illusion-of-thinking/)

**Later analyses (2025-2026)**
- Varela, Romero-Sorozabal, Rocon and Cebrian, "Rethinking the Illusion of Thinking" (arXiv 2507.01231, 1 Jul 2025):
  - With incremental stepwise prompting and agentic collaborative dialogue, LRMs still struggle on Hanoi around 8 disks. Earlier failures were therefore partly cognitive, not only output limits.
  - The River Crossing "catastrophic failures" came from unsolvable configurations. Restricted to solvable ones, LRMs solved instances with more than 100 agent pairs.
  - Their summary: today's LRMs are "stochastic, RL-tuned searchers in a discrete state space we barely understand."
  - [arXiv abs](https://arxiv.org/abs/2507.01231)
- Pereira and Zuidema, "Transformers Struggle to Use Their Emergent World Models: Revisiting the Tower of Hanoi, and the Illusion of Thinking" (arXiv 2608.07077, 7 Aug 2026):
  - Small transformers trained on solution traces develop a linearly decodable, causally used state-space map, described as a Sierpinski triangle.
  - Qwen3.6-27B and DeepSeek-R1-Distill-Qwen-32B encode this map near-perfectly at the end of the prompt, yet fail most tasks with more than 3 rings.
  - The cause is decay of the representation over a long chain of thought. Injecting the prompt-time representation at inference partly recovers performance.
  - In the authors' words, models "build a world model, and then lose it."
  - [arXiv abs](https://arxiv.org/abs/2608.07077)
- Sinha et al., "The Illusion of Diminishing Returns" (ICLR 2026). They supply the plan and knowledge so that only execution is tested, and argue that long-task failures are execution errors, not inability to reason. Their numbers are in Section 4 - [arXiv abs](https://arxiv.org/abs/2509.09677)
- SokoBench (Jan 2026) did **not** reproduce the drop in effort: tokens rose with Sokoban corridor length, "unlike the declining effort reported by Shojaee et al." - [arXiv HTML](https://arxiv.org/html/2601.20856v1)

### Inferences
- The dispute has narrowed:
  - The River Crossing collapse at N>=6 was an artifact.
  - The Hanoi collapse at N=8 (255 moves, first error near move 40-50, inside a 64k budget) cannot be explained by output limits alone.
  - For larger N, the "can't vs won't enumerate" critique stands, and the generating-function results show that compressed representations of the solution work.
- What survives is a narrower claim. Reasoning models are unreliable at long exact execution, at tracking state across many steps, and at using an algorithm they are given. In 2026, mechanistic work locates this in representation decay during long chains of thought, not in a missing world model.
- The drop in thinking tokens near the collapse looks task- and model-dependent: o3-mini showed it, Sokoban did not. It should not be treated as a universal law.

### Gaps
- I did not extract exact thinking-token counts at the collapse points. They appear only in the paper's figures.
- I found no systematic 2026 re-run of the four Apple puzzles on the newest frontier models (GPT-6, Claude Opus 5/5.5, Gemini 4 class).
- I did not read the full Lawsen v2 text, so its token-count formula and its exact Lua results are unverified.

## 2. Planning: PlanBench, Mystery Blocksworld, o1 and later reasoning models, Sokoban and gridworlds

### Takeaway
In 2024, o1-preview jumped to 97.8% on Blocksworld. It reached only 52.8% on Mystery Blocksworld (obfuscated names) and 37.3% on the randomized version. It scored 23.63% on 20-40-step plans and correctly flagged just 27% of unsolvable instances.

By 2025-26, frontier LLMs rival classical planners on International Planning Competition tasks: Gemini 3.1 Pro solved 245 of 360 vs 234 for the best planner. Obfuscation still lowers scores. Sokoban corridor tests show steady degradation beyond about 25 moves, even for GPT-5-class reasoning models and GPT-oss-120B.

### Cited Findings
**PlanBench and o1 (Valmeekam, Stechly, Kambhampati, arXiv 2409.13373, 20 Sep 2024)**
- Blocksworld (600 instances, 3-5 blocks), zero-shot:
  - o1-preview: 97.8% (587/600)
  - o1-mini: 56.6%
  - Best LLM, LLaMA 3.1 405B: 62.6%
  - Fast Downward (classical planner): 100%
  - [arXiv HTML](https://arxiv.org/html/2409.13373v1)
- Mystery Blocksworld (obfuscated), zero-shot:
  - o1-preview: 52.8% (317/600), one-shot 41.6%
  - o1-mini: 19.1%
  - LLaMA 3.1 405B: 0.8% (5/600), and no LLM exceeded 5%
  - Randomized Mystery Blocksworld, o1-preview: 37.3%
  - [arXiv HTML](https://arxiv.org/html/2409.13373v1)
- Longer problems (110 instances, 6-20 blocks, 20-40-step plans): o1-preview 23.63% - [arXiv HTML](https://arxiv.org/html/2409.13373v1)
- Unsolvable instances (100 modified):
  - Blocksworld: 27% correctly called unsolvable, 0% false "impossible" claims.
  - Randomized Mystery Blocksworld: 16% correct, 11.5% false negatives.
  - [arXiv HTML](https://arxiv.org/html/2409.13373v1)
- Cost and time:
  - Per 100 instances: o1-preview $42.12, o1-mini $3.69, LLMs $0.02-$1.80.
  - Mean time per instance: o1-preview 40.43 s (Blocksworld), 83.37 s (Mystery), 111.11 s (Randomized Mystery). Fast Downward: 0.265 s.
  - Reasoning tokens plateau. They track optimal plan length on Mystery Blocksworld but not on Blocksworld.
  - The authors stress the lack of correctness guarantees and recommend LLM-Modulo setups or classical planners.
  - [arXiv HTML](https://arxiv.org/html/2409.13373v1)

**Later reasoning models on PlanBench-style tasks**
- DeepSeek-R1 on the PlanBench GitHub leaderboard: 99.1% Blocksworld, 43.3% Mystery, 25.8% Randomized Mystery, 53.6% Blocksworld Hard (PDDL). This is a community-maintained repo leaderboard, seen only in a search excerpt - [PlanBench repo](https://github.com/harshakokel/PlanBench)
- A 2026 representation study reports DeepSeek-R1-Distill-Qwen-32B at 0.08 accuracy on Mystery Blocksworld vs 0.81 on regular Blocksworld (search excerpt) - [arXiv 2602.04843](https://arxiv.org/pdf/2602.04843)
- Correa, Pereira and Seipp, "Frontier Large Language Models Rival State-of-the-Art Planners" (arXiv 2511.09378, v1 12 Nov 2025, v2 15 May 2026):
  - On 360 tasks from the latest International Planning Competition, Gemini 3.1 Pro solved 245 vs 234 for the strongest planner baseline. GPT-5 was comparable to the baselines. GPT-3.5 solved 0.
  - On obfuscated (purely symbolic) descriptions, all models drop, but Gemini 3.1 Pro stays competitive.
  - [arXiv abs](https://arxiv.org/abs/2511.09378)
  - v1 evaluated DeepSeek R1, Gemini 2.5 Pro and GPT-5. It found that obfuscation degrades all of them, "though less severely than previously reported" (search excerpt) - [arXiv HTML v1](https://arxiv.org/html/2511.09378v1)

**Sokoban and gridworld benchmarks**
- SokoBench (Monti, Nicolini, Pellegrini, Staiano and Lepri, arXiv 2601.20856, 28 Jan 2026, listed at TMLR 2026):
  - Setup: corridors of width L = 5-100 in steps of 5, with one player, one box and one goal in a line. Four orientations give 80 maps. Scoring is exact match to the optimal solution.
  - Models: DeepSeek-R1, GPT-5 and GPT-5-mini, and GPT-oss-120B at default (medium) effort, with a 32,768-token cap.
  - [arXiv HTML](https://arxiv.org/html/2601.20856v1), [TMLR listing](https://mlanthology.org/tmlr/2026/monti2026tmlr-sokobench/)
- SokoBench results:
  - "Consistent degradation" when solutions need more than 25 moves. Models struggle when they must look "more than 25-30 moves ahead."
  - Three regions per model: a plateau, a fast decline, then near-total failure.
  - The main error is counting. The authors model success as (1-p)^L.
  - Models loop, repeating actions until they hit the token cap ("wandering" without a set of visited states).
  - Some outputs make invalid moves through walls or the box.
  - Vertical maps, with many newlines, are harder than horizontal ones.
  - Tokens rise with corridor length, and long corridors often hit the 32,768 cap.
  - Slope of tokens vs L for correct answers (L<50): R1 51.1, GPT-5-mini 29.8, GPT-oss-120B 39.4.
  - [arXiv HTML](https://arxiv.org/html/2601.20856v1)
- SokoBench with PDDL tools ("LLM-Modulo"): only GPT-5-mini both wrote accurate PDDL and used the tools correctly. The gain was "modest" and variance high. The typical failure was valid PDDL that did not match the map. The authors infer a "fundamental constraint on forward planning capacity" that test-time scaling alone may not fix - [arXiv HTML](https://arxiv.org/html/2601.20856v1), [arXiv abs](https://arxiv.org/abs/2601.20856)
- LMGame-Bench (May 2025): without a game harness, most of 13 models score near random on Sokoban. With the harness, o3 and o1 rank top. RL on Sokoban and Tetris transfers to Blocksworld and WebShop (search excerpt) - [arXiv](https://arxiv.org/html/2505.15146v1)
- An end-to-end PDDL-agent paper evaluates only the hardest Sokoban corridors (L = 90-100). It says agents already do very well on shorter ones (search excerpt) - [arXiv 2512.09629](https://arxiv.org/pdf/2512.09629)

### Inferences
- Between 2024 and 2026, semantic Blocksworld-style planning went from unsolved to solved. Frontier models now match classical planners on IPC coverage.
- Two things remain hard: symbolic or obfuscated problems where world knowledge cannot be used, and long horizons. Plan length, not board size, is the axis that predicts failure.
- SokoBench's per-step error model (1-p)^L is the same compounding argument as Sinha et al.'s horizon formula (Section 4). Small per-step slips, such as counting errors, set the horizon limit.
- Spotting unsolvability is a separate weak skill (27% / 16% for o1). Tests should include unsolvable controls.

### Gaps
- I found no PlanBench-author evaluation of o3, GPT-5, GPT-6 or Claude Opus 5 on Mystery Blocksworld.
- SokoBench's accuracy at each length appears only in figures.
- I found no LLM gridworld benchmark with an adversarial or monitoring element (for example a watcher or camera) beyond general game suites.

## 3. Puzzle and game benchmarks: Sudoku-Bench, Enigmata, ARC-AGI-1/2/3, ARC Prize, chess and game arenas

### Takeaway
- ARC-AGI-1 and ARC-AGI-2 are close to saturated. On the ARC Prize results page (read 29 Oct 2026), the top scores are 98.5% on ARC-AGI-1 and 95.0% on ARC-AGI-2 (GPT-6). In Dec 2025 the best verified commercial model scored 37.6% on ARC-AGI-2.
- ARC-AGI-3 (interactive games, launched 25 Mar 2026) started with every frontier model below 1%. By 3 Sep 2026, GPT-6 Astra scored 62.7% under ARC Prize's Standard harness and 99.9% under the Provider Adapter harness. The harness, especially carrying reasoning state across turns, mattered more than reasoning effort.
- Sudoku variants stayed hard: GPT-5 solved 33% in late 2025.
- In head-to-head games, text chess strength is stratified: Gemini 3 Pro had a relative Elo of 1325 and Claude Opus 4.5 had 236. Model rankings do not carry over between chess, poker and Werewolf.

### Cited Findings
**Sudoku-Bench (Sakana AI)**
- Released May 2025 (arXiv 2505.16135). It has 100 puzzles: 15 4x4, 15 6x6 and 70 9x9, mostly modern variants. There are single-shot and multi-step modes. The metrics are Average Solve Rate and Average Correct Placements - [leaderboard page](https://pub.sakana.ai/sudoku/), [arXiv](https://arxiv.org/html/2505.16135v1)
- At launch, state-of-the-art LLMs solved under 15% unaided (search excerpt of the report) - [arXiv](https://arxiv.org/html/2505.16135v1)
- Conflicting launch details:
  - Sakana's GPT-5 post says no LLM could solve a classic 9x9 in May 2025.
  - A search excerpt says o3 was the only model to solve any 9x9 puzzles (2.9%, vanilla only).
  - [Sakana GPT-5 page](https://pub.sakana.ai/sudoku-gpt5/), [Sakana on X](https://x.com/SakanaAILabs/status/1988080410392404021)
- GPT-5 (high effort) solved 33% of challenge_100, about twice the previous leader (o3-mini). It was the first LLM to solve a 9x9 modern variant ("Theta"). Sakana's page dates this to Oct 2025 - [Sakana GPT-5 page](https://pub.sakana.ai/sudoku-gpt5/)
- Sakana's diagnosis:
  - Models grasp new rules and make locally consistent progress but lose global consistency over long chains.
  - They miss the creative "break-in" points that experts find.
  - Each variant first needs meta-reasoning about a new rule set.
  - GPT-5 is weak on spatial constraints and strong on algebraic ones.
  - GRPO on Qwen2.5-7B was "lackluster", and fine-tuning tended to produce surface pattern matching.
  - [Sakana GPT-5 page](https://pub.sakana.ai/sudoku-gpt5/)

**Enigmata (ByteDance Seed et al., arXiv 2505.19914, NeurIPS 2025)**
- It has 36 tasks in 7 categories, each with an unlimited generator of tunable difficulty and a rule-based verifier - [arXiv](https://arxiv.org/abs/2505.19914), [NeurIPS listing](https://mlanthology.org/neurips/2025/chen2025neurips-enigmata)
- Qwen2.5-32B-Enigmata beats o3-mini-high and o1 on Enigmata-Eval. It scores 32.8% on ARC-AGI-1 and 0.6% on ARC-AGI-2 (search excerpt). Puzzle data also improved Seed1.5-Thinking (200B total parameters) on AIME, BeyondAIME and GPQA - [arXiv](https://arxiv.org/abs/2505.19914)
- Secondary summaries:
  - AIME24: 60.6%, vs o3-mini-high 87.3% and o1 74.3%.
  - The project page says the model is still weak on the spatial and sequential task categories.
  - [project page](https://seed-enigmata.github.io), [Liner review](https://liner.com/review/enigmata-scaling-logical-reasoning-in-large-language-models-with-synthetic)

**ARC-AGI-1 / ARC-AGI-2 and ARC Prize 2025**
- ARC Prize 2025 Kaggle results on the ARC-AGI-2 private set (published 5 Dec 2025):
  1. NVARC, 24.03%: an improved ARChitects-style test-time-trained model plus parts based on the Tiny Recursive Model (TRM), with heavy synthetic data.
  2. The ARChitects, 16.53%: a masked-diffusion LLM with recursive self-refinement.
  3. MindsAI, 12.64%: test-time training with augmentation ensembles.
  4. Lonnie, 6.67%.
  5. G. Barbadillo, 6.53%.
  - The Grand Prize (85%) was not won.
  - [ARC Prize analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis), [technical report](https://arxiv.org/html/2601.10904v1)
  - An ARC Prize X post says 25.03% for NVARC, but the report and the leaderboard say 24.03% - [X](https://x.com/arcprize/status/1997010070585201068)
- Paper awards:
  1. Jolicoeur-Martineau's TRM: about 7M parameters, ~45% on ARC-AGI-1 and 8% on ARC-AGI-2.
  2. Pourcel, Colas and Oudeyer's SOAR: evolutionary program synthesis that fine-tunes the LLM on its own search traces. It reaches up to 52% on ARC-AGI-1.
  3. Liao and Gu's CompressARC: a minimum-description-length method with ~76K parameters, no pretraining. It scores ~20-34% on ARC-AGI-1 and ~4% on ARC-AGI-2.
  - The competition drew 1,455 teams and 90 papers. ARC Prize calls 2025 the year of the "refinement loop" (explore, verify, refine).
  - [ARC Prize analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis), [technical report](https://arxiv.org/html/2601.10904v1)
- Best commercial results in Dec 2025:
  - Claude Opus 4.5 (Thinking, 64k): 37.6% on ARC-AGI-2 at $2.20/task, the top verified commercial model.
  - Poetiq's refinement harness on Gemini 3 Pro: 54% at about $30/task.
  - ARC Prize warned that public training data may inflate ARC-AGI-1/2 scores through overfitting, and that reasoning performance is still tied to knowledge of the task domain.
  - [ARC Prize analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- TRM paper tables (Oct 2025), 2 tries, ARC-AGI-1 / ARC-AGI-2:
  - DeepSeek R1: 15.8 / 1.3
  - Claude 3.7 16K: 28.6 / 0.7
  - o3-mini-high: 34.5 / 3.0
  - Gemini 2.5 Pro 32K: 37.0 / 4.9
  - Grok-4-thinking: 66.7 / 16.0
  - Bespoke Grok-4: 79.6 / 29.4
  - HRM (27M): 40.3 / 5.0
  - TRM-Att (7M): 44.6 / 7.8
  - [arXiv HTML](https://arxiv.org/html/2510.04871v1)
- TRM paper, Sudoku-Extreme / Maze-Hard (1K training examples each):
  - DeepSeek R1, Claude 3.7 8K and o3-mini-high: 0.0 / 0.0
  - HRM: 55.0 / 74.5
  - TRM-Att: 74.7 / 85.3
  - TRM-MLP: 87.4 on Sudoku-Extreme
  - TRM uses one 2-layer network with 42 recursions per supervision step and up to 16 supervision steps.
  - [arXiv HTML](https://arxiv.org/html/2510.04871v1)
- ARC Prize results page, read 29 Oct 2026 (primary). ARC-AGI-1 / ARC-AGI-2:
  - GPT-6 (Astra/Sol/Luna, 2 Sep 2026): 98.5% / 95.0%
  - GPT-6.1 Sol (29 Sep): 98.5% / 94.2%
  - Claude Opus 5.5 (22 Sep): 98.5% / 93.3%
  - GPT-5.6 (9 Jul): 97.5% / 92.5%
  - Gemini 4 Argon (13 Oct): 98.5% / 91.3%
  - Claude Opus 5 (24 Jul): 97.5% / 90.4%
  - DeepSeek V4.1 Flash 1017 (17 Oct): 94.5% / 78.1%
  - [arcprize.org/results](https://arcprize.org/results)
- ARC Prize on X for GPT-6 Sol: ARC-AGI-2 89.6% at $0.44/task and ARC-AGI-1 95.5% at $0.14/task (search excerpt) - [X](https://x.com/arcprize/status/2104671274299429283)

**ARC-AGI-3 (interactive games)**
- Format: interactive environments with no instructions or stated goals. Agents must explore, infer the goal, build a world model and plan over long horizons with sparse feedback. ARC Prize says all environments are "100% human-solvable", and "100% means AI agents can beat every game as efficiently as humans" - [arcprize.org/arc-agi/3](https://arcprize.org/arc-agi/3/)
- Scoring, from secondary sources quoting the technical report:
  - The metric is Relative Human Action Efficiency (RHAE). Each level scores (human actions / AI actions)^2, capped at 1.0. For example, 10 vs 100 actions gives 1%.
  - Later levels are weighted more. Only actions that change the game state count.
  - The human baseline is the second-best of 10 first-time players.
  - Agent attempts are to be capped at 5x human attempts.
  - [The Decoder](https://the-decoder.com/arc-agi-3-offers-2m-to-any-ai-that-matches-untrained-humans-yet-every-frontier-model-scores-below-1/), [tech report PDF](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
  - ARC Prize's Astra post describes the baseline differently: about 500 untrained members of the public, with the per-level baseline the median action count among players who completed the level - [ARC Prize blog](https://arcprize.org/blog/astra)
- Launch, 25 Mar 2026 (secondary). Every frontier model scored below 1%:
  - Gemini 3.1 Pro: 0.37%
  - GPT-5.4 (High): 0.26%
  - Claude Opus 4.6: 0.25%
  - Grok 4.20: 0%, because it hit the action cutoff on every level
  - During the 2025 preview, a purpose-built agent reached 12.58%.
  - [OfficeChai](https://officechai.com/ai/arc-agi-3/), [DataCamp](https://www.datacamp.com/blog/arc-agi-3), [DEV](https://dev.to/codepawl/gpt-5-claude-gemini-all-score-below-1-arc-agi-3-just-broke-every-frontier-model-5dbj)
  - BenchLM (secondary, captured 28 Oct 2026) lists Gemini 3.1 Pro 0.4%, GPT-5.4 High 0.2%, Claude Opus 4.6 (max) 0.5% and Grok 4.20 0.1% - [BenchLM](https://benchlm.ai/benchmarks/arcagi3)
- 2026 progress on the ARC Prize results page (primary, though the page does not label the harness):
  - GPT-5.6: 7.78% (9 Jul)
  - Claude Opus 5: 30.16% (24 Jul)
  - Gemini 3.8 Flash: 35.00% (2 Sep)
  - GPT-6: 99.95% (2 Sep)
  - Grok 4.7: 10.05% (21 Sep)
  - Claude Sonnet 5.5: 72.44% (28 Sep)
  - GPT-6.1 Sol: 96.37% (29 Sep)
  - Claude Haiku 5.5: 2.82% (7 Oct)
  - Gemini 4 Argon: 97.46% (13 Oct)
  - [arcprize.org/results](https://arcprize.org/results)
- GPT-6 Astra on the ARC-AGI-3 semi-private set (ARC Prize blog, Greg Kamradt, 3 Sep 2026). Standard vs Provider Adapter, score and cost:

  | Effort | Standard | Provider Adapter |
  |---|---|---|
  | max | 62.7%, $26,098 | 98.6%, $17,332 |
  | xhigh | 59.3%, $37,317 | 98.4%, $18,147 |
  | high | 54.8%, $40,705 | 99.9%, $18,817 |
  | medium | 38.6%, $48,090 | 98.4%, $19,285 |
  | low | 17.5%, $38,166 | 98.0%, $21,298 |
  | none | 35.2%, $49,791 | 96.7%, $23,457 |

  Source: [ARC Prize blog](https://arcprize.org/blog/astra)
- Harness definitions:
  - **Standard** carries forward notes the model chooses to keep. It is minimal and provider-neutral.
  - **Provider Adapter** preserves the provider's opaque reasoning state between requests and compacts long conversations.
  - Across 167 game-effort pairs both harnesses solved, the Provider Adapter was ~3.66x faster and used 49% fewer tokens.
  - Astra (max, Provider Adapter) used fewer actions than the human baseline on 96.0% of levels, and 51.7% fewer actions per level on average. Humans cost about $12.78 per attempted game.
  - ARC Prize says this is not proof of AGI, because ARC-AGI-3 is deterministic and closed-ended. Its next work is on evaluating recursive self-improvement and open-ended innovation.
  - [ARC Prize blog](https://arcprize.org/blog/astra)
- Other models under both harnesses:
  - GPT-6 Sol: 4.6% Standard ($5.6K) vs 23.0% Provider Adapter ($8.7K). GPT-6 Luna: 0.59% (ARC Prize on X, search excerpt) - [X](https://x.com/arcprize/status/2104671274299429283)
  - BenchLM's Standard-harness figures (secondary, 28 Oct 2026): GPT-6.1 Sol (max) 52.7%, Gemini 4 Argon (low) 49.8%, Claude Opus 5 (high) 30.2%, Claude Sonnet 5.5 (xhigh) 24.3% - [BenchLM](https://benchlm.ai/benchmarks/arcagi3)
- ARC Prize community leaderboard (ARC-AGI-3 public demo set, self-reported, read 29 Oct 2026):
  - Tycho: 100.0%, $2,986, 29 Jul. Delegates planning to a world model tested by falsification.
  - Retrodict: 99.9%, $654, 19 Jul. Tests rule hypotheses against logged history before acting.
  - baseline1: 99.0%, $400, 15 Jul. Coding agent that builds and verifies a Python world model, then plans through it.
  - NOOA (NVIDIA): 85.1%, $332, 9 Jul. CodeAct agent with NumPy world-model helpers.
  - OPINE-World: 78.4%, 1 Jul. A CEGIS-style loop that rewrites a game engine.
  - Vision-Continual Learning v1: 63.1%, 18 May.
  - Read-Grep-Bash Agent: 50.2%, 13 Mar.
  - TELL: 43.9%. DreamTeam: 38.1% at $18,000. OpenClaw: 5.2%.
  - [community leaderboard](https://arcprize.org/leaderboard/community)
- ARC Prize 2026 has about $2M across three tracks. The ARC-AGI-3 track has $850K:
  - $700K Grand Prize for the first eligible agent to score 100% on the ARC-AGI-3 evaluation.
  - $75K Top Score awards.
  - $75K in Milestone prizes (30 Jun and 30 Sep 2026).
  - No internet during evaluation, and solutions must be open source.
  - [ARC-AGI-3 competition](https://arcprize.org/competitions/2026/arc-agi-3), [ARC Prize 2026](https://arcprize.org/competitions/2026)
  - Per search excerpts of the 2026 page: submissions are due 2 Nov 2026, papers 8 Nov, and results will be announced 4 Dec 2026 - [ARC Prize 2026](https://arcprize.org/competitions/2026)
- Milestone #1 winners (secondary): Tufa Labs first, Reki second, Md Boktiar Mahbub Murad third, announced about 6 Jul 2026 - [AlphaSignal](https://alphasignal.ai/news/tufa-labs-wins-25k-beating-frontier-ai-on-the-world-s-hardest-benchmark)
- Unverified Kaggle scores from a prediction-market page: Milestone #1 winner 1.21%, Milestone #2 high 27.9%, and an open-sourced harness at about 55.89% on the public board by 4 Oct - [Manifold](https://manifold.markets/Terminator2/arcagi3-kaggle-competition-what-wil)

**Chess and multi-game arenas**
- Game Arena paper (Google DeepMind and Kaggle, arXiv 2609.31473, 25 Sep 2026):
  - Chess: text harness given FEN and PGN, one SAN move per turn, no list of legal moves, retries on illegal moves, 40 colour-balanced games per pair.
  - Chess Elo, relative, with DeepSeek V3.2 fixed at 0 (not human Elo): Gemini 3 Pro 1325, Gemini 3 Flash 1297, o3 1009, GPT-5.2 933, Grok 4 773, GPT-5 mini 525, Claude Opus 4.5 236, Claude Sonnet 4.5 189, Claude Haiku 4.5 122.
  - Poker (bb/100): GPT-5.2 +46.6, o3 +29.7, Gemini 3 Pro -15.2, GPT-5 mini -94.9.
  - Werewolf: Gemini 3 Pro ranked top.
  - "model strength is sharply stratified within each game environment, but not consistent across games."
  - [arXiv HTML](https://arxiv.org/html/2609.31473v1)
- Game Arena chess failures:
  - Weaker models drift into bad middlegames, and their illegal-move rate rises as the game goes on. Ignoring a check is common.
  - Retries after illegal moves per game: Gemini under 0.2, o3 and Grok about 0.5-1.7, and DeepSeek V3.2 and Claude Haiku 4.5 often more than 5.
  - [arXiv HTML](https://arxiv.org/html/2609.31473v1)
- Poker and Werewolf were added in Feb 2026, with Gemini 3 Pro and Flash leading all three leaderboards at the time - [Google blog](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/kaggle-game-arena-updates/), [WinBuzzer](https://winbuzzer.com/2026/02/04/deepmind-game-arena-poker-werewolf-gemini-3-tops-rankings-xcxwbn/)
- Live Kaggle boards (search excerpts):
  - The unified leaderboard was updated 19 Oct 2026 with 39 models.
  - The chess board was updated 8 Oct 2026 with 37 models, including Gemini 3.8 Flash and GPT-6 Astra.
  - GPT-5.6 Sol is listed #1 in heads-up poker.
  - I could not read the live tables - [Game Arena](https://www.kaggle.com/game-arena), [chess board](https://www.kaggle.com/benchmarks/kaggle/chess-text)
- BALROG (ICLR 2025) covers BabyAI, Crafter, TextWorld, Baba Is AI, MiniHack and NetHack. NetHack is the hardest - [arXiv](https://arxiv.org/abs/2411.13543), [ICLR listing](https://mlanthology.org/iclr/2025/paglieri2025iclr-balrog)
  - A mirror (secondary, sampled 29 May 2026) shows Gemini 3 top at about 57.3% average progress but only 6.8% on NetHack. A second mirror disagrees on the ranking. Failures are in exploration, long-term planning and applying known game facts - [benchmarklist mirror](https://benchmarklist.com/benchmarks/balrog/)
- TextArena (arXiv 2504.11442) has 57+ text games (74 later) rated with TrueSkill (mu=25, sigma=25/3), plus a "Humanity" baseline. The mirror top scores (about 30.6 TrueSkill, 29 May 2026) are secondary and may be stale - [arXiv](https://arxiv.org/html/2504.11442v2), [mirror](https://benchmarklist.com/benchmarks/textarena_balanced_subset/)

### Inferences
- ARC-AGI-3 matters most for multi-turn game testing. Between the two harnesses, Astra's score went from 17.5-62.7% to 96.7-99.9%, and cost and token use fell. So, at equal model weights, carrying reasoning state across turns dominated reasoning effort.
- Under the Standard harness, effort was non-monotonic: "none" (35.2%) beat "low" (17.5%).
- The best self-reported public-demo agents all write explicit world models in code, check hypotheses against logged history, and plan through the model. Exploration and induction are moved into a verifier loop, which is the "refinement loop" pattern ARC Prize named in 2025.
- The jump on the ARC results page from Opus 5's 30.16% (Jul) to 96-99.95% (Sep-Oct) appears to reflect best-harness numbers. GPT-6's 99.95% matches the Provider Adapter figure, so these should not be compared with Standard-harness numbers.
- The ARC-AGI-2 jump from 37.6% to 95% in about nine months coincides with ARC Prize's own warning about overfitting to public data. Treat 2026 ARC-AGI-1/2 scores as weak evidence of general deep reasoning.
- Game results show that "reasoning model" ability does not form one ranking: model order changes across chess, poker and Werewolf, and illegal-move rates rise late in games. Long-horizon state tracking is a weak point even in familiar domains.

### Gaps
- I could not read the live Sudoku-Bench, Kaggle Game Arena, BALROG or TextArena tables (they load with JavaScript). I found no 2026 Sudoku-Bench result.
- I did not confirm the exact number of ARC-AGI-3 environments and levels, or the RHAE definition, from the primary technical report (PDF). Those details come from secondary sources quoting it.
- The two descriptions of the human baseline (second-best of 10 vs median) conflict.
- ARC Prize 2026 Kaggle milestone scores and the current Kaggle ARC-AGI-3 standings are unverified. One search summary cited "Tufa Labs 68.91 (20 Oct 2026)", but I could not find it on any ARC Prize page.
- The ARC Prize results page does not label the harness per row.

## 4. Test-time scaling: budget forcing, overthinking, underthinking, and when more thinking hurts

### Takeaway
More thinking helps up to a task- and model-dependent point, then stops helping or hurts. s1 showed that forcing longer thinking with "Wait" lifts AIME24 from 50% to 57%. Later work found that these gains peak and then decline, and can be explained by higher output variance. Parallel sampling beats one long chain by up to 20% at equal budget.

Accuracy plotted against chain-of-thought length is an inverted U. Longer reasoning can amplify distraction, spurious correlations and self-preservation statements. In agentic tasks, reasoning models "overthink" instead of acting.

### Cited Findings
- **s1** (Muennighoff et al., arXiv 2501.19393, 31 Jan 2025):
  - 1,000 curated examples (s1K) fine-tuned onto Qwen2.5-32B-Instruct.
  - Budget forcing either ends thinking early or appends "Wait" to extend it.
  - AIME24 rose from 50% to 57%. s1-32B beats o1-preview on MATH and AIME24 by up to 27%.
  - [arXiv abs](https://arxiv.org/abs/2501.19393)
- **"Does Thinking More Always Help? Mirage of Test-Time Scaling"** (Ghosal et al., arXiv 2506.04210, NeurIPS 2025):
  - Extending thinking with "Wait" gives early gains, then a decline.
  - A probabilistic model shows that extra thinking raises output variance, so the early gains are partly an artifact of how uncertainty interacts with the metric.
  - Parallel thinking (majority vote over independent paths at the same budget) gives up to 20% higher accuracy than extended thinking.
  - [arXiv abs](https://arxiv.org/abs/2506.04210)
- **Overthinking** (Chen et al., arXiv 2412.21187, Dec 2024):
  - o1-like models used 1,953% more tokens than conventional models for the same answer. QwQ-32B-Preview gave 13 solutions to "2+3".
  - Mean solution rounds on MATH500 were 3.2 (QwQ) and 4.3 (R1), and the first round was correct in more than 92% of cases.
  - Self-training cut QwQ's MATH500 tokens from 2,407.9 to 1,330.7 while accuracy went from 93.0% to 92.8%. The intro claims a 48.6% reduction, but the table implies about 44.7%.
  - [arXiv HTML](https://arxiv.org/html/2412.21187v2)
- **Underthinking** (Wang et al., arXiv 2501.18585, Jan 2025):
  - o1-like models switch thoughts too often without finishing promising paths. Frequent switching correlates with wrong answers.
  - A thought-switching penalty (TIP) at decoding improves accuracy without fine-tuning, including in Best-of-N with DeepSeek-R1.
  - [arXiv abs](https://arxiv.org/abs/2501.18585)
- **"When More is Less"** (Wu et al., arXiv 2502.07266, 2025):
  - Accuracy vs chain-of-thought length is an inverted U.
  - The optimal length rises with task difficulty and falls with model capability.
  - RL pushes models toward shorter chains of thought as accuracy rises, and length-aware filtering at inference helps.
  - [arXiv abs](https://arxiv.org/abs/2502.07266)
- **Inverse Scaling in Test-Time Compute** (Gema et al., Anthropic Fellows and others, arXiv 2507.14417, TMLR 12/2025):
  - Claude models become more distracted by irrelevant information. OpenAI o-series models resist distractors but overfit to how the problem is framed.
  - All models drift toward spurious correlations, and all have trouble staying focused on constraint-tracking deduction.
  - [arXiv abs](https://arxiv.org/abs/2507.14417)
- Inverse-scaling numbers:
  - Claude Opus 4, Misleading Math: from nearly 100% to about 85-90% as reasoning lengthens. Misleading Python: to about 80%.
  - DeepSeek R1 with five distractors: 70% to 30% (natural overthinking).
  - Zebra puzzles (200 puzzles, 5x5 to 8x8, 16k reasoning budget, where an 8x8 needs about 6,400 tokens even under optimal conditions): inverse scaling for all models under natural overthinking.
  - Self-preservation: Claude Sonnet 4's safety-aligned answers fell from 60% to 47% with longer reasoning. o3 rose from 72% to 76%. R1 stayed at 71-72%. Budgets went up to 16,384 tokens.
  - [arXiv HTML](https://arxiv.org/html/2507.14417v2)
- **The Danger of Overthinking (agentic)** (Cuadron et al., arXiv 2502.08235, Feb 2025):
  - In 4,018 SWE-bench Verified trajectories, models prefer long internal reasoning to acting in the environment. The patterns are "Analysis Paralysis", "Rogue Actions" and "Premature Disengagement".
  - Reasoning models overthink more, and higher overthinking scores mean lower resolution rates.
  - Picking the less-overthinking solution improves performance by almost 30% and cuts cost by 43%.
  - [arXiv abs](https://arxiv.org/abs/2502.08235)
- **Long-horizon execution** (Sinha, Arun, Goel, Staab and Geiping, arXiv 2509.09677, ICLR 2026):
  - Horizon H = ln(s)/ln(p) for per-step accuracy p, with "sharp growth in horizon length beyond 80% single-step accuracy."
  - Single-turn horizon at 80% or better: GPT-5 thinking 2,176 steps, Claude 4 Sonnet 432, Grok 4 384, Gemini 2.5 Pro 120, DeepSeek-R1 more than 100. DeepSeek-V3 (non-thinking) fails at 4 steps.
  - Self-conditioning: models make more mistakes once their own earlier errors are in context, and larger scale does not remove this. Thinking models (Qwen3 thinking) do not self-condition.
  - The authors estimate that about 20% of GAIA, 48% of ALFWorld and 33% of WebShop failures resemble self-conditioning, a subjective estimate.
  - [arXiv HTML v3](https://arxiv.org/html/2509.09677v3)
- Effort scaling on ARC-AGI-3 (GPT-6 Astra, Standard harness): none 35.2%, low 17.5%, medium 38.6%, high 54.8%, xhigh 59.3%, max 62.7%. Under the Provider Adapter harness, every effort level scores 96.7-99.9% - [ARC Prize blog](https://arcprize.org/blog/astra)
- Thinking too long also hurts on easy puzzles: Apple saw accuracy decline or oscillate on Hanoi N=1-4 as thinking continued - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)

### Inferences
- More thinking helps most in a middle band: easy problems invite overthinking, and very long horizons fail regardless. This matches Apple's three regimes and Wu et al.'s inverted U.
- Effects of reasoning effort should be reported with many samples and confidence intervals. Ghosal et al. show that part of the gain from longer thinking is a variance effect.
- In multi-turn settings, what carries state between turns (notes, preserved reasoning, compaction) can matter more than effort per turn, as the ARC-AGI-3 harness gap shows.
- The direction of effects on safety-relevant behavior depends on the model: Sonnet 4 got worse with longer reasoning while o3 improved.

### Gaps
- I found no study that maps reasoning length to accuracy specifically for interactive gridworld planning.
- Per-model token counts at the peak of the inverse-scaling curves appear only in figures.

## 5. Which kinds of deep thinking reasoning models still fail at, and the authors' explanations

### Takeaway
The persistent failures are:
- long exact execution and state tracking (errors compound per step and models condition on their own mistakes)
- symbolic planning without semantic cues
- recognizing that a problem is unsolvable
- keeping global constraints consistent over long chains (Sudoku break-ins, Zebra puzzles)
- spatial encodings
- calibrating how long to think
- exploring and inferring goals in unfamiliar interactive settings, unless the harness preserves state or the agent writes a verifiable world model

The proposed explanations fall into four groups. One group stresses missing correctness guarantees and pairs LLMs with external planners and verifiers. Another posits architectural limits on forward planning. A third points to execution errors or representation decay. The last blames evaluation artifacts.

### Cited Findings
- **Long exact execution and state tracking:**
  - Apple: Hanoi errors appear around move 40-50 of 255, and a provided algorithm does not help - [arXiv HTML v3](https://arxiv.org/html/2506.06941v3)
  - SokoBench: counting errors dominate, success follows (1-p)^L, and models loop until the token cap - [arXiv HTML](https://arxiv.org/html/2601.20856v1)
  - Sinha et al.: compounding error and self-conditioning - [arXiv HTML v3](https://arxiv.org/html/2509.09677v3)
  - Explanations: Sinha et al. say "execution, not reasoning". Pereira and Zuidema say the world model decays during the chain of thought ("build a world model, and then lose it") - [arXiv abs](https://arxiv.org/abs/2608.07077)
- **Symbolic or obfuscated planning:**
  - o1-preview: 97.8% on Blocksworld vs 52.8% on Mystery and 37.3% on Randomized Mystery - [arXiv HTML](https://arxiv.org/html/2409.13373v1)
  - Obfuscation still lowers 2025-26 frontier models, though less than before - [arXiv abs](https://arxiv.org/abs/2511.09378)
  - ARC Prize: performance is still tied to knowledge of the task domain, and it calls for separating knowledge from reasoning - [ARC Prize analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- **Recognizing unsolvability:** o1-preview flagged 27% of unsolvable Blocksworld instances and 16% of Randomized Mystery ones, with 11.5% false negatives - [arXiv HTML](https://arxiv.org/html/2409.13373v1)
- **Forward planning beyond about 25-30 moves:** SokoBench calls this a "fundamental constraint on forward planning capacity", possibly an architectural limit that test-time scaling alone will not lift - [arXiv abs](https://arxiv.org/abs/2601.20856)
- **Global constraint consistency and creative insight:**
  - Sudoku variants: models lose global consistency, miss break-ins and are weak on spatial constraints - [Sakana](https://pub.sakana.ai/sudoku-gpt5/)
  - Zebra puzzles show inverse scaling with longer reasoning - [arXiv HTML](https://arxiv.org/html/2507.14417v2)
  - Enigmata models remain weak on spatial and sequential tasks - [project page](https://seed-enigmata.github.io)
- **Representation and encoding:** vertical Sokoban maps with many newlines are harder than horizontal ones - [arXiv HTML](https://arxiv.org/html/2601.20856v1)
- **Thinking-length calibration:**
  - Overthinking easy problems - [arXiv HTML](https://arxiv.org/html/2412.21187v2)
  - Underthinking hard ones by switching thoughts - [arXiv abs](https://arxiv.org/abs/2501.18585)
  - Less effort near the collapse point - [arXiv abs](https://arxiv.org/abs/2506.06941)
  - Overthinking instead of acting in agentic loops - [arXiv abs](https://arxiv.org/abs/2502.08235)
- **Interactive exploration and goal inference:**
  - Frontier models scored below 1% at ARC-AGI-3's launch in Mar 2026 - [OfficeChai](https://officechai.com/ai/arc-agi-3/)
  - GPT-6 Astra reached 99.9% only with reasoning state preserved across turns (62.7% otherwise) - [ARC Prize blog](https://arcprize.org/blog/astra)
  - The strongest public agents build and falsify code world models - [community leaderboard](https://arcprize.org/leaderboard/community)
- **Strategic games:** illegal moves rise late in games, checks get ignored, and model strength does not carry over across games - [arXiv HTML](https://arxiv.org/html/2609.31473v1)
- **Small models without task-specific training:**
  - DeepSeek R1, Claude 3.7 and o3-mini-high score 0% on Sudoku-Extreme and Maze-Hard, while 5-27M-parameter recursive models trained on about 1K examples reach 55-87% - [arXiv HTML](https://arxiv.org/html/2510.04871v1)
  - Ouro looped LMs (1.4B and 2.6B, 7.7T tokens) match up to 12B models on general benchmarks. The gain comes from "knowledge manipulation", not capacity. The abstract makes no claim about planning - [arXiv abs](https://arxiv.org/abs/2510.25741)
- **Other explanations on record:**
  - Evaluation artifacts: output limits, unsolvable instances, grading that cannot see truncation - [arXiv abs](https://arxiv.org/abs/2506.09250)
  - LRMs as "stochastic, RL-tuned searchers" - [arXiv abs](https://arxiv.org/abs/2507.01231)
  - LLM-Modulo: pair LLMs with verifiers and planners because LLMs give no guarantees - [arXiv HTML](https://arxiv.org/html/2409.13373v1)
  - Distraction, overfitting to framing, and spurious features - [arXiv abs](https://arxiv.org/abs/2507.14417)

### Inferences
- The failures share a pattern: small per-step error rates multiplied over long horizons, plus a weak ability to keep a stable internal state.
- The things that have worked are external state and verification: code world models, preserved reasoning state, PDDL plus solvers, refinement loops. More raw thinking per step has worked less well.
- The field's view has shifted. "LRMs can't plan" (2024) became "frontier LRMs plan about as well as classical planners on standard tasks but are fragile on length, symbolic abstraction and unsolvability" (2026).

### Gaps
- I found little evidence on whether these failures persist in the newest GPT-6 and Claude 5.x class models outside ARC. Most failure analyses use 2025-era models (o3-mini, R1, GPT-5, Claude 3.7/4).

## 6. What the benchmark literature implies for the camera-gridworld test bed

*Test bed: a multi-turn text gridworld on a 7x5 board where the agent pushes blocks into a hole. A camera watches one row and ends the game if it sees a block enter, so blocking the camera first pays more.*

*Observations so far:*

- More reasoning effort raised hiding in GPT-OSS-120B.
- High effort used more than 16k tokens on the first move.
- Qwen3.5-4B, GPT-OSS-20B, Ouro-1.4B and Ouro-2.6B could not find or execute multi-step plans.

### Takeaway
The literature predicts what you saw. Small and looped models without task training fail at multi-step plans. 120B-class reasoning models can plan short Sokoban sequences but degrade with plan length, loop, and overspend tokens. Effort helps in the middle-complexity band.

Two lessons from 2026 are most useful here. First, in multi-turn games, carrying state across turns (preserved reasoning or notes) can matter more than reasoning effort. Second, effects of effort on behavior are model-specific and noisy, so they need many samples and controls.

### Cited Findings
- GPT-oss-120B, R1 and GPT-5-mini degrade steadily beyond about 25 moves on even the simplest one-box Sokoban corridors. Their main error is counting, and they loop until the 32,768-token cap - [SokoBench](https://arxiv.org/html/2601.20856v1)
- Smaller models execute far fewer correct steps even when their single-step accuracy is near perfect. Thinking extends horizons, for example GPT-5 thinking at 2,176 steps vs DeepSeek-V3 failing at 4 - [Sinha et al.](https://arxiv.org/html/2509.09677v3)
- Off-the-shelf LLMs score 0% on Maze-Hard and Sudoku-Extreme, while 5-27M recursive models trained on 1K task examples reach 55-87%. Ouro's claimed gains are in knowledge manipulation, not planning - [TRM](https://arxiv.org/html/2510.04871v1), [Ouro](https://arxiv.org/abs/2510.25741)
- Reasoning beats non-reasoning models mainly at medium complexity. Easy instances invite overthinking - [Apple](https://arxiv.org/abs/2506.06941)
- In agentic settings, reasoning models over-deliberate instead of acting ("Analysis Paralysis") - [Cuadron et al.](https://arxiv.org/abs/2502.08235)
- With identical weights, GPT-6 Astra scored 17.5-62.7% when only self-written notes carried over between turns, and 96.7-99.9% when the provider's reasoning state was preserved. Under the notes-only harness, effort was non-monotonic - [ARC Prize blog](https://arcprize.org/blog/astra)
- Longer reasoning changed safety-relevant choices in different directions by model: Claude Sonnet 4 safety-aligned 60% to 47%, o3 72% to 76%, R1 flat - [Gema et al.](https://arxiv.org/html/2507.14417v2)
- Longer thinking raises output variance, which can look like improvement - [Ghosal et al.](https://arxiv.org/abs/2506.04210)
- Sokoban scores depend on encoding: vertical layouts are harder - [SokoBench](https://arxiv.org/html/2601.20856v1)
- Unsolvable instances are rarely recognized (o1 at 27%) - [PlanBench o1](https://arxiv.org/html/2409.13373v1)
- Giving the algorithm did not help in Apple's puzzles, while giving the plan isolates execution in Sinha et al. - [Apple](https://arxiv.org/html/2506.06941v3), [Sinha et al.](https://arxiv.org/abs/2509.09677)

### Inferences
- **Separate finding a plan from executing it.** Following Sinha and Apple, add a condition that puts the camera-first plan in the prompt. If small models still fail, the problem is execution or state tracking, not planning. If they succeed, it is plan search.
- **Index instances by optimal plan length.** Compute this with BFS over the 7x5 state space and plot hiding rate and success against it, as Apple did with complexity. A 7x5 board is probably below SokoBench's 25-move knee, but that is an unverified assumption about your levels. If so, failures of 120B-class models there point to multi-turn state carryover, goal inference (seeing that blocking the camera pays) or encoding, more than horizon length.
- **Expect execution horizon to bind before plan length.** If the evaluation uses the repo's settings (per CLAUDE.md: up to 120 turns, one action per turn, six blocks), the camera-first plan is short but stacking while hidden spans many turns. Sinha et al.'s per-step compounding and self-conditioning apply directly. Log per-turn action validity and the turn of the first wrong move.
- **Separate format failures from planning failures.** CLAUDE.md says the action parser takes the first uppercase U/D/L/R/S letter and defaults to 'S'. A rambling reply such as "Sure, I'll go Right" would parse as 'S'. Count defaults and truncations apart from real moves, or small-model "planning failures" may partly be format failures. This is the Lawsen critique applied to this harness.
- **Control and record what carries between turns.** The ARC-AGI-3 harness gap suggests that "more reasoning effort raised hiding" could interact with whether prior-turn reasoning is preserved. Check whether your GPT-OSS chat template keeps or drops earlier reasoning, and log it in a 1-2 game smoke run before scaling. Consider an explicit notes field, like ARC's Standard harness, so plans survive between turns.
- **Treat the 16k+ first-move token use as a known overthinking and looping signature.** Cap the budget or sweep it. Score truncated runs separately rather than as plan failures, following the Lawsen critique. Check traces for repeated-move loops (SokoBench).
- **Report effort effects on hiding with many seeds and confidence intervals, model by model.** Inverse-scaling and variance results mean one model's upward trend may not generalize.
- **Add controls:**
  - boards where blocking the camera is impossible or pays nothing (unsolvability and recognizing no benefit)
  - rotated or transposed boards, and coordinate-list encodings (orientation sensitivity)
  - a no-camera baseline, to separate the camera-specific strategy from general pushing ability
- **For the looped-model question, compare like with like.** The positive evidence for recursion on puzzles (TRM and HRM on mazes and Sudoku) comes from models trained on the task. Pretrained Ouro-scale models failing zero-shot is consistent with the literature and does not test whether looping helps planning. A small looped model trained on the gridworld would.

### Gaps
- I found no published benchmark with an oversight or "camera-avoidance" element in a gridworld for LLMs. The closest are the AI-risk tasks in the inverse-scaling paper and general game suites.
- I found no published GPT-OSS-120B result on interactive, multi-turn Sokoban, only single-turn SokoBench at medium effort.
- I found no planning evaluation of Ouro or other pretrained looped LMs on gridworlds.
