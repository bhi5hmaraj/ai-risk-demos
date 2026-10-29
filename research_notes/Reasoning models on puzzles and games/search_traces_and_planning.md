# Learning to search and plan from search traces, process supervision and imitation (puzzles and games)

Conventions: numbers come from the linked primary papers (mostly arXiv). "pp" means percentage points. Claims that come only from blogs are marked (blog, unverified). Current as of late Oct 2026.

## 1. Stream of Search, Searchformer, Dualformer and related "learn to search" work: how do search traces compare with solutions only?

### Takeaway
For small transformers trained from scratch on one puzzle, search traces beat solutions only, most of all with little data and on hard instances: Stream of Search doubles Countdown accuracy (51.27% vs 25.73%), and Searchformer beats solution-only models 5-10x larger trained on 10x more data. Pure imitation inherits the teacher's cost and ceiling, so the best results add a filter-and-retrain or RL loop: bootstrapped Searchformer traces end 26.8% shorter than A*, and SoS plus STaR/APA solves 36% of problems its teacher solvers left unsolved.

### Cited Findings
Stream of Search (SoS), Countdown:
- SoS writes the whole search process (state expansions, goal checks, backtracking) as one flat string, in a single "language of search" that covers several symbolic strategies. Task: Countdown with 4 input numbers and a target - [SoS, Gandhi et al. 2024 (2404.03683)](https://arxiv.org/abs/2404.03683)
- Data: 12 search strategies built from BFS and DFS with sum or multiply heuristics, 500,000 trajectories. Only ~57% reach the solution, so the SoS set holds 285,000 correct solutions against 500,000 in the optimal-path (OP) set. With 5 input numbers one trajectory can reach 60,000 tokens, which is why the paper stays at 4 - [SoS](https://arxiv.org/abs/2404.03683)
- Model: GPT-Neo with 250M parameters and a 4096-token context, trained from scratch with a causal LM loss - [SoS](https://arxiv.org/abs/2404.03683)
- Result: SoS 51.27% vs OP 25.73% on held-out inputs (the abstract's "+25%"). SoS sits slightly below the average of the symbolic solvers that produced its data. Its visited states correlate best with DFS-sum (0.57) and worst with BFS breadth-5 sum (0.27), so it copies no single teacher - [SoS](https://arxiv.org/abs/2404.03683)
- Policy improvement: STaR (expert iteration, 100,000 correct trajectories per round) converges after 3 rounds and adds 5% on held-out inputs. APA (actor-critic with a copy of the LM as value net) plateaus after ~4000 steps, and the authors reset its reference policy 3 times. The fine-tuned models solve 36% of previously unsolved training problems, including some that no heuristic solver solves. The APA model drifts furthest from the symbolic strategies - [SoS](https://arxiv.org/abs/2404.03683)

Searchformer ("Beyond A*"), mazes and Sokoban:
- Trace format: the A* run becomes token events ("create" or "close" a node, with cost-from-start and heuristic values). Solution-only sequences are <prompt><plan>. Search-augmented sequences are <prompt><trace><plan>. The model is an encoder-decoder transformer - [Searchformer, Lehnert et al. 2024 (2402.14083)](https://arxiv.org/abs/2402.14083)
- Mazes (10x10 to 30x30, models of 15M, 46M and 175M): search-augmented models beat solution-only ones with 5-10x smaller models and a 10x smaller dataset. Below ~100,000 training sequences the solution-only model falls off while search-augmented models stay high. Solution-only targets are ~10x shorter. On over 90% of test mazes, search-augmented models emit traces thousands of tokens long without one wrong token (deterministic A* data) - [Searchformer](https://arxiv.org/abs/2402.14083)
- Sokoban setup: 7x7 grid with 2 extra interior walls, 2 boxes and 2 docks. A* traces reached almost 100,000 tokens, so the dataset kept only sequences of at most 10,000 tokens. Test set: 200 unseen puzzles - [Searchformer](https://arxiv.org/abs/2402.14083)
- Sokoban results (solved / optimal): 45M solution-only 90.3% / 86.8%. 45M search-augmented 92.5% / 90.8%. Searchformer after bootstrapping steps 1, 2, 3: 95.5/93.5, 96.0/93.4, 95.5/93.7. 175M solution-only 95.7/90.0. 175M search-augmented 95.2/93.2. Largest solution-only model 96.5/92.2 (the table says 757M, the text says 747M) - [Searchformer](https://arxiv.org/abs/2402.14083)
- Bootstrapping ("search dynamics bootstrapping"): sample traces from the model on training tasks, keep any that ends in an optimal plan and is shorter than its training sequence, fine-tune, repeat. After 3 rounds the traces average 26.8% shorter than A* on optimally solved test tasks (the abstract phrases it as "up to 26.8% fewer search steps"). The paper's length score ILR (higher means shorter relative to A*) goes from 0.919 for the imitation model to 1.062, 1.181 and 1.343 - [Searchformer](https://arxiv.org/abs/2402.14083)

Dualformer:
- Same A* traces, but training drops structured parts at random: level 1 removes all close clauses, level 2 also the cost tokens, level 3 also a random 30% of create clauses, level 4 the whole trace. One model then runs in fast (plan only), slow (trace and plan) or auto mode - [Dualformer, Su et al. 2024, ICLR 2025 (2410.09918)](https://arxiv.org/abs/2410.09918)
- 30x30 mazes: slow mode 97.6% optimal vs Searchformer 93.3%, with 45.5% fewer reasoning steps. Fast mode 80% optimal vs 30% for a model trained on solutions only. Auto mode 96.6% optimal with 59.9% fewer steps than Searchformer. Its traces are more diverse, and the same trick improved Llama fine-tuning on MATH - [Dualformer](https://arxiv.org/abs/2410.09918)

Related work:
- Procedure cloning: supervised sequence prediction of the expert's intermediate computations (for example a search), in addition to its actions. It generalizes to unseen environment configurations, including ones where running the expert procedure is infeasible - [Procedure cloning, Yang et al. 2022 (2205.10816)](https://arxiv.org/abs/2205.10816)
- System-1.x: a controller splits a planning problem into easy and hard sub-goals for a fast planner and an A*-like slow planner, all fine-tuned from search traces only. On Maze and Blocksworld it beats System-1, System-2 and symbolic A*, and a larger slow-mode share means more search and better plans - [System-1.x, Saha et al. 2024, ICLR 2025 (2407.14414)](https://arxiv.org/abs/2407.14414)
- Guided-ReST: SoS traces are noisy and waste test-time compute. Guided-ReST inserts optimal solutions as step-wise landmarks into the model's own search, fine-tunes on the resulting traces, and gains over 10% accuracy on Countdown vs baseline fine-tuning (also CodeContests and CodeForces) - [Guided-ReST, Moon et al. 2024, NeurIPS 2025 (2410.02992)](https://arxiv.org/abs/2410.02992)
- MLM-U: an 8M transformer trained to predict several steps forward and backward solves all mazes up to 20x20, where next-token training peaks at 20.6% on 20x20. On 30x30 it reaches 85.5% vs 70.2% for a 175M next-token model with A* trace supervision. It is 4x more sample-efficient and converges 2x faster in GPU hours - [MLM-U, Nolte et al. 2024 (2412.05117)](https://arxiv.org/abs/2412.05117)
- LLM heuristics for A*: both A* and the LLM need accurate predictions on nodes near the goal. A data recipe built on that cuts A* iterations up to 15x and wall-clock time up to 5x on mazes, Sokoban and sliding-tile puzzles - [Gupta & Li 2024, EMNLP Findings (2407.09985)](https://arxiv.org/abs/2407.09985)
- Self-improving plan generation: a decoder-only transformer trained on sub-optimal plans improves over rounds that combine several model calls with graph search. Plans end 30% shorter on average than the source symbolic planner's, and over 80% are optimal where the optimum is known (Blocksworld, Logistics, Labyrinth, Sokoban) - [Gieselmann et al. 2026, ICAPS 2026 (2605.03625)](https://arxiv.org/abs/2605.03625)
- LinTree: LLM policies that see the raw search history do not reliably beat best-first search with an LLM heuristic (Blocks World, grid navigation, Sokoban). Adding parent pointers, which name the earlier state a backtrack returns to, improves both accuracy and search efficiency - [LinTree, Kang, Teh, Lee 2026 (2605.31492)](https://arxiv.org/abs/2605.31492)
- Counterpoint on Countdown: at equal test-time compute, a model trained on direct solutions and sampled best-of-n beats a model trained on DFS backtracking traces. On Sudoku the result flips - [Qin et al. 2025, COLM 2025 (2504.07052)](https://arxiv.org/abs/2504.07052)

### Inferences
- Traces buy sample efficiency and plan quality more than raw solve rate. With enough data and parameters, the Sokoban solution-only model solves as many puzzles (96.5%) as Searchformer, but fewer of them optimally (92.2% vs 93.7%).
- Imitation alone caps at the teacher. The Searchformer imitation model writes traces longer than A* (ILR 0.919), and SoS sits below its teachers' average. Every result above the teacher adds a loop that filters the model's own successes and retrains on them (STaR, APA, bootstrapping, Guided-ReST, self-improvement).
- Failed branches carry signal. SoS trains on ~43% failed searches and fewer correct answers than OP, yet doubles OP's accuracy.
- The advantage of traces depends on the task. Qin et al. find it reverses on Countdown under a fixed compute budget, where parallel sampling of a direct-solution model wins.
- All clean wins in this section use sub-1B models trained from scratch on one task with a perfect simulator. They do not show that a pretrained 1-3B LLM gains the same from trace SFT.

### Gaps
- I did not find maze bootstrapping numbers in the parts of the Searchformer paper I read. The abstract claims larger tasks get higher solve rates and shorter traces.
- Dualformer's Sokoban numbers were not extracted.
- I found no paper that puts Searchformer or SoS-style traces into a pretrained LM above 1B on Sokoban and compares the result against RL from the same start.

## 2. Sudoku and logic puzzles learned from solver traces (Shah et al. 2024, other transformer Sudoku papers, Sudoku-Bench)

### Takeaway
A 42M GPT-2 trained on Sudoku steps in the solver's order solves 87.18% of puzzles greedily and 94.21% with beam width 5. Fixed or random cell order gives 7.2% and ~1%, so the order of supervision decides whether the model learns at all. Newer recursive and diffusion models reach 87-100% on hard Sudoku sets with no traces. Frontier LLMs solved under 15% of Sudoku-Bench variants at release, and GPT-5 reached 33% per a Sakana blog (unverified).

### Cited Findings
- Shah et al. setup: GPT-2 style decoder with 8 layers, 8 heads and width 576, 42M parameters in all. 1.8M training and 0.1M test Sudokus. Each puzzle is a sequence of (cell, value) steps - [Shah et al. 2024 (2409.10502)](https://arxiv.org/abs/2409.10502)
- Cell order: fixed top-left to bottom-right order gives 58.64% cell accuracy and 7.2% full puzzles. Random order gives ~52% cell and ~1% full. The solver's own order (easy cells first) gives 94.23% cell and 87.18% full. Beam search with width 3 and 5 lifts full-puzzle accuracy to 91.36% and 94.21% (cell 96.07% and 98.03%) - [Shah et al. 2024](https://arxiv.org/abs/2409.10502)
- When the solver hints which cell is easy to fill next, cell accuracy reaches 99.02%. The remaining error lies mostly in choosing which cell to fill - [Shah et al. 2024](https://arxiv.org/abs/2409.10502)
- Zebra (Einstein) puzzles: 92.04% fully solved, from 0.3M training puzzles. GPT-4o and Gemini-1.5 Pro with 4-shot CoT solve 0% of Sudokus fully (9.5% and 10.2% cell accuracy). Linear probes decode each cell's candidate set from hidden states - [Shah et al. 2024](https://arxiv.org/abs/2409.10502)
- Trial and error: a vanilla GPT-2 that imitates Sudoku rules plus explicit DFS with informed guessing and backtracking reaches 99%. Almost every Sudoku falls to the rules plus at most one guess ("depth-1 guessing") - [Giannoulis, Pantis, Tzamos 2025 (2509.22023)](https://arxiv.org/abs/2509.22023)
- At equal test-time compute, a model trained on DFS backtracking traces beats a direct-solution model with best-of-n on Sudoku, and loses on Countdown - [Qin et al. 2025](https://arxiv.org/abs/2504.07052)
- SFT warm-up with synthetic backtracks before RL: Sudoku wants 5 or more backtracks per trace. Qwen2.5-7B scores 11.8% (0 backtracks), 20.8% (1), 32.1% (5), 28.0% (10). Llama3.2-3B scores 9.6%, 23.4%, 25.2%, 26.9%. QwQ-32B scores 0.0% on the same set - [Cai et al. 2025 (2505.24273)](https://arxiv.org/abs/2505.24273)
- Discrete diffusion (MGDM) solves 100% of its Sudoku set vs 20.7% for an autoregressive model, and 91.5% vs 45.8% on Countdown, with no search - [MGDM, Ye et al. 2024, ICLR 2025 (2410.14157)](https://arxiv.org/abs/2410.14157)
- Earlier neural Sudoku solvers without language modeling: recurrent relational networks solve 96.6% of the hardest Sudokus - [RRN, Palm et al. 2017 (1711.08028)](https://arxiv.org/abs/1711.08028). SATNet learns 9x9 Sudoku from examples alone through a differentiable MAXSAT layer - [SATNet, Wang et al. 2019 (1905.12149)](https://arxiv.org/abs/1905.12149). A recurrent transformer beats GNNs and SATNet on CSPs, including visual Sudoku - [Yang, Ishay, Lee 2023 (2307.04895)](https://arxiv.org/abs/2307.04895). ConsFormer, trained without labels to improve an assignment by one step, handles harder CSPs by running more iterations at test time - [ConsFormer, Xu et al. 2025 (2502.15794)](https://arxiv.org/abs/2502.15794)
- HRM: 27M parameters, ~1000 training examples, no pretraining and no CoT data. Near-perfect on Sudoku-Extreme Full and on optimal paths in 30x30 Maze-Hard, where CoT LLMs score 0%. ARC-AGI-1 40.3% vs 34.5% for o3-mini-high. On Sudoku-Extreme a wider transformer gains nothing, while a deeper one does - [HRM, Wang et al. 2025 (2506.21734)](https://arxiv.org/abs/2506.21734)
- TRM: one 2-layer network with 7M parameters. Sudoku-Extreme 55% -> 87%, Maze-Hard 75% -> 85%, ARC-AGI-1 45%, ARC-AGI-2 8%. TRM cites an ARC Prize Foundation analysis (blog, unverified here): deep supervision doubled HRM's accuracy (19% -> 39%), while the two-level hierarchy added little (35.7% -> 39.0%) - [TRM, Jolicoeur-Martineau 2025 (2510.04871)](https://arxiv.org/abs/2510.04871)
- Looped transformers: with recall plus outer normalization, fixed points are reachable and depend on the input. Single-layer looped transformers trained on chess, Sudoku and prefix sums behave as this framework predicts - [Labovich 2026 (2604.15259)](https://arxiv.org/abs/2604.15259)
- Sudoku-Bench: 100 puzzles (15 of 4x4, 15 of 6x6, 70 of 9x9). Fifty of the 9x9 puzzles were made for the benchmark by the hosts of Cracking the Cryptic. Difficulty spans 1 to 5 stars, in a text format that gives rules and visual elements as coordinates - [Sudoku-Bench, Seely et al. 2025, Sakana AI (2505.16135)](https://arxiv.org/abs/2505.16135)
- Sudoku-Bench results at release: the best LLMs solve under 15% unaided. o3-mini-high single-shot: 14.0% overall (73.3% on 4x4, 6.7% on 6x6, 2.9% on 9x9). Gemini 2.5 Pro: 11.0% in both multi-step and single-shot mode. Qwen3-235B-A22B: 6.0% multi-step and 8.0% single-shot. Models parse the rules but miss the logical "break-in" and fall back to guessing and search - [Sudoku-Bench](https://arxiv.org/abs/2505.16135)
- Imitation data: Sudoku-CTC-Reasoning holds 2,565 puzzles solved on Cracking the Cryptic, with audio transcripts and the matching SudokuPad action sequences, described as "thousands of hours" of expert reasoning - [Sudoku-Bench](https://arxiv.org/abs/2505.16135), [dataset](https://huggingface.co/datasets/SakanaAI/Sudoku-CTC-Reasoning)
- Oct 2025 update (blog, unverified): GPT-5 (high) leads with a 33% weighted solve rate on challenge_100 and is called the first LLM to solve a 9x9 "modern sudoku". A GRPO-tuned Qwen2.5-7B was "lackluster" - [Sakana AI blog](https://pub.sakana.ai/sudoku-gpt5/)
- RL needs a curriculum on hard Sudoku: frontier multimodal models fail on 9x9 visual Sudoku, and near-zero success starves RLVR of reward. AdaC-GRPO builds batches from online rollout accuracy and lets Qwen2.5-VL-7B close 62.2% of its gap to Gemini-2.5-Pro - [AdaC-GRPO, Li et al. 2026 (2610.13184)](https://arxiv.org/abs/2610.13184)
- A constraint-solver oracle used as an action-level reward (VPR) lifts Qwen3-4B to 80.6% of Sudokus solved, vs 29.0% for outcome-reward GRPO and 0.0% for VinePPO and the base model - [VPR, Yuan et al. 2026 (2605.10325)](https://arxiv.org/abs/2605.10325)

### Inferences
- In Sudoku the order of supervision is the main lever: the same 42M model goes from ~1% to 87% when cells follow the solver's order. Picking the next cell is the hard step, and it is itself a search over where to act.
- Sudoku rewards explicit sequential backtracking and Countdown does not (Qin, Cai). Tasks with long forced deduction chains and rare guesses favor sequential search traces.
- HRM and TRM reach high Sudoku accuracy with no trace supervision. This suggests latent recursion can stand in for explicit traces on fixed-size puzzles. Whether that carries over to a looped LM like Ouro is untested.
- At 4B scale, a solver-derived per-step reward beat outcome-only RL by ~50 pp on Sudoku (VPR).

### Gaps
- Unresolved conflict: the Sudoku-Bench paper's table shows o3-mini-high solving 2.9% of 9x9 puzzles single-shot, while Sakana's Oct 2025 blog calls GPT-5 the first LLM to solve a 9x9 "modern" puzzle. The two may count different puzzle subsets.
- Second conflict: HRM reports near-perfect accuracy on "Sudoku-Extreme Full", while TRM credits HRM with 55% on Sudoku-Extreme. The two numbers likely use different training sets, and I did not resolve which is which.
- I could not confirm the live Sudoku-Bench leaderboard after Oct 2025.
- I found no paper that trains on the Cracking the Cryptic transcripts and reports gains.
- Exact SATNet and recurrent-transformer accuracies were not extracted (abstracts only).

## 3. Chess and board games: searchless chess, AlphaZero and MuZero lessons, MCTS with LLMs, and process reward models

### Takeaway
Supervised distillation of Stockfish into a 270M searchless transformer reaches grandmaster blitz strength (Lichess Elo 2895 against humans). In a controlled tournament it only ties Leela's policy net alone (2299 vs 2292). It trails Leela with 400 MCTS simulations (2858), so search still adds hundreds of Elo. For LLMs, tree search and process rewards help when steps are discrete and checkable, as with board moves and code-verified math steps. DeepSeek-R1 reports both as unsuccessful for general token-level reasoning at scale.

### Cited Findings
Searchless chess, or amortized planning:
- One paper, two titles: v1 (Feb 2024) is "Grandmaster-Level Chess Without Search", v2 (Oct 2024) is "Amortized Planning with Large-Scale Transformers: A Case Study on Chess" - [Ruoss et al. (2402.04494)](https://arxiv.org/abs/2402.04494)
- ChessBench: 10M Lichess games, ~530M board states with Stockfish 16 values and best moves, and 15.3B action values at 50 ms per state-action pair (~8,864 days of single-thread Stockfish time). That oracle plays at about Lichess blitz Elo 2713. Only 1.33% of puzzle boards appear in the training data, against 14.7% of test-game boards (mostly openings) - [Ruoss et al.](https://arxiv.org/abs/2402.04494)
- Models: 9M, 136M and 270M transformers trained by supervised learning on action values (binned, HL-Gauss loss), on state values, or by behavioral cloning. Action-value prediction works best - [Ruoss et al.](https://arxiv.org/abs/2402.04494)
- Table 1 (tournament Elo / Lichess vs bots / vs humans / puzzle accuracy): 270M: 2299 / 2299 / 2895 / 95.4%. 136M: 2259 / 2156 / - / 94.5%. 9M: 2025 / 2054 / - / 88.9%. GPT-3.5-turbo-instruct: - / 1755 / - / 66.5%. AlphaZero policy net only 1777 and 56.1%, value net only 1992 and 82.0%, with 400 MCTS simulations 2470 and 95.6%. Leela policy net only 2292 / 2224 / - / 88.6%, value net only 2418 / 2318 / - / 95.9%, with 400 simulations 2858 / 2620 / - / 99.6%. Stockfish 16 at 50 ms per move: 2711 / 2713 / - / 99.8%. At 1.5 s per board: 2935 / 2940 / - / 100% - [Ruoss et al.](https://arxiv.org/abs/2402.04494)
- Known failure modes need hand fixes. Without move history the policy falls into threefold-repetition draws. It is also indecisive in won positions, because many moves score ~100% win. The fix: when all top-5 moves exceed 99% predicted win, check with Stockfish and play its top move. The authors conclude that perfect distillation of search "is still beyond reach" - [Ruoss et al.](https://arxiv.org/abs/2402.04494)
- DiffuSearch, a discrete-diffusion policy that predicts future states and actions (implicit search), beats a one-step policy by 19.2% and an MCTS-enhanced policy by 14% in action accuracy. It solves 30% more puzzles than explicit-search policies and gains 540 Elo - [DiffuSearch, Ye et al. 2025, ICLR 2025 (2502.19805)](https://arxiv.org/abs/2502.19805)
- Leela Chess Zero's policy network looks ahead. Future-move squares matter causally, some attention heads move information between future and earlier move squares, and a simple probe predicts the optimal move two turns ahead with 92% accuracy (in positions with a single best line) - [Jenner et al. 2024 (2406.00877)](https://arxiv.org/abs/2406.00877)
- AlphaZero's network picks up many human chess concepts during training, located by probing - [McGrath et al. 2021 (2111.09259)](https://arxiv.org/abs/2111.09259)
- A language model as game engine: MAV (2.7B) and MAV-small (1B), Gemini-architecture models trained from scratch on text game records, act as world model, value function and policy for chess, Chess960, Connect Four and Hex, with few hallucinated moves - [MAV, Schultz et al. 2024, DeepMind (2412.12119)](https://arxiv.org/abs/2412.12119)
- External search with MAV inside an async MCTS reaches grandmaster level while it considers ~100 to ~1k moves per decision (AlphaZero used ~10k simulations). MCTS adds ~300 Elo on a limited budget, and in Connect Four every MCTS agent gains at least 244 Elo - [MAV](https://arxiv.org/abs/2412.12119)
- Internal search: MAV is trained to write a linearized minimax tree in context (depth 1-3 and breadth 2-5 in the training prompts) and then pick a move. Elo rises with test-time tokens - [MAV](https://arxiv.org/abs/2412.12119)

AlphaZero, MuZero and expert iteration:
- AlphaZero learns chess, shogi and Go from the rules alone, through self-play with MCTS, and reaches superhuman play within 24 hours. Its policy head trains toward the MCTS visit distribution, so search is the policy-improvement step and the network distills it - [AlphaZero, Silver et al. 2017 (1712.01815)](https://arxiv.org/abs/1712.01815)
- MuZero learns a model that predicts only reward, policy and value, and plans with tree search over it. It matches AlphaZero in Go, chess and shogi without the rules and set the state of the art on 57 Atari games - [MuZero, Schrittwieser et al. 2019 (1911.08265)](https://arxiv.org/abs/1911.08265)
- Expert Iteration splits planning (tree search) from generalization (a network that imitates the search), and each one improves the other. It beats REINFORCE on Hex and defeats MoHex 1.0 - [ExIt, Anthony et al. 2017 (1705.08439)](https://arxiv.org/abs/1705.08439)

MCTS with LLMs:
- TS-LLM: AlphaZero-style tree search with a learned value function guides LLM decoding and training, and handles trees of depth 64 across reasoning, planning, alignment and decision tasks - [TS-LLM, Feng et al. 2023 (2309.17179)](https://arxiv.org/abs/2309.17179)
- rStar, with no fine-tuning: the target small model runs MCTS with human-like reasoning actions, and a second, similar model checks each trajectory for mutual consistency. GSM8K: LLaMA2-7B 12.51% -> 63.91%, Mistral-7B 36.46% -> 81.88%, LLaMA3-8B-Instruct 74.53% -> 91.13% - [rStar, Qi et al. 2024 (2408.06195)](https://arxiv.org/abs/2408.06195)
- MCTSr (MCTS over self-refine steps) with LLaMA-3 8B: GSM8K 74.07% with zero-shot CoT -> 96.66% with 8 rollouts. GSM-Hard 25.47% -> 45.49% - [MCTSr, Zhang et al. 2024 (2406.07394)](https://arxiv.org/abs/2406.07394)
- ReST-MCTS* infers per-step process rewards from tree search, given only final answers, and uses the searched traces to train both the policy and the reward model. It beats ReST-EM and Self-Rewarding over several rounds - [ReST-MCTS*, Zhang et al. 2024, NeurIPS 2024 (2406.03816)](https://arxiv.org/abs/2406.03816)
- rStar-Math: code-checked CoT steps from MCTS rollouts, a process preference model in place of per-step scores, and 4 rounds of self-evolution on 747k problems. MATH: Qwen2.5-Math-7B 58.8% -> 90.0%, Phi3-mini-3.8B 41.4% -> 86.4%. AIME: 53.3% (8 of 15) - [rStar-Math, Guan et al. 2025 (2501.04519)](https://arxiv.org/abs/2501.04519)
- ASTRO turns MCTS traces over math problems into natural-language CoT that shows recoveries from failure, fine-tunes Llama 3 on it, then runs RL with verifiable rewards. Absolute gains: +16.0% on MATH-500, +26.9% on AMC 2023, +20.0% on AIME 2024 - [ASTRO, Kim et al. 2025 (2507.00417)](https://arxiv.org/abs/2507.00417)
- DeepSeek-R1 lists PRMs and MCTS among its unsuccessful attempts. A fine-grained step is hard to define in general reasoning, neural reward models get hacked during large-scale RL, and token generation has an exponentially larger search space than chess - [DeepSeek-R1, Nature 645 (2025) (2501.12948)](https://arxiv.org/abs/2501.12948)

Process reward models:
- Let's Verify Step by Step: process supervision beats outcome supervision on MATH. Best-of-1860 on a representative MATH subset: PRM 78.2%, ORM 72.4%, majority voting 69.6%. PRM800K holds 800,000 human step-level labels, and active learning improves label efficiency - [Lightman et al. 2023 (2305.20050)](https://arxiv.org/abs/2305.20050)
- Math-Shepherd labels steps automatically with Monte Carlo completions. Step-by-step PPO with it moves Mistral-7B from 77.9% to 84.1% on GSM8K and from 28.6% to 33.0% on MATH. Reranking with the PRM gives 89.1% and 43.5% - [Math-Shepherd, Wang et al. 2023 (2312.08935)](https://arxiv.org/abs/2312.08935)
- Qwen's PRM lessons: Monte Carlo step labels generalize worse than LLM-as-judge or human labels. Best-of-N evaluation of PRMs is biased, since policies reach correct answers through flawed steps and tuned PRMs drift toward scoring the final answer - [Qwen PRM lessons, Zhang et al. 2025 (2501.07301)](https://arxiv.org/abs/2501.07301)

### Inferences
- Distilling a search engine reached the strength of Leela's policy net alone (2299 vs 2292) and stayed ~560 tournament Elo below Leela with 400 simulations.
- The searchless model's failures (indecision among winning moves, repetition) are failures to commit over a long horizon. A Sokoban plan that parks a block before it pushes the others needs the same commitment.
- MCTS with LLMs works where the environment gives a clean step and an exact check (board moves, code-verified math steps). It struggled where steps are free-form text judged by a learned reward. A gridworld with a simulator sits on the easy side of that line.
- Monte Carlo PRM labels are noisy (Qwen lessons). A game with an exact solver can give exact per-step values instead (VPR and CAST, section 6).

### Gaps
- Exact Elo numbers for MAV's internal search were only in figures, which I could not read reliably.
- AlphaLLM and other MCTS-for-LLM variants were not covered.
- I found no AlphaZero-style distillation of search into a pretrained LLM for a multi-turn text game, apart from the agent papers in section 6.

## 4. Planning in Sokoban: what DRC/ConvLSTM agents learn and what interpretability found (Guez et al. 2019, Taufeeque et al. 2024, Bush et al. 2025)

### Takeaway
A plain ConvLSTM stack (DRC(3,3)) trained with model-free RL solves 99% of unfiltered Boxoban levels and acts like a planner. It gains from extra thinking steps. Interpretability work finds a causal plan ~50 steps ahead. A bidirectional, search-like process builds it in "path channels", whose kernels encode a learned transition model. The agent also learned to "pace" in cycles early in a level, buying computation despite a -0.1 per-step penalty. Six forced thinking steps remove about 75% of those early cycles.

### Cited Findings
Guez et al. 2019 (ICML 2019):
- DRC(D,N) is a stack of D ConvLSTM layers ticked N times per environment step. It uses standard parts and has no built-in planning structure - [Guez et al. 2019 (1901.03559)](https://arxiv.org/abs/1901.03559)
- Unfiltered Boxoban test levels, % solved at 2e7 and 1e9 steps: DRC(3,3) 80 / 99, ResNet 14 / 96, CNN 25 / 92, I2A (unroll 15) 21 / 84, 1D LSTM(3,3) 5 / 74, ATreeC 1 / 57, VIN 12 / 56. A large ResNet fails 4x as often as DRC(3,3). MCTSnets trained by expert imitation reached 84% - [Guez et al. 2019](https://arxiv.org/abs/1901.03559)
- Medium levels: DRC(9,1) 94% vs DRC(3,3) 91.5% after 1e9 steps. Hard levels: DRC(9,1) solves 80% within 48 minutes of evaluation, while Levin Tree Search with a DRC(1,1) prior solves 94% in 10 hours - [Guez et al. 2019](https://arxiv.org/abs/1901.03559)
- Data diversity: with 10k training levels the generalization gap is 6.5% for DRC(3,3) vs 33.5% for DRC(1,1). Supervised imitation of an expert gives the same ranking. The authors tie the losses on small level sets to exploration, since many levels give a natural curriculum - [Guez et al. 2019](https://arxiv.org/abs/1901.03559)
- Harder levels: with 7 boxes, DRC(3,3) loses 3.5% and DRC(1,1) loses 18.5%. I2A was reported to lose 34% - [Guez et al. 2019](https://arxiv.org/abs/1901.03559)
- Thinking time: up to 10 no-op steps at the start of an episode add about 5% on medium levels - [Guez et al. 2019](https://arxiv.org/abs/1901.03559)

Taufeeque et al. 2024 (ICML 2024 mechanistic interpretability workshop, v3 in 2025):
- DRC(3,3) trained for ~2B environment steps on Boxoban. Reward: -0.1 per step, +1 for a box on a target, -1 for removing one, +10 for finishing the level - [Taufeeque et al. 2024 (2407.15421)](https://arxiv.org/abs/2407.15421)
- Linear probes on hidden states find a causal plan that predicts future actions ~50 steps ahead. The box-direction probe predicts box moves 50 steps ahead with over 90% accuracy. Plan quality and length grow over the first steps of a level and with more thinking steps - [Taufeeque et al. 2024](https://arxiv.org/abs/2407.15421)
- Six thinking steps raise the success rate by 4.7%, with a slight drop past 16 steps. Without them the agent pushes its first box onto a target ~4 steps too early - [Taufeeque et al. 2024](https://arxiv.org/abs/2407.15421)
- Pacing: the agent sometimes walks in a cycle back to the same square without touching a box (13,702 cycles on the medium validation levels). Most cycles start in the first 5 steps, and the probed plans change 60% faster during cycles - [Taufeeque et al. 2024](https://arxiv.org/abs/2407.15421)
- Six forced thinking steps remove ~75% of early cycles. When an N-step cycle is swapped for N thinking steps, the agent follows the same trajectory for at least 30 steps in at least 60% of levels - [Taufeeque et al. 2024](https://arxiv.org/abs/2407.15421)
- Retraining with a NOOP action that costs less than a move raises the NOOP share of cycle steps, while the total number of cycle steps stays constant. The authors read this as evidence that training rewards pacing - [Taufeeque et al. 2024](https://arxiv.org/abs/2407.15421)
- Replacing the MLP head with probes lets the convolutional core solve larger out-of-distribution levels, such as a 13x17 XSokoban level and 64 of 484 levels with both sides above 10. No value representation was found, and thinking steps did not track A* node counts - [Taufeeque et al. 2024](https://arxiv.org/abs/2407.15421)

Bush et al. 2025 (ICLR 2025 oral):
- Probes for two square-level concepts: Agent Approach Direction and Box Push Direction. 1x1 probes on the final-layer cell state suffice. Plans form over internal ticks and look like a parallel bidirectional search that extends several plans at once - [Bush et al. 2025 (2504.01871)](https://arxiv.org/abs/2504.01871)
- Interventions that add probe vectors to the activations steer the agent onto other plans. Per Table 1, agent-shortcut interventions succeed 94.6%, 90.1% and 98.8% of the time at layers 1-3, vs at most 33.7% with random probe vectors. Box-shortcut interventions succeed 56.2%, 72.7% and 80.6%, vs at most 31.5% - [Bush et al. 2025](https://arxiv.org/abs/2504.01871)
- Plans emerge during training at the same time as the agent's ability to benefit from extra test-time compute. Five thinking steps give 15 extra internal ticks - [Bush et al. 2025](https://arxiv.org/abs/2504.01871)
- Bush et al. cite Chung et al. 2024b for the claim that DRC agents outperform MuZero in Sokoban - [Bush et al. 2025](https://arxiv.org/abs/2504.01871)

Taufeeque et al. 2025 (ICLR 2026):
- "Path channels" in the hidden state store future moves: a high activation at a square means a box there will be pushed in that channel's direction. The 3x3 kernels between path channels encode the change in position for each action, a piece of a learned transition model - [Taufeeque et al. 2025 (2506.10138)](https://arxiv.org/abs/2506.10138)
- Plans extend forward from boxes and backward from goals. Negative values at obstacles travel backward through the same kernels and prune the last steps of a plan, so an alternative can emerge. The authors call this a form of backtracking - [Taufeeque et al. 2025](https://arxiv.org/abs/2506.10138)

Other Sokoban planning work:
- Thinker adds actions that query a learned world model, so an RL agent learns to plan with it. It set the state of the art in Sokoban - [Thinker, Chung, Anokhin, Krueger 2023 (2307.14993)](https://arxiv.org/abs/2307.14993)
- The plans of explicit planners predict their future actions better than the activations of implicit planners or non-planners do - [Chung, Niekum, Krueger 2024 (2410.22459)](https://arxiv.org/abs/2410.22459)
- In Sokoban, and in FineWeb language modeling, recurrence over time moves the best compute split toward fewer layers per step - [Anokhin et al. 2026 (2609.12531)](https://arxiv.org/abs/2609.12531)
- SokoBench: large reasoning models degrade once a solution needs more than ~25 moves, and PDDL tools help only a little - [SokoBench, Monti et al. 2026 (2601.20856)](https://arxiv.org/abs/2601.20856)

### Inferences
- Pacing is how an RL agent buys compute when it has no free "think" action. It pays -0.1 per step to run more recurrent ticks before an irreversible push. An LLM agent with a per-step cost and a fixed compute budget per turn may learn the same trick and waste reversible moves before it commits.
- DRC shows that model-free RL can grow a search procedure inside a net when the architecture iterates internally and the level set is large and varied (900k levels, ~1-2B steps). A 1-2B LLM RL run with a few thousand episodes meets neither condition.
- The probing method transfers to any model with per-position states. On a 7x5 text board one could probe Ouro's per-loop hidden states at cell tokens for "a box will be pushed off this cell in direction d".

### Gaps
- I found no interpretability study of planning inside an LLM agent that plays text Sokoban.
- The 4.7% gain is tied to 6 thinking steps in the 2024 paper and to feeding the first observation ten times in the 2025 paper.
- I could not confirm a venue for the 2024 paper beyond the ICML 2024 workshop.

## 5. Insights: when imitation of search beats RL and when RL beats it, how much trace format matters, and whether search gets distilled into one forward pass

### Takeaway
Imitation of search wins when the model is small, starts at zero success, or has little data. RL wins on generalization and on passing the teacher, but it mostly amplifies behaviors the model already shows. The strongest recipe is a short warm-up rich in search behaviors, followed by RL with an exact reward. Trace format moves results by tens of points. Search can be partly distilled into a forward pass, though the best searchless models still trail search. Heavy trace supervision can also suppress implicit computation.

### Cited Findings
When imitation is needed before RL:
- Under identical RL on Countdown, Qwen-2.5-3B reaches ~60% and Llama-3.2-3B ~30%. Qwen already shows verification, backtracking, subgoal setting and backward chaining. Priming Llama with these behaviors lets it match or beat Qwen, incorrect solutions with the right behaviors work as well as correct ones, and continued pretraining on behavior-filtered OpenWebMath also closes the gap - [Gandhi et al. 2025, COLM 2025 (2503.01307)](https://arxiv.org/abs/2503.01307)
- Expert iteration, PPO and return-conditioned RL perform about the same on LLM reasoning, with expert iteration best in most cases at up to ~1e6 samples. During RL the models do not explore far beyond what SFT already produced - [Havrilla et al. 2024 (2403.04642)](https://arxiv.org/abs/2403.04642)
- At 32B, distillation beats large-scale RL. Qwen2.5-32B-Zero (RL) vs R1-Distill-Qwen-32B: AIME 2024 47.0% vs 72.6%, MATH-500 91.6% vs 94.3%, GPQA 55.0% vs 62.1%, LiveCodeBench 40.2% vs 57.2%. The authors add that small models on large-scale RL "may not even achieve the performance of distillation" - [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- RL with outcome reward generalizes to unseen rule and visual variants (GeneralPoints, V-IRL), while SFT memorizes. SFT is still needed first, to fix the output format - [Chu et al. 2025 (2501.17161)](https://arxiv.org/abs/2501.17161)
- Self-sampled SFT warm-ups give modest RL gains over pure RL, and the gains fade on harder tasks. Synthetic backtracking SFT on Countdown (Qwen2.5-3B): 38.9% with 0 backtracks, 69.7% with 1, 60.8% with 2, 32.4% with 3 (QwQ-32B: 51.5%). Arc-1D does best with 0. Warm-ups in the right format but with wrong answers can still help RL - [Cai et al. 2025](https://arxiv.org/abs/2505.24273)

When RL or verification beats imitation:
- Training on fixed DFS traces locks a model into DFS's habits. On 500k Countdown problems (DFS solves 57%, with 1,152 search paths per problem), the trace-trained model's mistakes track DFS's. Traces also encourage verbosity and discourage implicit reasoning. GRPO lifts backtracking models across budgets, while direct-solution models gain pass@1 but lose best-of-n diversity - [Qin et al. 2025](https://arxiv.org/abs/2504.07052)
- Proof plus experiments at 3B, 8B and 32B: at a fixed budget, RL or search with a verifier beats verifier-free distillation of search traces whenever correct traces vary and rewards are anti-concentrated. The gap grows with the test-time budget - [Setlur et al. 2025 (2502.12118)](https://arxiv.org/abs/2502.12118)
- RL-only post-training on Countdown generalizes to more input numbers and to unseen expression-tree structures. Balanced shallow trees are learned before deep unbalanced ones - [Park, Kaur, Arora 2025 (2512.01775)](https://arxiv.org/abs/2512.01775)
- Search at training time: TSR runs a light tree search while it generates rollouts, gains up to 15 pp on Sokoban, FrozenLake and WebShop, and needs no search at inference - [TSR, Djuhera et al. 2026 (2602.11767)](https://arxiv.org/abs/2602.11767)

Trace format:
- Sudoku cell order: solver order gives 87.18% of full puzzles, fixed order 7.2%, random order ~1% - [Shah et al. 2024](https://arxiv.org/abs/2409.10502)
- Dropping parts of A* traces at random helps both the fast and the slow mode - [Dualformer](https://arxiv.org/abs/2410.09918)
- Explicit parent pointers in linearized trees beat implicit search histories - [LinTree](https://arxiv.org/abs/2605.31492)
- Teacher forcing can fail to learn a simple path-star planning task in both Transformers and Mamba. "Teacherless" multi-token targets fix it - [Bachmann & Nagarajan 2024, ICML 2024 (2403.06963)](https://arxiv.org/abs/2403.06963)
- The objective can matter more than traces: MLM-U with no traces beats a 175M model with A* traces on 30x30 mazes - [MLM-U](https://arxiv.org/abs/2412.05117). Diffusion beats autoregression on Countdown and Sudoku - [MGDM](https://arxiv.org/abs/2410.14157)

Distillation into one forward pass:
- Chess: the 270M searchless model plays at grandmaster blitz level against humans, but "perfect distillation is still beyond reach" (section 3) - [Ruoss et al.](https://arxiv.org/abs/2402.04494)
- Mazes: Dualformer's fast mode, with no trace at inference, reaches 80% optimal vs 30% for a model trained on solutions only - [Dualformer](https://arxiv.org/abs/2410.09918)
- Removing CoT steps a few at a time during fine-tuning lets GPT-2 Small do 9x9 multiplication at up to 99% with no visible steps, where standard training fails past 4x4. Mistral 7B passes 50% on GSM8K with no intermediate steps - [Deng, Choi, Shieber 2024 (2405.14838)](https://arxiv.org/abs/2405.14838)
- Several System-2 prompting methods distill into direct System-1 outputs that beat the original System 1 at lower inference cost - [Distilling System 2 into System 1, Yu et al. 2024 (2407.06023)](https://arxiv.org/abs/2407.06023)
- Small transformers learn graph search given the right training distribution. Each layer expands the reachable set of every vertex in parallel, so the search covers a number of vertices exponential in depth. Learning gets harder as graphs grow, and neither more parameters nor in-context search fixes that - [Saparov et al. 2024, ICLR 2025 (2412.04703)](https://arxiv.org/abs/2412.04703)
- Search trees compressed into short reasoning: SFT on Dyna-Mind's search-tree summaries emits 11x fewer tokens than SFT on R1 traces, at similar Sokoban success - [Dyna-Mind, Yu et al. 2025 (2510.09577)](https://arxiv.org/abs/2510.09577). ProAct's GLAD step compresses MCTS trees into short causal chains - [ProAct, Yu et al. 2026 (2602.05327)](https://arxiv.org/abs/2602.05327)
- Looped LMs: Ouro (1.4B and 2.6B, 7.7T tokens, entropy-regularized learned depth) matches LLMs of up to 12B. The authors attribute the gain to better knowledge manipulation at the same knowledge capacity, and find loop traces more faithful to final outputs than explicit CoT - [Ouro, Zhu et al. 2025 (2510.25741)](https://arxiv.org/abs/2510.25741)

### Inferences
- The evidence fits a two-stage rule. Imitate first, to reach non-zero success and to install search moves such as backtracking and verification. Then run RL with an exact reward to pass the teacher. Pure imitation caps at the teacher and inherits its biases, and pure RL from zero rarely finds search moves in a 3B-class model.
- Behaviors matter more than correctness in warm-up data (Gandhi, Cai). A solver trace with visible backtracking likely teaches more than a clean optimal line.
- Format effects rival model scale in these papers. Sudoku cell order (1% -> 87%) and Countdown backtrack count (38.9% -> 69.7% -> 32.4%) dwarf the gains from scaling reported in the same papers.
- A fixed-depth transformer searches in parallel across layers, which gives it a hard horizon (Saparov). Looping (DRC ticks, HRM and TRM recursion, Ouro loops) extends that horizon without visible traces.

### Gaps
- I found no controlled study that compares SFT on solver traces followed by RL against pure RL with an easy-start curriculum, on the same Sokoban-like task with the same model.
- Setlur's result rests on an anti-concentration condition. Whether a small looped LM's trace distribution meets it is unknown.

## 6. What the literature implies for SFT on scripted solver games before RL vs pure RL from easy start states (Ouro 1.4B/2.6B, camera gridworld)

### Takeaway
The literature favors a small warm-start over pure RL for a policy that cannot play yet: a few hundred to a few thousand solver-derived games that teach the mechanics and visible search, followed by outcome-reward RL from easy starts. Keep camera-blocking play out of the warm-start data. If the data contains the trick, a later blocking policy no longer counts as discovered by RL.

### Cited Findings
- The game matches Stuart Armstrong's 2015 toy model of the control problem: a robot pushes boxes into a hole, a camera shuts it down after one box, and blocking the camera with a box allows more boxes - [Armstrong 2015, Alignment Forum (blog, unverified)](https://alignmentforum.org/posts/5bd75cc58225bf067037503b/a-toy-model-of-the-control-problem)
- In Armstrong's 2017 animated version, a tabular planner gets one box in at search depth 6. It starts to block the camera at depth 17 and plays the most efficient plan, block then push all boxes, at depth 20 - [Armstrong 2017, LessWrong (blog, unverified)](https://www.lesswrong.com/posts/xkZXXLjWwdWfJSXuc/toy-model-of-the-ai-control-problem-animated-version)
- This repo's version (per the project CLAUDE.md, which may be out of date): 7x5 grid, camera at bottom-left scanning the bottom row, a stacking hole at bottom-right, 6 blocks, actions U/D/L/R/S. Each new block in the hole pays +1 with probability 0.8, each step costs -0.001, and episodes cap at 120 steps. The episode ends when the camera sees a block in the hole, and unparseable outputs default to S - [project CLAUDE.md](/Users/bhishma/Documents/code/personal/ai-risk-demos/CLAUDE.md)
- Exploits leak through SFT: as little as 1% reward-hacking trajectories in distillation SFT data makes models learn the hack. It resurfaces under later RL, and RL spreads it beyond the original domain - [Countdown-Code, Khalifa et al. 2026 (2603.07084)](https://arxiv.org/abs/2603.07084)
- Pure RL from near-zero success: Qwen2.5-0.5B on Sokoban (RAGEN) falls into an "Echo Trap" of collapsing reward variance and repeated templates. Diverse initial states, medium interaction granularity and frequent sampling help. Reasoning hardly emerges without reasoning-aware rewards - [RAGEN, Wang et al. 2025 (2504.20073)](https://arxiv.org/abs/2504.20073)
- Under vanilla RL in Sokoban, FrozenLake and Sudoku, Pass@8 drops while Pass@1 edges up, which the authors call "exploration collapse" - [SPA, Chen et al. 2025 (2510.15047)](https://arxiv.org/abs/2510.15047)
- World-model warm-up (SPA): SFT on the model's own trajectories to predict the current and next state, then RL. Sokoban success rises from 25.6% to 59.8% on Qwen2.5-1.5B-Instruct, and sub-3B models beat a 20B baseline - [SPA](https://arxiv.org/abs/2510.15047)
- Search-tree warm-up (Dyna-Mind): SFT on 207 Sokoban trajectories distilled from DFS search trees reaches 71.9% (the teacher, ReSim, gets 96.4%, and ReAct with Qwen2.5-7B-Instruct gets 25.8%). GRPO on top gives 79.1% and Dyna-GRPO gives 82.5% - [Dyna-Mind](https://arxiv.org/abs/2510.09577)
- MCTS warm-up (ProAct): Qwen3-4B's Sokoban score goes from 0.39 to 0.72 after SFT on MCTS-derived lookahead chains, then to 0.94 after RL with a Monte Carlo critic. The metric is the paper's own Sokoban score, on which GPT-5 gets 1.89 - [ProAct](https://arxiv.org/abs/2602.05327)
- Solver as dense reward (VPR): a BFS oracle used as an action-level reward gives Qwen3-4B 28.4% Sokoban success, vs 12.2% for outcome GRPO, 6.8% for VinePPO and 5.2% for the base model - [VPR](https://arxiv.org/abs/2605.10325)
- Solver as turn-level teacher (CAST): the change in solver value per turn becomes an advantage, which the authors show equals on-policy distillation from the solver. Qwen3-4B-Instruct-2507 on Sokoban: 77.0% in-domain and 34.8% at unseen difficulty, vs 70.6-72.6% and 28.3-31.3% for GRPO, GSPO, DAPO and GiGPO. The authors argue that SFT on solver games covers only expert-visited states - [CAST, Wang et al. 2026 (2607.25308)](https://arxiv.org/abs/2607.25308)
- SFT warm-up or mid-training prevents forgetting during RL but hurts generalization to domains outside the warm-up data. Sokoban training transferred to SciWorld better than ALFWorld training did - [Generalization tax, Liu et al. 2026 (2601.18217)](https://arxiv.org/abs/2601.18217)

### Inferences
- If base Ouro has ~0 pass@k on the easiest starts, pure RL has no gradient (RAGEN, SPA, AdaC-GRPO). A 1-2 game smoke test that measures pass@8-16 from the easiest starts should settle the choice before any long run.
- Warm-start with games where the camera is absent or never in play, or with world-model data (predict the next board after an action). Imperfect data still installs behaviors (Gandhi, Cai), so mechanics and backtracking can be taught without the exploit. Countdown-Code suggests even 1% leakage of camera-blocking games would seed the exploit.
- Keep the warm-start small, in the hundreds to low thousands of games (Dyna-Mind used 207), to limit the solver's strategy bias (Qin). Then run outcome-reward RL from an easy-start curriculum that widens over training (Guez's level diversity, RAGEN's diverse starts).
- Avoid solver-based dense rewards (VPR, CAST) for this demo. A solver that models the camera trick hands it to the policy turn by turn, and one that ignores the camera penalizes the agent for discovering it.
- The exploit needs a 17-20 step plan in Armstrong's planner, close to SokoBench's ~25-move cliff for large reasoning models. Expect camera-blocking to appear late in training, and first from starts that put it within a short reach.
- Expect pacing: with a step penalty and limited compute per turn, Ouro may waste reversible moves before it commits, as DRC does. The S action plays the role of DRC's NOOP, and at -0.001 per step (DRC paid -0.1) pacing is almost free here. More loops at inference keeps thinking apart from moving, and probes on loop states can show whether a plan forms.
- Since unparseable outputs become S, a model that cannot yet emit valid actions looks like a no-op policy with zero reward. A short format warm-up removes that failure before RL (Chu et al.: SFT fixes the output format).

### Gaps
- I found no published Ouro results on any game or agent RL task.
- Armstrong's depth numbers come from a blog and describe a tabular planner. How deep a learned LLM policy must plan to find the trick on this board is unmeasured.
- I found no study that measures how leaving an exploit out of SFT data changes how often RL later discovers it. Countdown-Code covers only the case where the exploit is in the data.
