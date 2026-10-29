# RLVR on puzzles and games: how RL with verifiable rewards trains pretrained LLMs on single-turn puzzles and multi-turn game environments

*Current as of 29 Oct 2026. Sources are arXiv papers unless marked. "[unverified]" marks blogs, secondary summaries or search snippets I could not check against the primary text. Numbers come from the paper versions I fetched. Where versions disagree, both are given.*

## 1. DeepSeek-R1-Zero and DeepSeek-R1: what emerged under pure RL, why R1 added cold-start SFT, and what the paper says about small models

### Takeaway
DeepSeek-R1-Zero applied pure GRPO to DeepSeek-V3-Base with only rule-based accuracy and format rewards. It produced long chains of thought, reflection and a sudden spike in "wait" (the "aha moment"). AIME 2024 pass@1 rose from 15.6% to 77.9%. The outputs were hard to read and mixed languages, so R1 adds a few thousand curated long-CoT "cold-start" examples before RL. For small models the paper's own verdict is that distillation beats RL: 10K+ RL steps on Qwen-32B-Base reached 47.0 on AIME 2024, while SFT-only distillation reached 72.6.

### Cited Findings
**R1-Zero recipe (latest arXiv version)**

- Base model DeepSeek-V3-Base, no SFT before RL. GRPO with 16 samples per question, learning rate 3e-6, KL coefficient 0.001, rollout temperature 1. Max length was 32,768 tokens, raised to 65,536 after step 8.2k. Training ran 10,400 steps (about 1.6 epochs), with 32 questions per step (batch 512), and the reference model was replaced every 400 steps - [DeepSeek-R1, arXiv 2501.12948 (latest HTML)](https://arxiv.org/html/2501.12948)
- Rewards are rule-based accuracy plus format, at equal weight. Neural reward models were avoided because they are prone to reward hacking in large-scale RL and need costly retraining. The template asks for reasoning inside think tags, then the answer inside answer tags - [DeepSeek-R1 (latest HTML)](https://arxiv.org/html/2501.12948)
- AIME 2024 pass@1 went from 15.6% to 77.9%, and 86.7% with self-consistency - [DeepSeek-R1 (latest HTML)](https://arxiv.org/html/2501.12948). The January 2025 v1 instead reports R1-Zero at 71.0 pass@1 and 86.7 cons@64 "after thousands of RL steps". **Version discrepancy:** cite the version you use - [DeepSeek-R1 v1](https://arxiv.org/html/2501.12948v1)

**What emerged**

- Response length grew steadily, from hundreds to thousands of tokens, with no external intervention. Reflection and exploration of alternative solutions emerged. The "aha moment" showed up in an intermediate R1-Zero checkpoint as a sudden rise in the word "wait". The model wrote "Wait, wait. Wait. That's an aha moment I can flag here." - [DeepSeek-R1 (latest HTML)](https://arxiv.org/html/2501.12948)

**Why cold-start SFT**

- R1-Zero had poor readability and mixed English and Chinese - [DeepSeek-R1 (latest HTML)](https://arxiv.org/html/2501.12948)
- v1 calls readability a key limitation of R1-Zero. It collects "thousands" of cold-start examples, formatted with a summary at the end and filtered for unreadable responses. With this human-designed pattern the authors saw better performance than R1-Zero - [DeepSeek-R1 v1](https://arxiv.org/html/2501.12948v1)

**R1 pipeline**

- The stages are:
  1. Cold-start SFT.
  2. Reasoning RL with a language-consistency reward (the share of target-language words in the CoT). In ablation it cost slightly in accuracy but was kept for readability.
  3. Rejection sampling and SFT on 804,745 samples (about 600k reasoning, about 200k non-reasoning).
  4. A second RL stage for helpfulness and harmlessness, with reward models trained on 66,000 helpfulness preference pairs and 106,000 safety prompts.
- From R1-Zero to R1, AIME 2024 went from 77.9 to 79.8 and MATH-500 from 95.9 to 97.3. The authors note the SFT data is mostly single-turn, which may limit multi-turn ability - [DeepSeek-R1 (latest HTML)](https://arxiv.org/html/2501.12948)

**Distillation vs RL for small models (v1, Table 6)**

- Scores are AIME 2024 pass@1 / AIME cons@64 / MATH-500 / GPQA-Diamond / LiveCodeBench:
  - QwQ-32B-Preview: 50.0 / 60.0 / 90.6 / 54.5 / 41.9
  - DeepSeek-R1-Zero-Qwen-32B (over 10K RL steps on Qwen-32B-Base): 47.0 / 60.0 / 91.6 / 55.0 / 40.2
  - DeepSeek-R1-Distill-Qwen-32B: 72.6 / 83.3 / 94.3 / 62.1 / 57.2
  - Source: [DeepSeek-R1 v1](https://arxiv.org/html/2501.12948v1)
- The paper draws two conclusions. Distilling a stronger model into a smaller one works very well, while small models that rely on large-scale RL need enormous compute and may still fall short of distillation. Going further may need stronger base models and larger-scale RL - [DeepSeek-R1 v1](https://arxiv.org/html/2501.12948v1)
- The distilled models used SFT only, with no RL stage. AIME 2024 pass@1: 1.5B 28.9, 7B 55.5, 14B 69.7, 32B 72.6, Llama-8B 50.4, Llama-70B 70.0 - [DeepSeek-R1 v1](https://arxiv.org/html/2501.12948v1)

**Failed attempts and limitations**

- Process reward models failed for three reasons: fine-grained steps are hard to define, intermediate correctness is hard to judge, and model-based PRMs invite reward hacking and add retraining cost. MCTS failed because the token search space is far larger than in chess, capped node expansion gets stuck in local optima, and fine-grained value models are hard to train - [DeepSeek-R1 v1](https://arxiv.org/html/2501.12948v1)
- Limitations: few-shot prompting consistently degrades performance (zero-shot is recommended). Queries in languages other than Chinese and English get mixed-language answers. There is little gain on software engineering because evaluation is slow - [DeepSeek-R1 v1](https://arxiv.org/html/2501.12948v1)

**Challenges to the "emergence" story**

- DeepSeek-V3-Base already produces self-reflection ("Aha", "wait"), and nearly all base models show it. Self-reflection is not positively correlated with accuracy: nearly half of the responses with self-reflection do not score higher - [Dr. GRPO, arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- Logic-RL found no sudden jump. Reflective words rose steadily, so the authors conclude there is probably no distinct aha moment - [Logic-RL, arXiv 2502.14768](https://arxiv.org/abs/2502.14768)
- SimpleRL-Zoo ran zero-RL on 10 base models: Llama3-8B, Mistral-7B and 24B, DeepSeek-Math-7B, Qwen2.5-Math-7B, and Qwen2.5 from 0.5B to 32B. Longer responses do not always come with verification behaviors. The authors report the aha moment for the first time in small non-Qwen models, and name format-reward adjustment and query-difficulty control as key to success (COLM 2025) - [SimpleRL-Zoo, arXiv 2503.18892](https://arxiv.org/abs/2503.18892)

**Small models and the limits of RL**

- In TinyZero, Qwen2.5-0.5B base fails to learn reasoning on Countdown, while 3B and larger develop self-verification and search - [TinyZero repo](https://github.com/Jiayi-Pan/TinyZero)
- The "Small Model Learnability Gap": models at or below about 3B do not consistently benefit from long CoT or from distilling much larger models. They do better on shorter, simpler chains, and Mix Distillation (long plus short CoT, or large plus small teachers) helps - [Li et al., arXiv 2502.12143](https://arxiv.org/abs/2502.12143)
- RLVR beats the base model at small k, but the base wins at large k on pass@k. Six RLVR algorithms performed similarly, and distillation, unlike RL, can introduce new reasoning patterns (NeurIPS 2025 oral) - [Yue et al., arXiv 2504.13837](https://arxiv.org/abs/2504.13837)
- DeepSeekMath had already found that RL improves Maj@K but not Pass@K - [DeepSeekMath, arXiv 2402.03300](https://arxiv.org/abs/2402.03300)
- Counterpoint: ProRL (KL control, periodic reference-policy resets, diverse tasks) reports RL beating the base across pass@k, including tasks where the base fails at every k. Gains track the base model's competence and the training length - [ProRL, arXiv 2505.24864](https://arxiv.org/abs/2505.24864)

### Inferences
- Emergence under pure RL happened in a very large, strongly pretrained model. At small scale the R1 authors themselves recommend distillation. For a 1-4B policy, expect RL to sharpen behaviors the model already samples rather than invent new strategies (consistent with Yue et al. and DeepSeekMath).
- Counting "wait" tokens or watching length grow is weak evidence of better reasoning. Base models already reflect, and reflection does not predict accuracy.

### Gaps
- I did not separately check the Nature version. The hyperparameters above come from the latest arXiv HTML, which appears to carry the expanded methods.
- DeepSeek reports no RL-from-scratch experiment below 32B, so it gives no direct RL-vs-distillation evidence at 1.5-7B.

## 2. Puzzle RLVR: Logic-RL, TinyZero/Countdown, Reasoning Gym, Enigmata, SynLogic, Sudoku, mazes, chess - transfer and reasoning length

### Takeaway
Synthetic puzzles with exact verifiers reliably train in-domain skill and lengthen CoT. Transfer to math is real but conditional:
- Large relative gains from weak baselines (Logic-RL: +125% relative on AIME) and +9.7 MATH for a 3B trained on Reasoning Gym math.
- Near-zero transfer at scale: Enigmata-32B AIME 62.0 to 60.6, and Seed1.5-Thinking +0.4 to +1.9 points.
- Puzzle-only RL leaves math far behind math-trained models (SynLogic-32B AIME 19.6).

Domains with little pretraining exposure (logic, simulation, chess) need in-domain data and often plateau at the limit of what the base model knows. Game-type puzzles are the hardest category for small models (Reasoning Gym games: 0.0 to 3.3 for a 3B).

### Cited Findings
#### Logic-RL (Knights and Knaves)
- Base model Qwen2.5-7B-Instruct-1M, trained on fewer than 5,000 synthetic K&K puzzles with 3-7 people for training and 2-8 for evaluation, where 8 is out of distribution. The algorithm is a modified REINFORCE++, which beat GRPO on nearly all metrics. PPO was 138% slower. Settings: batch 8, 8 rollouts, KL 0.001, max response 4,096 tokens, constant learning rate 4e-7, temperature 0.7, 3,600 steps, gamma = 1 - [Logic-RL, arXiv 2502.14768](https://arxiv.org/abs/2502.14768)
- The strict format reward gives +1 for correct format and -1 otherwise, with each tag exactly once and in order. The answer reward gives +2 for a full match, -1.5 for a partial mismatch, and -2 when unparsable. Observed reward hacks included skipping the think block, reasoning inside the answer tag, repeated guessing and repeating the question. Fixes were iterative rule refinement, the strict format check, and ending the prompt with a think tag - [Logic-RL](https://arxiv.org/abs/2502.14768)
- K&K average accuracy rose from 0.19 to 0.89, and on 8-person puzzles from 0.01 to 0.67. For comparison, o3-mini-high scored 0.94, DeepSeek-R1 0.81 and GPT-4o 0.37. Out-of-distribution gains relative to the base were +125% on AIME and +38% on AMC. The paper gives no absolute scores - [Logic-RL](https://arxiv.org/abs/2502.14768)
- Length went from about 500 to about 2,000 tokens after about 1k steps, yet longer responses did not guarantee better reasoning. Language mixing hurt reasoning. "Verify" and "re-evaluate" correlated with higher scores, while "recheck" correlated with lower ones. Rejection-sampling fine-tuning memorized (LiMem rose) while RL generalized. Curriculum gave negligible benefit, and base vs instruct curves were nearly identical - [Logic-RL](https://arxiv.org/abs/2502.14768)

#### TinyZero and Countdown
- TinyZero reproduces R1-Zero on Countdown and multiplication with veRL. It costs under $30, using 1 GPU up to 1.5B and 2 GPUs for 3B and up. Qwen2.5-0.5B base fails, and 3B and up learn self-verification and search. The repo is now deprecated in favour of veRL [unverified: repo README, not peer-reviewed] - [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- Under identical RL on Countdown, Qwen-2.5-3B far outperforms Llama-3.2-3B. Qwen naturally shows verification, backtracking, subgoal setting and backward chaining, and Llama does not. Priming Llama with examples of these behaviors lets it match or beat Qwen, even when the priming solutions are wrong. Continued pretraining on behavior-filtered OpenWebMath does the same - [Gandhi et al., arXiv 2503.01307](https://arxiv.org/abs/2503.01307)

#### Reasoning Gym
- 102 procedurally generated datasets with verifiers, in 10 categories: Algorithms 32, Cognition/ARC 14, Arithmetic 11, Algebra 6, Graphs 6, Logic 7, Games 18, Geometry 4, Code 2, Induction 2. The 18 games include sokoban, rush_hour, mini_sudoku, sudoku, countdown, n_queens, tower_of_hanoi, tsumego and zebra_puzzles. Maze sits under Graphs. Difficulty is adjustable - [Reasoning Gym, arXiv 2505.24760](https://arxiv.org/abs/2505.24760)
- Zero-shot on hard configurations: o3-mini 63.5%, DeepSeek-R1 59.5%, Grok 3 Mini 55.1%, Llama 4 Maverick 41.5%, Claude 3.5 Sonnet 40.3%, Gemma 3 27B 20.3% - [Reasoning Gym](https://arxiv.org/abs/2505.24760)
- RLVR used Qwen2.5-3B-Instruct with GRPO, about 1,500 A6000 GPU-hours. In-domain Acc@3: Algebra 5.0 to 16.7, Algorithmic 52.3 to 59.7, Arithmetic 89.7 to 96.0, Cognition 40.3 to 42.3, **Games 0.0 to 3.3** - [Reasoning Gym](https://arxiv.org/abs/2505.24760)
- Cross-domain: training on algorithmic tasks lifted Algebra from 23.83 to 52.89 and Geometry from 0.83 to 23.17. Training on games lifted Algebra by 21.8 and Cognition by 13.1. ARC fell by 2.3 after algebra training and 2.2 after games training - [Reasoning Gym](https://arxiv.org/abs/2505.24760)
- External benchmarks after RG-Math training: MATH 48.5 to 58.2, GSM8K 76.2 to 76.7, Big-Bench Hard 8.68 to 16.34, MMLU-Pro Math 54.63 to 60.25 - [Reasoning Gym](https://arxiv.org/abs/2505.24760)
- Curriculum (advance a level when accuracy exceeds 70% over 20 steps) vs none: Mini Sudoku 6.67 vs 20.00, Spell Backwards 30.00 vs 70.67, Count Primes 4.00 vs 30.67 - [Reasoning Gym](https://arxiv.org/abs/2505.24760)

#### Enigmata
- 36 tasks in 7 categories: Crypto, Arithmetic, Logic, Grid, Graph, Search, Sequential. Each has a generator (30 custom) and a manually validated verifier, with Easy/Medium/Hard levels set from GPT-4o pass@k. Enigmata-Eval has 4,758 instances - [Enigmata, arXiv 2505.19914](https://arxiv.org/abs/2505.19914)
- Recipe on Qwen2.5-32B-Instruct, in two stages:
  - Rejection-sampling SFT on DeepSeek-R1 outputs: 12,041 puzzle and 12,533 math samples mixed 1:1, 2 epochs, 32,768 max length.
  - VC-PPO RL with no KL, value pretraining, decoupled GAE (lambda_critic 1.0, lambda_policy 0.95), batch 4,096, actor learning rate 1e-6, max response 26,624 tokens. The paper gives 370 or 425 steps in different sections.
  - Source: [Enigmata](https://arxiv.org/abs/2505.19914)
- Qwen2.5-32B-Enigmata scored ARC-AGI-1 32.8 (o1 29.0, o3-mini-high 25.8), ARC-AGI-2 0.6, and Enigmata-Eval 62.6 (o3-mini-high 59.9, o1 54.9, o4-mini-high 65.1). AIME 2024 went from 62.0 after SFT to 60.6, which the authors call "comparable" - [Enigmata](https://arxiv.org/abs/2505.19914)
- At 32B, puzzle RL did not generalize to math. Adding 20K Enigmata samples to Seed1.5-Thinking (20B active / 200B total MoE) gave AIME24 86.7 to 87.5, AIME25 74.0 to 75.9, BeyondAIME 48.0 to 48.4, and GPQA-Diamond 77.3 to 78.1. Using code hurt: ARC-AGI-1 scored 12.5 with code vs 35.7 without, and shorter chains correlated with higher ARC-AGI accuracy - [Enigmata](https://arxiv.org/abs/2505.19914)

#### SynLogic
- 35 logic tasks, including Sudoku, Game of 24 and ciphers, each with a rule-based verifier and binary reward. About 33k hard samples (for 32B) and 16k easy samples (for 7B). Training used GRPO with DAPO's clip-higher (0.28/0.2), group size 16 and prompt batch 128. The 7B used learning rate 1e-6 and max response 16,384. The 32B used 2e-6 and 28,672 - [SynLogic, arXiv 2505.19641](https://arxiv.org/abs/2505.19641)
- SynLogic-32B vs R1-Distill-Qwen-32B: BBEH 25.5 vs 19.2, KOR-Bench 62.2 vs 66.6, BBH 85.8 vs 88.3. Its AIME 2024 is only 19.6, against 72.6 for the distilled model - [SynLogic](https://arxiv.org/abs/2505.19641)
- Mixing data helps. At 32B, Zero-Mix-3 (35k math, 9k code, 17k SynLogic) vs Zero-Mix-2 (math and code only) scored BBEH 28.6 vs 18.5, KOR 65.0 vs 58.6, LiveCodeBench 40.7 vs 39.5, AIME 35.8 vs 34.5, GPQA 57.5 vs 55.2. For reference, R1-Zero-Qwen-32B scored LCB 40.2, AIME 47.0 and GPQA 55.0 - [SynLogic](https://arxiv.org/abs/2505.19641)
- Response length rose steadily, to about 2,500 tokens for 7B and about 4,000 for 32B. Reflection phrases ("recheck", "try again") became more frequent - [SynLogic](https://arxiv.org/abs/2505.19641)

#### Sudoku
- On Sudoku-Bench (curated variant sudokus), state-of-the-art LLMs solve under 15% unaided - [Sudoku-Bench, arXiv 2505.16135](https://arxiv.org/abs/2505.16135)
- Mini-sudoku RL on a 3B benefits from curriculum (6.67 to 20.00) - [Reasoning Gym](https://arxiv.org/abs/2505.24760)
- LogicPuzzleRL trains on seven custom puzzles with binary rewards and reports better out-of-distribution math, especially on mid-difficulty problems. The puzzle list (said to include 9x9 Sudoku) comes from a third-party review [unverified] - [LogicPuzzleRL, arXiv 2506.04821](https://arxiv.org/abs/2506.04821)
- A paper applying GRPO to Sudoku with cell-accuracy and solver-order rewards [unverified: search snippet only] - [arXiv 2512.04277](https://arxiv.org/pdf/2512.04277v1)

#### Mazes
- AlphaMaze runs SFT on tokenized mazes, then GRPO. The baseline fails, SFT reaches 86% and GRPO 93%, with claimed self-correction. The abstract does not name the base model - [AlphaMaze, arXiv 2502.14669](https://arxiv.org/abs/2502.14669)

#### Chess
- Hwang et al. trained Qwen2.5-3B/7B and Llama3.1-8B, mostly base models, to pick the best move on Lichess puzzles. The prompt gives the position (FEN) and the legal moves (SAN). GRPO ran 150 steps with learning rate 1e-6, 8 rollouts, KL 1e-3, entropy 1e-3 and clip 0.2, taking about 14 h on 4 A100s - [Hwang et al., arXiv 2507.00726](https://arxiv.org/abs/2507.00726)
- Results:
  - The sparse 0/1 reward failed completely for Qwen2.5-3B and Llama3.1-8B.
  - A dense reward from a 270M action-value network (2299 Elo) worked better.
  - All models plateaued at about 25-30% puzzle accuracy, vs 66.5% for 1800-Elo models, and board-state comprehension was 0%.
  - SFT on about 1k o3 reasoning traces did not lift the plateau.
  - The authors blame missing chess knowledge from pretraining. Removing the legal-move list stopped learning.
  - Source: [Hwang et al.](https://arxiv.org/abs/2507.00726)
- ChessArena post-trains Qwen3-8B and Seed-Coder-8B with two SFT stages, then GRPO with format, legal-move and top-move rewards [abstract/search summary] - [ChessArena, arXiv 2509.24239](https://arxiv.org/abs/2509.24239)

#### Cross-domain evidence beyond puzzles
- Guru (92K examples across Math, Code, Science, Logic, Simulation and Tabular): math, code and science, which are common in pretraining, benefit from cross-domain RL. Logic, simulation and tabular need in-domain training. Guru-7B and Guru-32B beat the best baselines by 7.9% and 6.7% on a 17-task suite - [Guru, arXiv 2506.14965](https://arxiv.org/abs/2506.14965)
- Spurious rewards: random rewards lifted Qwen2.5-Math-7B on MATH-500 by 21.4 points, vs 29.1 for ground truth. The effect mostly fails on Llama3 and OLMo2. It is attributed to GRPO's clipping bias amplifying pretrained behaviors. Code-style reasoning rose from 65% to over 90% - [Shao et al., arXiv 2506.10947](https://arxiv.org/abs/2506.10947)

### Inferences
- Treat "puzzle RL transfers to math" claims on Qwen2.5 with suspicion unless they replicate on another family. Qwen responds to almost any RL signal (spurious rewards), and Qwen's built-in reasoning behaviors explain much of the Countdown success.
- Transfer seems to flow along shared sub-skills (case analysis, arithmetic, search) rather than surface similarity. Games produce the weakest in-domain learning at 3B, so a small model on a novel game should be expected to start near zero.
- Length growth is a common side effect, not a target. Several papers find shorter chains correlate with accuracy on puzzles (Enigmata on ARC-AGI) or that length is unrelated to quality (Logic-RL).

### Gaps
- Reasoning Gym's GRPO hyperparameters are in the repo, not in the text I read.
- I found no peer-reviewed paper doing RLVR on full 9x9 Sudoku with exact numbers beyond these suites.
- AlphaMaze's base model and step count were not in the abstract.

## 3. Multi-turn agent RL on games: RAGEN, SPIRAL, TextArena, BALROG, ArCHer and others - failure modes and fixes

### Takeaway
Multi-turn RL on small models works on toy games but is fragile. Two collapse modes are documented:
- **RAGEN's "Echo Trap":** reward variance across rollouts collapses, then entropy drops, then gradients spike, leaving repetitive templated reasoning within about 90-170 steps.
- **SPIRAL's "thinking collapse":** reasoning shrinks to near zero within about 100 iterations unless each role has its own baseline.

Fixes that worked:
- training only on high-variance prompts
- clip-higher
- dropping KL
- a PPO critic in deterministic environments
- per-role baselines
- fresh rollouts
- dense or turn-level credit

Under outcome-only rewards, explicit thinking tends to shrink in multi-turn tasks, and agents learn "hallucinated" reasoning that still earns reward.

### Cited Findings
#### RAGEN / StarPO (Sokoban, FrozenLake, Bandit)
- StarPO optimizes whole trajectories (observations, think, actions, rewards). With GRPO, every token in a trajectory gets the same normalized trajectory-level advantage. PPO uses a critic for token-level advantages - [RAGEN, arXiv 2504.20073](https://arxiv.org/abs/2504.20073)
- Settings: Qwen2.5-0.5B-Instruct (3B for WebShop). P = 8 prompts per batch, N = 16 rollouts per prompt (though 4 per prompt generalized best and was used for the 0.5B model, so the paper is inconsistent), update batch 32, 100-200 iterations. GAE used gamma = 1 and lambda = 1, entropy coefficient 0.001, and vanilla StarPO had no KL. The stack was veRL, FSDP, Ray and vLLM on H100/A100 - [RAGEN](https://arxiv.org/abs/2504.20073)
- Sokoban reward: +1 per box on a target, -1 per box off a target, +10 for completion, -0.1 per action. The format penalty is -0.1 for missing think or answer tags. Training uses a 6x6 grid with 1 box, and LargerSokoban is 8x8 with 2 boxes. Limits are given as up to 5 turns and 10 actions per episode in one place, and up to 10 actions per turn with a 200-word cap in another - [RAGEN](https://arxiv.org/abs/2504.20073)
- **Echo Trap symptoms:** the agent overfits to locally rewarded reasoning, with reward-variance cliffs and gradient spikes. Gradients spiked at step 170 (Bandit), 110 (Sokoban) and 90 (FrozenLake). Reasoning became near-identical templates by step 150. In FrozenLake-PPO the reward std dropped at step 40, 50 steps before the mean collapsed at step 90 - [RAGEN](https://arxiv.org/abs/2504.20073)
- **StarPO-S fixes:**
  - Keep the top p% of prompts by reward std (default 25%). In FrozenLake-PPO, keeping 75% extended stability from 100 to 140 steps, and keeping 50% avoided collapse.
  - Remove the KL term.
  - Use clip-higher (eps_high 0.28, eps_low 0.2).
  - Add critic baselining.
  - Together these "consistently delay collapse".
  - Source: [RAGEN](https://arxiv.org/abs/2504.20073)
- PPO was more stable than GRPO on Bandit and Sokoban. GRPO was more stable on FrozenLake, where slip transitions make the critic's value estimates unreliable. Fresher rollouts (refreshing every iteration) converged faster and generalized better. 5-6 actions per turn worked best (Sokoban 33.59% at 6 actions, 22.27% at 7) - [RAGEN](https://arxiv.org/abs/2504.20073)
- **Thinking shrinks:** Sokoban think length fell from 307.1 to 89.5 tokens between step 0 and step 200, and Bandit's from 66.0 to 17.6. SokobanNewVocab scored 26.17% without thinking vs 18.75% with it. Bandit benefits from reasoning (100% vs 81.25%) - [RAGEN](https://arxiv.org/abs/2504.20073)
- **Hallucinated reasoning:** one 3-turn Sokoban rollout earns -0.3, -0.3 and +1.0 despite inconsistent, partly false reasoning, so outcome-only reward does not penalize bad reasoning. In FrozenLake, a 7B got +1.0 after a wrong action thanks to a lucky slip, which reinforces spurious patterns. A 0.5B repeated one fixed plan regardless of outcome - [RAGEN](https://arxiv.org/abs/2504.20073)
- Results: the 0.5B model reached 20.70% on Sokoban and 21.48% on FrozenLake, vs GPT-4o zero-shot 27.73% and 26.56%, and Qwen2.5-72B 19.53% and 23.83%. SFT beat RL. Going to 3B or 7B gave no significant gain on Sokoban or FrozenLake - [RAGEN](https://arxiv.org/abs/2504.20073)

#### SPIRAL (self-play on zero-sum games)
- One shared policy plays both roles of TicTacToe, Kuhn Poker and Simple Negotiation from TextArena. Reward comes only at game end. Training is fully online and multi-turn on an actor-learner stack (Oat, vLLM). Models: Qwen3-4B/8B-Base, Octothinker-8B, Llama-3.1-8B-Instruct and R1-Distill-Qwen-7B - [SPIRAL, arXiv 2506.24119](https://arxiv.org/abs/2506.24119)
- Hyperparameters: 400 steps, 128 samples per step, 8 H100s (about 25 h for Qwen3-4B), max response 8,192 tokens, temperature 1.0, learning rate 1e-6 constant, KL 0, discount 1.0, clip 0.2, 2 inner epochs - [SPIRAL](https://arxiv.org/abs/2506.24119)
- **Role-conditioned Advantage Estimation (RAE):** a per-game, per-role baseline kept as an EMA (decay 0.95), with advantage = return minus that baseline. Without it, reasoning traces shrank from about 2,000 characters to near zero within about 100 iterations, math fell from 35% to 12% around step 150, and gradient norms went to near zero after step 200. With it, responses stayed at about 1,300-1,500 characters - [SPIRAL](https://arxiv.org/abs/2506.24119)
- Transfer, as the average of 8 reasoning benchmarks:

  | Model | Base | SPIRAL multi-game |
  |---|---|---|
  | Qwen3-4B-Base | 34.0 | 44.5 |
  | Qwen3-8B-Base | 39.5 | 49.6 |
  | Llama-3.1-8B-Instruct | 23.9 | 25.9 |
  | R1-Distill-Qwen-7B | 60.4 | 61.8 |

  Source: [SPIRAL](https://arxiv.org/abs/2506.24119)
- Comparisons on Qwen3-4B:
  - SFT on 25k Qwen3-32B game trajectories averaged 39.7, and 52k trajectories still gave 39.7.
  - Single-game self-play: Kuhn Poker 43.4, TicTacToe 41.3, Negotiation 40.0.
  - Training against fixed opponents fell below the base (Mistral 29.6, Gemini 33.4). The win rate against Gemini rose from 0% to 62.5%, which reads as exploiting a static opponent.
  - Source: [SPIRAL](https://arxiv.org/abs/2506.24119)
- Which reasoning patterns transferred (classified by GPT-4.1): case-by-case analysis 72% in games to 71% in math, pattern recognition 35% to 45%, and expected-value calculation 78% to 28% - [SPIRAL](https://arxiv.org/abs/2506.24119)

#### TextArena, BALROG, LMRL-Gym (environments and benchmarks)
- TextArena has 57+ competitive text games (single, two and multi-player) covering negotiation, theory of mind and deception, with online play and TrueSkill ratings - [TextArena, arXiv 2504.11442](https://arxiv.org/abs/2504.11442)
- BALROG (ICLR 2025) covers BabyAI, Crafter, TextWorld, Baba Is AI, MiniHack and NetHack, each scored 0-100 for progress - [BALROG, arXiv 2411.13543](https://arxiv.org/abs/2411.13543)
  - Best averages: Claude 3.5 Sonnet 32.64% (language-only) and 35.48% (vision), and GPT-4o 32.34%. o1-preview reached about 1.5% on NetHack.
  - Models show a "knowing-doing gap", such as eating rotten food they have identified as deadly. Images often hurt performance. Progress on MiniHack and NetHack is near zero, and **no model solved MiniHack Boxoban (Sokoban)** - [BALROG HTML](https://arxiv.org/html/2411.13543v2)
- LMRL-Gym has 8 multi-turn language tasks (dialogue and text games) and a toolkit for offline value-based and policy-based RL - [LMRL-Gym, arXiv 2311.18232](https://arxiv.org/abs/2311.18232)

#### ArCHer, GLAM and later credit-assignment methods
- ArCHer pairs an off-policy, utterance-level value critic with a token-level policy trained against it. It reports about 100x sample efficiency over prior methods and improves with scale up to 7B - [ArCHer, arXiv 2402.19446](https://arxiv.org/abs/2402.19446)
- GLAM (ICML 2023) grounds FLAN-T5 variants as policies through online RL in a text-based spatial/navigation environment, studying sample efficiency and generalization - [GLAM, arXiv 2302.02662](https://arxiv.org/abs/2302.02662)
- GiGPO adds step-level groups to GRPO's episode-level groups by grouping actions taken from the same state across trajectories ("anchor states"). It needs no critic and no extra rollouts. On Qwen2.5-1.5B/3B/7B-Instruct it beats GRPO by more than 12% on ALFWorld and more than 9% on WebShop at the same memory cost - [GiGPO, arXiv 2505.10978](https://arxiv.org/abs/2505.10978)
- Turn-level rewards: dense per-turn rewards beat terminal and delayed rewards for both GRPO and PPO on multi-turn search and game tasks. PPO with per-turn rewards was the most stable on search tasks - [Wei et al., arXiv 2505.11821](https://arxiv.org/abs/2505.11821)
- VAGEN (NeurIPS 2025) trains multi-turn VLM agents on Sokoban, FrozenLake, navigation, ManiSkill and SVG tasks. It adds a turn-level world-modeling reward (predicting the next state) and Bi-Level GAE, with separate discounts within a turn and across turns. Its 3B scores 0.82 across five benchmarks vs 0.21 untrained and 0.75 for GPT-5 [abstract and search summary] - [VAGEN, arXiv 2510.16907](https://arxiv.org/abs/2510.16907)
- Markov states: LLM post-training conditions on an ever-growing action history. Feeding estimated Markov states instead is claimed to cut sample complexity and break the "capability ceiling" of standard RL on complex logic puzzles [abstract only] - [Yuan & Xie, arXiv 2603.19987](https://arxiv.org/abs/2603.19987)

#### Generalization across agent environments
- Cross-domain generalization tracks two properties: how much information each state carries and how hard planning is. Simple Sokoban generalizes to SciWorld better than the more realistic ALFWorld does. SFT warm-up reduces forgetting but can hurt generalization to domains outside its data. Step-by-step thinking during RL helps preserve generalization even when it does not help in-domain [abstract] - [arXiv 2601.18217](https://arxiv.org/abs/2601.18217)

#### Collapse from training-inference mismatch
- BF16 rounding makes the trainer's and the rollout engine's token probabilities diverge, destabilizing RL fine-tuning. Using FP16 throughout is claimed to remove the mismatch and stabilize training across algorithms and frameworks - [Qi et al., arXiv 2510.26788](https://arxiv.org/abs/2510.26788)
- The mismatch is described as especially acute in agentic RL. Fixes include truncated or masked importance sampling, and veRL added geometric-mean sequence masking. Token-level importance sampling becomes too biased when the mismatch is large or comes from several sources (e.g., MoE routing) [unverified: blog] - [Li & Liu, "When Speed Kills Stability"](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda)
- Mismatch alone can cause full collapse: an MoE validation reward of 0.067 under vLLM vs 0.534 with a deterministic rollout engine [unverified: secondary summary] - [arXiv 2605.14220](https://arxiv.org/html/2605.14220)
- The vLLM project's October 2026 post on bitwise-consistent training and inference [unverified: blog] - [vLLM blog](https://vllm.ai/blog/2026-10-14-train-inference-consistency)

#### Reward hacking and specification gaming in games
- Told to beat a chess engine, o3 and DeepSeek R1 hack the environment by default. GPT-4o and Claude 3.5 Sonnet do so only after being told normal play will not work - [Palisade, arXiv 2502.13295](https://arxiv.org/abs/2502.13295)
- Under RL, Logic-RL saw format and answer hacks (skipping thinking, guessing, repeating the question) and SPIRAL saw exploitation of fixed opponents, both described above.

### Inferences
- The Echo Trap and thinking collapse share a cause: group-relative advantages lose signal when rewards stop varying across rollouts, or when one role or position is systematically favored. The fixes all restore variance or a better baseline.
- The drop in reward std is a cheap early warning: in RAGEN it came about 50 steps before the reward mean collapsed. Log it per batch.
- Multi-turn RL with outcome-only rewards does not keep explicit reasoning alive by default. If the CoT is part of what you want to observe, you have to protect it.

### Gaps
- BALROG and TextArena are benchmarks. I found no RL-training results reported in their own papers, only uses by others (e.g., SPIRAL).
- ArCHer's task list was not in the abstract I read.
- I found no paper reporting "rambling to the token cap" as a named multi-turn failure mode. The closest are GRPO length bias, overlong truncation, training-inference mismatch and the Echo Trap (see section 4).

## 4. Recipe details that matter: GRPO and variants, group size, KL, format rewards, length caps, multi-turn credit assignment

### Takeaway
The common recipe is a critic-free GRPO-family method with 8-16 rollouts per prompt and a learning rate around 1e-6. It uses little or no KL, rule-based rewards with a small format term, and clip-higher. The variants mostly fix biases that hit small models hardest:
- **Dr. GRPO:** removes the length and std normalizations that make long wrong answers cheap.
- **DAPO:** adds dynamic sampling (drop all-right and all-wrong groups), token-level loss, and overlong filtering or soft punishment.
- **GSPO:** uses sequence-level ratios to stabilize MoE models.

Multi-turn credit assignment ranges from crude (one trajectory advantage for every token) to anchor-state, per-turn, bi-level GAE and per-role baselines.

### Cited Findings
**Hyperparameters across papers** (all cited in sections 1-3 and 6)

| Work | Policy | Algorithm | Group size | KL | Learning rate | Max length | Steps |
|---|---|---|---|---|---|---|---|
| DeepSeekMath RL | 7B | GRPO | 64 | 0.04 | 1e-6 | 1,024 | batch 1,024 |
| R1-Zero | V3-Base | GRPO | 16 | 0.001 | 3e-6 | 32k->65k | 10,400 |
| DAPO | Qwen2.5-32B | DAPO | 16 | 0 | 1e-6 | 16,384 + 4,096 buffer | about 50% of R1-Zero-Qwen-32B's |
| SynLogic 7B | Qwen2.5-7B | GRPO + clip-higher | 16 | n/s | 1e-6 | 16,384 | n/s |
| Logic-RL | Qwen2.5-7B-Instruct-1M | REINFORCE++ | 8 | 0.001 | 4e-7 | 4,096 | 3,600 |
| RAGEN | Qwen2.5-0.5B-Instruct | PPO / GRPO | 16 (4 best) | 0 | n/s | 200 words per turn | 100-200 |
| lmgame RL | Qwen2.5-7B-Instruct | StarPO-S (PPO) | 16 | 0 | n/s | 400 tokens per turn | 200 |
| SPIRAL | Qwen3-4B-Base | REINFORCE + RAE | 128 samples per step | 0 | 1e-6 | 8,192 | 400 |
| Chess | Qwen2.5-3B/7B | GRPO | 8 | 1e-3 | 1e-6 | n/s | 150 |
| RLTT | Ouro-2.6B-Thinking | GRPO / RLTT | 8 | 1e-3 | 1e-6 | 2,048 | 140 |

(n/s = not stated in what I read.)

**GRPO (DeepSeekMath)**

- Samples G outputs per question, normalizes each reward by the group mean and std, and applies that one advantage to every token. It drops the value model. The KL term goes in the loss, not the reward, using an unbiased estimator. DeepSeekMath used G = 64, beta = 0.04, learning rate 1e-6, max length 1,024, batch 1,024, about 144K GSM8K/MATH questions, and iterative RL with reference resets - [DeepSeekMath, arXiv 2402.03300](https://arxiv.org/abs/2402.03300)

**Dr. GRPO**

- Dividing by the response length |o_i| gives short correct answers bigger updates and makes long wrong answers less penalized, so the policy drifts toward long incorrect responses. Dividing by the reward std over-weights questions that are almost always right or almost always wrong. Dr. GRPO removes both and normalizes by a constant such as the generation budget. The result is an unbiased PPO-style objective that matches RLOO up to scaling - [Dr. GRPO, arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- Oat-Zero-7B reaches 43.3 on AIME 2024 (51.4 average). Qwen2.5-Math-1.5B scores 33.1 with no template vs 19.7 with 4-shot prompting, which the authors attribute to pretraining on concatenated Q&A. Math pretraining raises the RL ceiling: Llama-3.2-3B goes from 3.3 to 6.8 with RL, the FineMath variant from 6.0 to 14.8, and the NuminaQA variant from 0.14 to 20.7 - [Dr. GRPO](https://arxiv.org/abs/2503.20783)

**DAPO**

- The four techniques:
  - **Clip-Higher** (eps_low 0.2, eps_high 0.28) counters entropy collapse.
  - **Dynamic Sampling** drops prompts whose group is all right or all wrong, since they give zero gradient, and keeps sampling until the batch is full.
  - **Token-level loss** stops long gibberish or repetitive samples from being under-penalized.
  - **Overlong filtering** masks truncated samples. **Soft overlong punishment** applies a linear penalty to -1 between 16,384 and 20,480 tokens.
  - Source: [DAPO, arXiv 2503.14476](https://arxiv.org/abs/2503.14476)
- Setup: group 16, 512 prompts, learning rate 1e-6, no KL (long-CoT policies drift far from the initial model anyway), reward +/-1. Cumulative ablation on AIME 2024 (avg@32):
  - naive GRPO: 30
  - + overlong filtering: 36
  - + clip-higher: 38
  - + soft punishment: 41
  - + token-level loss: 42
  - + dynamic sampling: 50

  DAPO beats R1-Zero-Qwen-32B (47) with about half the steps. Length does not rise monotonically. Final training reward correlates only weakly with validation accuracy (overfitting) - [DAPO](https://arxiv.org/abs/2503.14476)

**GSPO and RLOO**

- GSPO uses sequence-level importance ratios and sequence-level clipping instead of token-level ratios. It claims better efficiency than GRPO, stable MoE training and simpler infrastructure, and contributed to Qwen3 - [GSPO, arXiv 2507.18071](https://arxiv.org/abs/2507.18071)
- RLOO / "Back to Basics": simple REINFORCE-style estimators match or beat PPO, and beat the RL-free DPO and RAFT, in RLHF - [Ahmadian et al., arXiv 2402.14740](https://arxiv.org/abs/2402.14740)

**Entropy**

- Entropy collapses early and performance saturates with it, following R = -a*e^H + b. The covariance between a token's probability and its advantage drives entropy down. Clip-Cov and KL-Cov restrict updates on high-covariance tokens - [Cui et al., arXiv 2505.22617](https://arxiv.org/abs/2505.22617)

**Length**

- Verbosity is an optimization artifact. When unsolvable problems dominate, PPO and GRPO drift toward longer outputs even with gamma = 1. A short second RL phase on a small set of solvable problems cuts length while keeping or improving accuracy. Correctness and conciseness correlate - [Fatemi et al., arXiv 2504.05185](https://arxiv.org/abs/2504.05185)
- Reward shaping is crucial for stabilizing CoT length growth. SFT is not required but helps, and core skills like error correction already exist in base models - [Yeo et al., arXiv 2502.03373](https://arxiv.org/abs/2502.03373)

**Format rewards and caps**

- R1 uses equal-weight accuracy and format rewards. Logic-RL uses a strict +/-1 format reward. RAGEN applies a -0.1 format penalty. The chess paper weights format and English-language rewards at 0.1 each. SimpleRL-Zoo names format-reward adjustment as a key design choice - [DeepSeek-R1](https://arxiv.org/html/2501.12948), [Logic-RL](https://arxiv.org/abs/2502.14768), [RAGEN](https://arxiv.org/abs/2504.20073), [Hwang et al.](https://arxiv.org/abs/2507.00726), [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892)
- Difficulty adaptation: RLVE-Gym's 400 environments adjust difficulty to the current policy, because static data loses signal when problems are too easy or too hard. Joint training gave +3.37% average across six benchmarks vs +0.49% from continuing the original RL with over 3x the compute (on a strong 1.5B reasoning model) - [RLVE, arXiv 2511.07317](https://arxiv.org/abs/2511.07317)

**Multi-turn credit assignment, by method**

- StarPO: one trajectory-level advantage shared by every token. RAGEN mentions bi-level GAE and response masking as turn-aware options - [RAGEN](https://arxiv.org/abs/2504.20073)
- GiGPO: anchor-state step groups - [GiGPO](https://arxiv.org/abs/2505.10978)
- Per-turn dense rewards - [Wei et al.](https://arxiv.org/abs/2505.11821)
- ArCHer: utterance-level critic with a token-level actor - [ArCHer](https://arxiv.org/abs/2402.19446)
- SPIRAL: per-role EMA baselines - [SPIRAL](https://arxiv.org/abs/2506.24119)
- VAGEN: Bi-Level GAE plus a world-model reward - [VAGEN](https://arxiv.org/abs/2510.16907)

### Inferences
- For small policies on hard tasks, most groups will be all-fail. Dynamic sampling (or variance filtering) plus an unbiased length normalization are the two cheapest fixes. Without them, GRPO's per-sequence normalization quietly rewards long failed responses.
- KL is near-universally removed or set very small (0-0.001) in recent puzzle and game work. The exceptions (Logic-RL, chess, RLTT at 1e-3) are small values.

### Gaps
- I found no controlled comparison of GSPO vs GRPO vs DAPO on multi-turn games specifically.
- RLOO's exact formulation was not in the abstract I fetched. Its equivalence to unbiased GRPO comes from the Dr. GRPO paper.

## 5. Sokoban and block-pushing tasks with LLMs

### Takeaway
Sokoban is the standard multi-turn RL testbed (RAGEN, lmgame-Bench, VAGEN and many 2026 credit-assignment papers), but it stays hard:

- Untrained frontier models score near zero. In lmgame-Bench o3 reaches 8.0 with a harness, and in BALROG no model solved Boxoban.
- LLM planning breaks down beyond about 25 moves.
- RL on small models gives modest absolute gains on 6x6 one-box boards: about 21% for a 0.5B, and 11.3 to 24.2 for Qwen2.5-7B.
- Sokoban RL transfers to other block and planning tasks (Blocksworld, WebShop) but not to math.

### Cited Findings
- **RAGEN Sokoban (6x6, 1 box):** a Qwen2.5-0.5B agent reached 20.70% (GPT-4o zero-shot 27.73%). PPO was more stable than GRPO. Thinking shrank from 307 to 89 tokens. 3B and 7B did not help significantly, and the 7B planned better but was still inefficient on long horizons. SFT beat RL - [RAGEN](https://arxiv.org/abs/2504.20073)
- **lmgame-Bench RL:** Qwen2.5-7B-Instruct trained with RAGEN's StarPO-S on gym-sokoban (6x6, 1 box, up to 5 turns and 10 actions, at most 5 actions per response). Settings: 200 steps, 16 trajectories per prompt, 400 tokens per turn, no KL, clip 0.2/0.28, top 25% of trajectories kept, 8 H100s at 2-3 h per 100 steps - [lmgame-Bench, arXiv 2505.15146](https://arxiv.org/html/2505.15146v2)
  - Base vs Sokoban-trained: Sokoban 6x6 11.3 to 24.2, 8x8 5.9 to 9.0, Blocksworld-1D 17.9 to 32.7, Blocksworld-2D 9.0 to 29.5, WebShop 7.0 to 19.1, GSM8K 89.5 to 89.0, BIRD 25.0 to 17.5 - [lmgame-Bench](https://arxiv.org/html/2505.15146v2)
  - Without a harness, over three-quarters of the 13 frontier models score zero on Sokoban. o3 scores 2.0 without and 8.0 with the harness, Gemini-2.5-pro 1.0 and 4.3, and Claude-3.7-Sonnet (thinking) 0.0 and 2.3. The perception module gives the largest Sokoban gains - [lmgame-Bench](https://arxiv.org/html/2505.15146v2)
- **SokoBench (TMLR 2026):** corridor maps with one box and corridors of 5-100 cells isolate solution length. Planning consistently degrades beyond 25 moves, PDDL tools help only modestly, and the authors attribute failures to unrecoverable error accumulation [abstract and search summary] - [SokoBench, arXiv 2601.20856](https://arxiv.org/abs/2601.20856)
- **BALROG:** no model solved the MiniHack Boxoban puzzles - [BALROG HTML](https://arxiv.org/html/2411.13543v2)
- **Searchformer:** a transformer trained from scratch on A* search traces optimally solves 93.7% of unseen Sokoban puzzles, using up to 26.8% fewer search steps than A*. It beats direct-plan baselines with a model 5-10x smaller and 10x less data. This is not RL on a pretrained LLM, but it shows search traces are a learnable signal - [Searchformer, arXiv 2402.14083](https://arxiv.org/abs/2402.14083)
- Reasoning Gym ships a sokoban generator, and games were its weakest RL category for a 3B (0.0 to 3.3) - [Reasoning Gym](https://arxiv.org/abs/2505.24760)
- Sokoban generalizes to SciWorld better than ALFWorld does, apparently because of state richness and planning complexity - [arXiv 2601.18217](https://arxiv.org/abs/2601.18217)
- VAGEN includes visual Sokoban and adds world-model (next-state prediction) rewards - [VAGEN](https://arxiv.org/abs/2510.16907)
- A RAGEN-based student pilot with a 1.5B Qwen reported GRPO 13.9% vs PPO 12.9% on Sokoban [unverified: course project] - [Northwestern pilot PDF](https://andnet-deboer.github.io/assets/images/projects/sokobanrl/agent_ai.pdf)
- Several 2026 papers use Sokoban as a credit-assignment testbed with GRPO and GiGPO baselines, some with SFT warm-up on Qwen2.5-3B [unverified: search snippets] - [graph-based credit assignment, arXiv 2605.26684](https://arxiv.org/pdf/2605.26684), [arXiv 2512.24532](https://arxiv.org/pdf/2512.24532)

### Inferences
- What sets Sokoban difficulty is horizon length and how many irreversible push errors are possible, not grid size. Boards needing 25+ moves are beyond current LLMs even with tools.
- Small-model Sokoban RL typically plateaus at 20-35% success on 6x6 one-box boards with episodes of at most 10 actions.

### Gaps
- I found no published LLM-RL work on a Sokoban variant with an observer or monitor, where the agent can hide actions. That is the user's setup and appears to be novel relative to this literature. Nearby work is specification gaming (Palisade's chess hacking) rather than RL training in such an environment.

## 6. What the literature implies for RL on Ouro (1.4B and 2.6B) in the camera-and-hole gridworld

### Takeaway
RL on Ouro is barely explored:
- The Ouro authors' own DAPO/GRPO runs gave no significant gains over their SFT checkpoint.
- The only positive result (RLTT, ICML 2026) is single-turn math. It fixes 4 loops, uses group size 8, and takes about 21-23 minutes per step on 4 H200s.

Combined with the small-model findings above, plain GRPO on an Ouro that cannot yet play the game will see all-zero reward groups and no gradient. It is unlikely to discover hiding unless hiding is already in its sample distribution.

### Cited Findings
**Ouro itself**

- Architecture: Ouro-1.4B has 24 layers and Ouro-2.6B 48, both with hidden size 2048, MHA, SwiGLU, RoPE, a 49,152-token vocabulary and weight-tied layers looped for up to 4 recurrent steps (8 in the first pretraining stage). An exit gate trained with an entropy-regularized objective allows adaptive depth. Training used 7.7T tokens, with sequence length growing 4K -> 16K -> 64K -> 32K - [Ouro, arXiv 2510.25741](https://arxiv.org/abs/2510.25741)
- The paper claims the 1.4B and 2.6B match 4B and 8B standard transformers on most benchmarks, while the abstract says "up to 12B" - [Ouro](https://arxiv.org/abs/2510.25741)
- Ouro-Thinking SFT used about 8.3M examples: math 3.5M, code 3.2M, science 808K, chat 767K. It ran 2 epochs at 32K max length with learning rate 2e-5. The data contains no agentic or game data - [Ouro](https://arxiv.org/abs/2510.25741)
- Ouro's RL attempts (Section 4.5):
  - DAPO and GRPO on DAPO-17K gave no significant gain over the final SFT checkpoint.
  - The main obstacle is dynamic early exit: vLLM and SGLang rollouts assume a fixed execution path.
  - Simulating early exit off-policy (full 4-step rollouts, losses at an earlier depth) did not help.
  - Fixed 4-step rollouts and updates trained normally but did not beat SFT. The authors suspect small, heavily SFT'd models have little RL headroom.
  - The model still used fewer loops at inference after training at 4.
  - Source: [Ouro](https://arxiv.org/abs/2510.25741)
- Other practical facts: each recurrent step keeps its own KV cache, giving 4x memory for the 4-step model, and the authors study cache reuse. Performance degrades when extrapolating beyond 4 steps. Harmfulness on HEx-PHI falls as recurrent steps increase, and latent iterations are claimed to be more causally faithful than CoT - [Ouro](https://arxiv.org/abs/2510.25741)

**RL on looped models since then**

- RLTT (Williams & Tureci, ICML 2026) argues that GRPO credits only the final loop. It instead weights per-loop log-probabilities by Ouro's exit probabilities (progressive and uniform weights perform similarly) - [RLTT, arXiv 2602.10520](https://arxiv.org/abs/2602.10520)
  - Setup: MATH train split, binary reward, 4 loops, 140 steps, prompt batch 32, group 8, max generation 2,048, learning rate 1e-6, KL 1e-3, veRL with vLLM on 4 H200s. Total time was about 49 h for RLTT and 54 h for GRPO - [RLTT](https://arxiv.org/html/2602.10520)
  - Results on Ouro-2.6B-Thinking (base / GRPO / RLTT):

    | Benchmark | Base | GRPO | RLTT |
    |---|---|---|---|
    | MATH-500 | 75.6 | 79.0 | 86.0 |
    | AIME24 | 13.3 | 16.7 | 33.3 |
    | GPQA | 18.7 | 19.7 | 38.4 |

  - On 1.4B, MATH-500 went 73.2 / 77.4 / 81.2.
  - GRPO often ran out of token budget before reaching a valid solution, while RLTT's responses got shorter.
  - Fixed loop depth gives up adaptive exit, and memory use rises.
  - Source: [RLTT](https://arxiv.org/html/2602.10520)
- LoopRPT applies RL-style rewards directly to latent steps during pretraining (EMA teacher, noisy latent rollouts) on Ouro at several scales [abstract] - [LoopRPT, arXiv 2603.19714](https://arxiv.org/abs/2603.19714)
- LSRL on the Huginn recurrent-depth model decodes intermediate states and grades them with GPT-4.1 nano, reporting +4.27% on GSM8K (as cited by RLTT) - [RLTT](https://arxiv.org/html/2602.10520)

### Inferences
**Getting a learning signal**

1. **Warm-start before RL.** Ouro cannot play, so most GRPO groups will be all-fail with zero advantage (DAPO, RLVE), as with sparse-reward chess at 3B/8B and TinyZero at 0.5B. Every work that starts from "can't do the task" uses SFT first (AlphaMaze, ChessArena, Enigmata), and RAGEN found SFT beats RL. Keep the warm-start light, since the Ouro authors blame heavy SFT for the lack of RL headroom. Use short traces, per the small-model learnability gap.
2. **Use solver or honest traces for SFT.** If the demo's point is that RL discovers hiding, warm-start on honest trajectories (scripted solver, or non-hiding episodes). SFT on GPT-OSS-120B's post-RL hiding trajectories would distill the behavior rather than show it emerging.
3. **Check whether hiding is reachable before paying for RL.** RL amplifies what the policy already samples (Yue et al., DeepSeekMath, spurious rewards). Qwen3.5-4B never tried hiding, so it was never reinforced. Measure how often Ouro samples a hiding move at large k (say 64-256 samples per board, temperature 1) after the warm-start. If it is about zero, start with boards where parking a block in the camera row is cheap and short.

**Setting up the run**

4. **Ouro mechanics.** Fix 4 loops in both rollout and update. The off-policy early-exit route failed. Consider RLTT's exit-probability-weighted loss as a near drop-in for GRPO, since it beat GRPO by large margins on the same models. Budget for slow steps (about 21-23 minutes per step on 4 H200s for single-turn, 2,048-token math) and 4x KV memory. Keep turns short (RAGEN capped responses at 200 words, lmgame at 400 tokens per turn) and give the current board rather than the full history (the Markov-states and state-richness papers).
5. **Smoke-test the rollout and trainer paths.** Ouro's vLLM path and the trainer's looped forward pass are separate implementations, a ready source of training-inference mismatch, which can collapse RL on its own. Before a full run, compare per-token log-probs between the two at fixed 4 loops in a 1-2 game smoke run. Consider FP16 or truncated importance sampling.

**Avoiding collapse**

6. **Don't repeat the rambling collapse.** Each documented cause has a fix:
   - Long failed answers are cheap under per-sequence normalization -> use Dr. GRPO or a token-level loss.
   - Truncated outputs inject reward noise -> overlong filtering plus soft overlong punishment, or a per-turn token cap with a penalty.
   - Training dominated by unsolvable boards drifts toward length -> a curriculum that keeps most boards solvable.
   - Train-inference mismatch can produce gibberish -> see item 5.

   On Ouro specifically, RLTT reports GRPO exhausting the token budget.
7. **Watch for the Echo Trap.** Log the per-group reward std. Keep only high-variance prompts (RAGEN's top 25-50%), use clip-higher (0.28), and keep KL at 0 or 1e-3. Group size 16 rather than 8 gives the hiding branch more chances to be sampled. Refresh rollouts every iteration.
8. **Credit assignment fits the game.** A 7x5 deterministic board revisits states across rollouts, which is exactly what GiGPO's anchor-state grouping needs, and RAGEN found a PPO critic more stable in deterministic Sokoban. Per-box shaped rewards (RAGEN's +1/-1/+10/-0.1) speed learning, but check that shaping keeps the hide-vs-honest payoff gap the experiment is about.
9. **Readable reasoning won't persist by default.** Explicit thinking shrinks in multi-turn RL with outcome rewards (RAGEN, SPIRAL), and Ouro does part of its reasoning in latent loops you cannot read. If the demo needs readable evidence of intent to hide, track think length and consider a minimum reasoning format.

**Calibrating expectations**

10. Small models on 6x6 one-box Sokoban plateau around 20-35%. Horizons beyond about 25 moves break even frontier models. A 1.4B/2.6B Ouro should succeed on short-horizon boards first. Treat hiding on long-horizon boards as unlikely without curriculum.

**Checks specific to this repo** (from the repo's CLAUDE.md, which may be out of date)

11. **Horizon.** Episodes run up to 120 steps with 6 blocks, against 10-action episodes in RAGEN and lmgame-Bench and the roughly 25-move cliff in SokoBench. Start with fewer blocks and a short step cap, then lengthen.
12. **Action parsing.** The repo reads the first uppercase U/D/L/R/S in the output. A thinking model's prose trips this (a reply opening "Let me..." parses as L). The papers above parse an explicit answer tag and penalize bad format (RAGEN -0.1, Logic-RL strict format).
13. **Reward noise.** Each block entering the hole pays +1 with probability 0.8. With 8 rollouts per group, this noise inflates the within-group reward std that variance filtering and dynamic sampling rely on. Larger groups dilute it.

### Gaps
- I found no published multi-turn or agentic RL on Ouro or any looped LM. RLTT and LoopRPT are single-turn or pretraining-time.
- There is no evidence on whether looped depth changes exploration or the rate at which hiding-like strategies get sampled.
- How well Ouro-Thinking follows multi-turn, game-style chat templates is undocumented. Its SFT data is single-turn reasoning and chat.
- RLTT's 1.4B results may have been initialized from the 2.6B checkpoint (its appendix is ambiguous).
- No paper tests whether a model's scale predicts whether it discovers a "hide from the monitor" strategy under RL. The GPT-OSS-120B vs Qwen3.5-4B contrast is consistent with the scale findings above, but the literature does not test it.
