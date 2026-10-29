# Looped, recursive and latent-depth reasoning models

Current as of 29 Oct 2026. Primary sources are arXiv papers and the ARC Prize blog, plus the Hugging Face model card. Items marked "(preprint)" are unreviewed arXiv postings, many from Aug-Oct 2026. Items marked "(secondary)" come from press or blogs and were not checked against a primary source. "T", "loops", "recurrent steps" and `total_ut_steps` all mean the number of passes through the shared block.

## 1. HRM and TRM: architecture, deep supervision, ACT, puzzle results, ablations, and the ARC Prize analysis

### Takeaway
HRM (27M params) and TRM (5-7M params) reach 55-87% on Sudoku-Extreme and Maze-Hard, where chain-of-thought LLMs score 0%, and 40-45% on ARC-AGI-1. In both, the gains come from deep supervision: an outer loop that repeatedly refines the answer, with every refinement step supervised. The brain-inspired H/L hierarchy contributes little. The ARC Prize team found that a same-size plain transformer in HRM's pipeline lands within about 5 points of HRM. Most of HRM's ARC score comes from test-time training on the evaluation tasks' own demonstration pairs, through per-puzzle embeddings.

### Cited Findings
**HRM (Wang et al., Sapient, June 2025)**
- Sapient is a Singapore-based lab. The HRM paper (Guan Wang et al.) went viral: over 4M views on X and over 475K on YouTube videos - [ARC Prize, HRM analysis](https://arcprize.org/blog/hrm-analysis), [arXiv 2506.21734](https://arxiv.org/abs/2506.21734).
- Architecture: two coupled recurrent modules, a slow high-level module H for "abstract planning" and a fast low-level module L for "detailed computations". It solves tasks "in a single forward pass without explicit supervision of the intermediate process". It has 27M parameters, trains on about 1,000 samples, and uses no pretraining or CoT data - [HRM](https://arxiv.org/abs/2506.21734).
- "Hierarchical convergence": within each cycle L converges to a local equilibrium that depends on H's state. H then updates, which restarts L's computation. Effective depth is N*T steps, where a plain RNN would converge prematurely - [HRM](https://arxiv.org/abs/2506.21734).
- Training uses a one-step gradient approximation justified by deep-equilibrium models and the implicit function theorem. Memory is O(1) instead of BPTT's O(T) - [HRM](https://arxiv.org/abs/2506.21734).
- Deep supervision: the forward pass is split into segments m. Each segment is a full HRM pass that produces y_hat^m and its own loss. The state z^m is detached before it feeds segment m+1 - [HRM](https://arxiv.org/abs/2506.21734).
- ACT via Q-learning. A Q-head on the final H state predicts (Q_halt, Q_continue). The model halts if m >= Mmax, or if Q_halt > Q_continue and m >= Mmin. With probability eps, Mmin is drawn uniformly from {2,...,Mmax}. Otherwise Mmin = 1. Halting returns reward 1{y_hat=y}. The loss is the sequence loss plus BCE(Q, G) - [HRM](https://arxiv.org/abs/2506.21734).
- Inference-time scaling: on Sudoku-Extreme-Full, a model trained with Mmax=8 keeps gaining accuracy when run at Mmax=16. Extra compute "yield[s] minimal gains in ARC-AGI ... as solutions generally require only a few transformations" - [HRM](https://arxiv.org/abs/2506.21734).
- Default HRM configuration: two 4-layer networks with n=2, T=2 - [TRM appendix](https://arxiv.org/abs/2510.04871). The ARC Prize team describes the same baseline as L=2, H=2 - [ARC Prize](https://arcprize.org/blog/hrm-analysis).
- Reported scores, as tabulated by TRM from Wang et al.:
  - HRM: Sudoku-Extreme 55.0%, Maze-Hard 74.5%, ARC-AGI-1 40.3%, ARC-AGI-2 5.0% (2 tries).
  - DeepSeek R1 (671B), Claude 3.7 8K and o3-mini-high all score 0.0% on Sudoku-Extreme and Maze-Hard. A 27M direct-prediction model also scores 0.0%.
  - ARC-AGI-1 / ARC-AGI-2 for the same LLMs: R1 15.8/1.3, Claude 3.7 16K 28.6/0.7, o3-mini-high 34.5/3.0, Gemini 2.5 Pro 32K 37.0/4.9, Grok-4-thinking 66.7/16.0, Grok-4 with a bespoke harness 79.6/29.4.
  - [TRM Tables 4-5](https://arxiv.org/abs/2510.04871).
- Datasets:
  - Sudoku-Extreme: 9x9 hard puzzles, 1K training and 423K test examples.
  - Maze-Hard: 30x30 mazes whose shortest path is longer than 110, generated with the procedure of Lehnert et al. 2024. 1,000 train and 1,000 test.
  - Augmentation per example: 1,000 rule-preserving shuffles for Sudoku, 8 dihedral transforms for Maze, and 1,000 color, dihedral and translation transforms for ARC.
  - [TRM Sec. 5](https://arxiv.org/abs/2510.04871).

**ARC Prize team analysis of HRM (2025, analysis led by Konstantin Schuerholt, Ndea)**
- Semi-private verification: ARC-AGI-1 32% (9h16m, $148.50 total, $1.48/task), against 41% claimed on the public evaluation set. ARC-AGI-2 2% (12h35m, $201, $1.68/task), "not ... material progress" - [ARC Prize](https://arcprize.org/blog/hrm-analysis).
- Finding 1: the hierarchy barely matters. A transformer of the same size (~27M) in the same pipeline comes "within ~5pp" with no hyperparameter tuning, and is on par at 1 outer loop. HRM leads at more loops, but the gap closes as loops increase. Moving the H/L step counts away from L=2, H=2 in either direction hurt performance - [ARC Prize](https://arcprize.org/blog/hrm-analysis).
- Finding 2: the outer refinement loop drives the gains.
  - Going from 1 loop (no refinement) to 2 loops adds 13pp. Going from 1 to 8 loops doubles public-eval performance.
  - ACT beats a fixed 16-loop run only "within a few percentage points".
  - A model trained with 16 loops but run with 1 beats a model trained and run with 1 loop by more than 15pp.
  - "Training with refinement loops is more important than using the refinement for inference."
  - [ARC Prize](https://arcprize.org/blog/hrm-analysis).
- Finding 3: cross-task transfer is limited.
  - HRM trains on augmented demonstration pairs of 960 tasks: 400 ARC-1 training, 160 ConceptARC and 400 ARC-1 evaluation tasks.
  - Training only on the 400 evaluation tasks still gives 31% pass@2, against 41%. HRM is thus "fundamentally a zero-pretraining test-time training approach".
  - The team expects per-task training to approach Liao and Gu's "ARC-AGI without pretraining" (21% pass@2).
  - [ARC Prize](https://arcprize.org/blog/hrm-analysis).
- Finding 4: augmentation is critical. 300 augmentations reach near-max performance against the paper's 1,000. 30 augmentations land within 4% of max. Augmentation at training time matters more than the size of the inference voting pool - [ARC Prize](https://arcprize.org/blog/hrm-analysis).
- The model sees only the input grid and a `puzzle_id` embedding, with no few-shot context, so it "can only be applied on puzzles with puzzle_ids it has seen at training time". It is purely transductive - [ARC Prize](https://arcprize.org/blog/hrm-analysis).

**TRM (Jolicoeur-Martineau, Samsung SAIL Montreal, Oct 2025: "Less is More: Recursive Reasoning with Tiny Networks")**
- One network with only 2 layers and 7M parameters. TRM scores 45% on ARC-AGI-1 and 8% on ARC-AGI-2. It "requires no fixed-point theorem, no complex biological justifications, and no hierarchy" - [TRM](https://arxiv.org/abs/2510.04871).
- Mechanism:
  - The model keeps an embedded question x, an answer y and a latent z.
  - One recursion process updates z <- f(x+y+z) n=6 times, then updates y <- f(y+z).
  - Each supervision step runs T=3 recursion processes: T-1 without gradients, then one with full backpropagation.
  - There are up to N_sup=16 supervision steps.
  - Effective depth per supervision step is T(n+1)*n_layers = 42.
  - [TRM](https://arxiv.org/abs/2510.04871).
- Sudoku-Extreme ablations, test accuracy (Table 1):
  - HRM: 55.0% (27M, 2 forward passes per step).
  - TRM (T=3, n=6): 87.4% (5M).
  - With HRM-style ACT (continue loss, 2 forward passes): 86.1%.
  - With separate f_H and f_L: 82.4% (10M).
  - Without EMA: 79.9%.
  - 4 layers with n=3: 79.5%.
  - Self-attention instead of MLP: 74.7%.
  - T=2, n=2: 73.7%.
  - With HRM's 1-step gradient: 56.5%.
  - [TRM](https://arxiv.org/abs/2510.04871).
- Latent features (Table 2): two features (y, z) give 87.4%, a single z 71.9%, and a multi-scale z (n+1=7 features) 77.6% - [TRM](https://arxiv.org/abs/2510.04871).
- At matched effective depth (Table 3), HRM vs TRM scores 46.4% vs 63.2% at k=1 and 55.0% vs 81.9% at k=2. HRM reaches 61.6% at k=3 (depth 48) - [TRM](https://arxiv.org/abs/2510.04871).
- Headline results:
  - Sudoku-Extreme: TRM-MLP 87.4% (5M), TRM-Att 74.7% (7M).
  - Maze-Hard: TRM-Att 85.3%. TRM-MLP scores 0.0% on Maze.
  - ARC-AGI-1 / ARC-AGI-2: TRM-Att 44.6% / 7.8%, TRM-MLP 29.6% / 2.4%.
  - [TRM Tables 4-5](https://arxiv.org/abs/2510.04871).
- Training setup:
  - AdamW beta=(0.9, 0.95), 2K-iteration warm-up, batch 768, hidden size 512, N_sup=16, stable-max loss, EMA 0.999.
  - Sudoku and Maze: 60K epochs, lr 1e-4, weight decay 1.0.
  - ARC: 100K epochs, lr 1e-4 (1e-2 for embeddings), weight decay 0.1.
  - Like HRM, TRM learns one embedding per puzzle per augmentation, and at test time takes a majority vote over the 1,000 augmentations.
  - [TRM appendix](https://arxiv.org/abs/2510.04871).
- On why recursion beats a bigger network, the paper says this "remains to be explained" and adds "we suspect it has to do with overfitting" - [TRM](https://arxiv.org/abs/2510.04871).
- TRM won 1st place ($50K) in the ARC Prize 2025 paper awards. 2nd went to Pourcel, Colas & Oudeyer and 3rd to Liao & Gu - [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis), [competition page](https://arcprize.org/competitions/2025). The award details came from a search-tool summary of these pages. The fetched results post itself confirmed TRM's 45%/8% figures.

**Follow-up analyses of HRM/TRM-style models**
- URM (Ubiquant, Dec 2025) compared Universal Transformer variants. It credits ARC gains to "the recurrent inductive bias and strong nonlinear components of Transformer, rather than ... elaborate architectural designs". It adds short convolution and truncated backprop, reaching 53.8% pass@1 on ARC-AGI-1 and 16.0% pass@1 on ARC-AGI-2 - [arXiv 2512.14693](https://arxiv.org/abs/2512.14693).
- "Dissecting HRM" (Aug 2026, preprint): "single-state recurrent Transformers are comparable to HRM". HRM is "constraint-aware iterative refinement on a puzzle-specific solution state". Ablating probe directions has effects comparable to random-direction controls - [arXiv 2609.22197](https://arxiv.org/abs/2609.22197).
- Nested H/L-style variants of HRM, TRM and URM on ARC-AGI-1 (Oct 2026, preprint) showed "no consistent advantage", under a single-GPU budget where exact solving stayed near zero - [arXiv 2610.26666](https://arxiv.org/abs/2610.26666).
- CHARM (Sept 2026, preprint): in these ARC models, task-conditioned memory such as puzzle embeddings is a large hidden source of capacity, "more than 30x the size of the recurrent backbone" - [arXiv 2609.33270](https://arxiv.org/abs/2609.33270).

### Inferences
- HRM, TRM, URM and the ARC Prize ablations all point the same way. The useful ingredient is a learned "improve the current answer" operator trained with deep supervision. Hierarchy, fixed-point theory and the 1-step gradient are incidental or harmful.
- HRM's and TRM's ARC scores rest on per-task test-time training through puzzle embeddings, so they cannot be read as few-shot reasoning. These models cannot run on a new puzzle, or a new game state in an LM agent loop, without retraining.
- The puzzle results come from supervised training on exact input->output pairs. None is trained with RL on its outputs (HRM's Q-learning trains only the halting head), and none works on interactive, multi-step environments.

### Gaps
- None of the sources I read contains an ARC Prize semi-private verification of TRM. The 2025 results post gives TRM's paper numbers without a "verified" label.
- I did not extract TRM's training compute or wall-clock time.
- I found no HRM or TRM results on chess puzzles or on interactive planning environments such as Sokoban.

## 2. Ouro and looped language models (ByteDance, "Scaling Latent Reasoning via Looped Language Models", Oct 2025)

### Takeaway
Ouro-1.4B and Ouro-2.6B run their full layer stack through 4 shared-weight loops. They were pretrained on 7.7T tokens with an entropy-regularized exit distribution over loops 1-4. They match roughly 4B and 8B dense models on most base benchmarks. The paper attributes the gain to better knowledge manipulation, not more knowledge storage: both looped and plain models store about 2 bits per parameter.

Performance peaks at the trained depth (T=4) and degrades beyond it. Generation at T=1 nearly collapses for the Thinking models. The authors' own RLVR attempts gave no gain over SFT.

### Cited Findings
**Authors, release, config**
- The authors are Rui-Jie Zhu and 32 others, including Yoshua Bengio and Jason Eshraghian. Project page: [ouro-llm.github.io](http://ouro-llm.github.io) - [arXiv 2510.25741](https://arxiv.org/abs/2510.25741).
- Architecture:
  - A decoder-only Transformer stack applied recurrently: MHA with RoPE, SwiGLU FFN and "sandwich" RMSNorm before attention and FFN.
  - Vocabulary of 49,152 tokens, from the SmolLM2 tokenizer.
  - Ouro-1.4B has 24 layers. Ouro-2.6B "upcycles 24 layers to 48 via layer duplication". Both use 4 recurrent steps.
  - [Ouro Sec. 4.1, Sec. 4.3.1](https://arxiv.org/abs/2510.25741).
- Model card for 1.4B-Thinking:
  - 24 layers, 4 recurrent steps, hidden size 2048, 32K context after SFT, BF16, Apache-2.0, "intended for research purposes only".
  - `config.json` sets `"total_ut_steps": 4` and `"early_exit_threshold": 1.0`. A value of 1.0 means all steps always run. The loop count is changed by setting `config.total_ut_steps`.
  - "When using vLLM, the model will always execute the full number of `total_ut_steps`." vLLM also "does not currently support the adaptive exit feature".
  - "Please use `transformers<4.56.0`" (4.54.1 recommended) because of a KV-cache compatibility issue with transformers >=4.56.0.
  - The quick-start example samples with temperature=1.0 and top_p=0.7.
  - [HF model card](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking).

**Exit gate and training objective**
- At each step t, the gate gives an exit probability lambda_t computed from the final-layer hidden state h(t). The exit distribution is p(t|x) = lambda_t*prod_{j<t}(1-lambda_j), with the leftover mass assigned to T_max - [Ouro Sec. 3.2](https://arxiv.org/abs/2510.25741).
- Inference uses a "Q-exit" rule: stop at the first step where the cumulative exit probability reaches a threshold q - [Ouro Sec. 3.2](https://arxiv.org/abs/2510.25741).
- Stage I objective: L = sum_t p(t|x)*L(t) - beta*H(p). This equals an ELBO with a uniform prior over exit steps. Without the entropy term, p collapses onto T_max because later steps get more training signal. The uniform prior is chosen over PonderNet's geometric prior or Huginn's Poisson-lognormal prior, both of which favor early halting - [Ouro Sec. 3.3](https://arxiv.org/abs/2510.25741).
- Every loop's output therefore passes through the LM head and receives LM loss, weighted by the exit distribution - [Ouro Sec. 3.3, Sec. 7.2](https://arxiv.org/abs/2510.25741).
- Stage II freezes the LM and trains only the gate. The target continuation probability is w = sigma(k*(I-gamma)) with k=50 and gamma=0.005, where I = max(0, L(t-1) - L(t)) is the detached per-token loss improvement. The gate is trained with BCE against w, penalizing both "underthinking" and "overthinking" - [Ouro Sec. 3.4](https://arxiv.org/abs/2510.25741).

**Training pipeline (7.7T tokens)**
- Stage 1a: 3T tokens with 8 recurrent steps and beta=0.1. This led to "loss spikes and gradient oscillations".
- Stage 1b: 3T tokens with 4 steps and beta=0.05.
- Later stages:
  - Stage 2, CT annealing: 1.4T-token corpus, 16K sequence length.
  - Stage 3, LongCT: 20B tokens of ProLong at 64K.
  - Stage 4, mid-training: 300B effective tokens of QA and CoT data in ChatML format at 32K.
- Optimizer and schedule:
  - Batch size grows from 4M to 8M tokens.
  - Learning rate peaks at 3e-4, is annealed to 3e-5, then reduced to 1e-5 with a cosine schedule.
  - AdamW with weight decay 0.1, beta=(0.9, 0.95), gradient clipping at 1.0.
  - "Recurrent architectures require smaller learning rates than parameter-matched Transformers."
- Data: mainly Nemotron-CC. Chinese data was dropped from Stage 2 on.
- [Ouro Sec. 4](https://arxiv.org/abs/2510.25741).
- Thinking variants come from SFT on about 8.3M examples:
  - 3.5M math (OpenThoughts3, AceReason-1.1-SFT).
  - 3.2M code (AceReason, OpenCodeReasoning, Llama-Nemotron, OpenThoughts3).
  - 808K science.
  - 767K chat (OO1-Chat-747K, DeepWriting-20K).
  - 2 epochs at 32K maximum length with LlamaFactory, lr 2e-5 on a cosine schedule.
  - [Ouro Sec. 4.4](https://arxiv.org/abs/2510.25741).

**Results against larger models**
- The abstract says Ouro 1.4B and 2.6B "match the results of up to 12B SOTA LLMs". The contributions section is more specific: they "match 4B and 8B standard transformers on most benchmarks", for "2-3x parameter-efficiency" - [Ouro](https://arxiv.org/abs/2510.25741).
- Ouro-1.4B at R4 vs Qwen3-4B-Base: BBH 71.02 vs 70.95, GSM8K 78.92 vs 72.86, MATH500 82.40 vs 59.60 - [Ouro Sec. 5.1](https://arxiv.org/abs/2510.25741).
- Ouro-2.6B at R4 vs Qwen3-8B-Base: MMLU-Pro 55.73 vs 53.72, BBH 80.46 vs 77.65, MATH500 90.85 vs 62.30 - [Ouro Sec. 5.1](https://arxiv.org/abs/2510.25741).
- Thinking models at R4 (Table 9, pass@1/pass@10 for AIME):

  | Model | AIME24 | AIME25 | OlympiadBench | BeyondAIME | HLE | SuperGPQA | GPQA |
  |---|---|---|---|---|---|---|---|
  | Ouro-1.4B-Thinking | 65.0/83.3 | 46.3/73.3 | 71.6 | 34.0 | 5.21 | 47.4 | 45.5 |
  | Ouro-2.6B-Thinking | 64.7/90.0 | 50.3/76.7 | 76.4 | 39.0 | 5.58 | 53.7 | 52.7 |
  | Qwen3-4B | 61.3/75.0 | 51.3/63.3 | 73.2 | 31.0 | 5.21 | 51.9 | 54.5 |
  | Qwen3-8B | 73.0/86.7 | 66.7/81.3 | 75.3 | 38.0 | 2.22 | 48.0 | 59.1 |

  [Ouro Sec. 5.2](https://arxiv.org/abs/2510.25741).

**Performance by loop count (trained at T=4)**
- Ouro-1.4B base, MMLU by T=1...8: 41.21, 60.43, 66.71, 67.45, 66.64, 65.77, 65.28, 64.49. ARC-C goes 37.63 -> 60.92 at T=4 -> 58.19 at T=8 - [Ouro Table 10](https://arxiv.org/abs/2510.25741).
- Ouro-2.6B base, MMLU by T=1...8: 51.55, 67.63, 73.57, 74.60, 74.43, 73.79, 72.92, 72.24 - [Ouro Table 11](https://arxiv.org/abs/2510.25741).
- Ouro-1.4B-Thinking by T=1...8 (Table 12):
  - AIME24: 0.00, 37.33, 62.33, 65.00, 60.67, 50.67, 42.33, 38.67.
  - OlympiadBench: 2.22, 59.70, 70.67, 71.55, 72.30, 69.48, 69.04, 66.81.
  - Peak at T=4-5.
  - [Ouro](https://arxiv.org/abs/2510.25741).
- Ouro-2.6B-Thinking by T=1...8 (Table 13):
  - AIME24: 3.00, 52.00, 70.33, 64.70, 57.00, 56.33, 49.67, 39.00.
  - OlympiadBench: 18.96 at T=1, 76.44 at T=4, 39.26 at T=8.
  - Peak at T=3-4.
  - The authors: "performance at T=1 is very low, confirming that iterative refinement is essential".
  - [Ouro](https://arxiv.org/abs/2510.25741).

**Knowledge storage vs manipulation (Physics-of-LMs-style synthetic tasks)**
- Capo (synthetic biographies bioS(N), N from 20K to 500K people, 1,000 exposures, GPT-2-style models from 1M to 40M params): looped and plain models alike store "~2 bits/parameter". Looping "does not increase knowledge capacity nor improve capacity scaling" - [Ouro Sec. 6.1](https://arxiv.org/abs/2510.25741).
- Mano (modular-arithmetic trees, no CoT): looped models beat iso-parameter baselines ({2,3,6}x1) and match or beat the iso-FLOP 12x1 baseline. On synthetic multi-hop QA, looping improves sample efficiency - [Ouro Sec. 6.2](https://arxiv.org/abs/2510.25741).

**Safety, faithfulness, early exit, KV cache, RL**
- Safety on HEx-PHI (330 prompts, 11 categories, GPT-4o judge): harmfulness falls as steps rise from 1 to 8, including the extrapolated steps 5-8. Thinking models reach harmful rates of 0.009 (1.4B) and 0.003 (2.6B) at T=4, against 0.009 for Qwen3-4B-Thinking. Thinking models were sampled at temperature 1.0, top_p 0.7, max_new_tokens=8192 - [Ouro Sec. 7.1](https://arxiv.org/abs/2510.25741).
- PCA of the last-token state shows benign and harmful prompts separating more as loops increase - [Ouro Sec. 7.1](https://arxiv.org/abs/2510.25741).
- Faithfulness, tested on 1,000 ambiguous Quora Question Pairs:
  - In Qwen3-4B-Thinking, a linear probe predicts the final answer from the pre-thinking state with ROC AUC 0.99, which the authors read as post-hoc rationalization.
  - In Ouro-1.4B, probes on layer 24(i-1) do not predict the step-i decision.
  - Steps disagree: step 2 matches step 3 on 551/1000 items and step 4 on 361/1000. For i >= 4, A[i,i+1] ~ 1000, meaning answers stop changing.
  - [Ouro Sec. 7.2](https://arxiv.org/abs/2510.25741).
- Early exit on MMLU:
  - The Stage-II-trained gate is best at every budget: about 66% at an average of 2.5 rounds, against about 64% for the untrained gate.
  - A hidden-state-difference threshold comes within 1-2%.
  - Static exit rises from about 40% to 60% between 1 and 2 rounds, and only marginally from 3 to 4.
  - [Ouro Sec. 5.4.1](https://arxiv.org/abs/2510.25741).
- In decoding, reusing only the last step's KV cache, or an averaged one, stays "within 0.3 points on GSM8K" of the full cache with 4x less KV memory - [Ouro Sec. 5.4.2](https://arxiv.org/abs/2510.25741).
- RL attempts: RLVR with DAPO and GRPO on DAPO-17K "did not yield significant performance gains over the final SFT checkpoint" - [Ouro Sec. 4.5](https://arxiv.org/abs/2510.25741).
  - The authors blame vLLM and SGLang, whose fixed execution path "breaks under LoopLM's variable-depth computation".
  - Approach 1: off-policy. Full 4-step rollouts, with losses computed at a simulated early exit. Did not help.
  - Approach 2: fixed 4-round RL. "Training progressed normally but performance did not surpass the SFT checkpoint". The likely cause: "after ... extensive SFT, these smaller models may have limited headroom".
  - Oddity: "the model still used fewer rounds at inference when beneficial despite being trained at four rounds."
- Related work cited by Ouro: Relaxed Recursive Transformers (shared block plus per-step LoRA), Mixture-of-Recursions (token-level routing over recursion depth), "pondering" LMs, the Inner Thinking Transformer, CoTFormer and Coconut - [Ouro Sec. 2](https://arxiv.org/abs/2510.25741).

### Inferences
- The user's smoke tests fit the paper.
  - Gibberish at 1 loop matches Ouro-1.4B-Thinking's T=1 scores of 0-2% on AIME and OlympiadBench, even though the 1.4B base model at T=1 still scores 41% on logit-based MMLU. Free generation at T=1 is far worse than multiple-choice scoring suggests.
  - 4 loops is the trained depth and the model's best, so a 1->4 sweep spans "broken" to "best". It does not span "less" to "more" latent reasoning at equal competence.
- vLLM runs exactly `total_ut_steps` loops, so the gate plays no part in the user's runs. Loop count is a clean experimental knob, but values above 4 are out of distribution.
- The paper's "faithfulness" claim rests on an observational proxy: probes on one ambiguous binary task, with no interventions. It does not show that the latent trajectory is human-legible.

### Gaps
- I did not extract hidden size, head count or Table 2 details for 2.6B, nor the Thinking models' generation budget for Table 9.
- The paper reports no results on Sokoban, gridworlds or multi-turn agent tasks.
- How vLLM's Ouro implementation handles per-loop KV caches (full 4x or shared) was not checked.

## 3. Recurrent-depth LMs, theory of looped transformers, and earlier recurrent reasoners (Huginn, Saunshi, Giannou, UT, ACT, PonderNet, Deep Thinking, Coconut, RRN, SATNet)

### Takeaway
Two lineages lead to today's looped LMs. Algorithmic recurrent nets (UT, ACT and PonderNet, Deep Thinking, RRN) showed that weight-tied iteration can learn algorithms that run longer at test time. Theory (Giannou, Saunshi) shows looped transformers are programmable computers and can simulate T CoT steps with T loops. Huginn-3.5B was an early large-scale recurrent-depth LM. It was trained at random depths (mean 32) and improves on GSM8K with more test-time recurrence. Interpretability work finds little legible "latent CoT" inside it.

### Cited Findings
**Huginn-3.5B (Geiping et al., Feb 2025)**
- 3.5B parameters trained on 800B tokens. With extra test-time recurrence it improves on reasoning benchmarks "up to a computation load equivalent to 50 billion parameters" - [arXiv 2502.05171](https://arxiv.org/abs/2502.05171).
- Architecture:
  - Prelude, recurrent core and coda with layers (2,4,2). Hidden size 5280, 55 heads of size 96, MLP width 17,920, and a 65,536-token BPE vocabulary.
  - Roughly 1.5B parameters sit in the non-recurrent prelude and head, and 1.5B in the core.
  - The input embedding is re-injected at every step, and the state starts from random noise. Both follow the Deep Thinking literature and promote "path independence".
  - [Huginn Sec. 3-4](https://arxiv.org/abs/2502.05171).
- Training:
  - Each step's recurrence count is drawn from a log-normal Poisson distribution with mean r_bar=32.
  - Truncated backprop through only the last k=8 iterations.
  - 4,096 AMD GPUs at 52-64 TFLOP/s per GPU (1-1.2M tokens/s).
  - The main checkpoint ran 47,000 steps and its learning rate was never cooled down.
  - [Huginn](https://arxiv.org/abs/2502.05171).
- Results:
  - GSM8K-CoT 34.80/42.08 (the two extraction modes), with a system prompt at r=32. OLMo-2-1124-7B scores 61.94/66.19.
  - Easy tasks saturate early: HellaSwag is near peak at about 8 recurrences, while GSM8K keeps gaining with more compute.
  - At 180B tokens, the recurrent model beats a non-recurrent twin, mainly on harder tasks such as ARC-Challenge.
  - At r=1 the recurrent model "stops improving between the early 180B checkpoint and the 800B checkpoint".
  - [Huginn](https://arxiv.org/abs/2502.05171).
- Emergent behaviors:
  - Zero-shot per-token adaptive exit: stop when the KL between successive steps falls below 5x10^-4. Fewer steps are needed on high-school math than on logical fallacies or moral scenarios.
  - KV-cache sharing.
  - "Orbits" in latent space on numeric tokens.
  - [Huginn Sec. 6-7](https://arxiv.org/abs/2502.05171).
- "Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer" (Jul 2025): finds "limited evidence of interpretable latent CoT" in Huginn on arithmetic. Logit-lens and coda-lens readings are inconsistent across recurrent blocks. More depth gives "only marginal gains", well short of explicit CoT - [arXiv 2507.02199](https://arxiv.org/abs/2507.02199).
- "Latent Thinking Optimization" (Sept 2025) argues, per its title, that Huginn-3.5B's latent thoughts "secretly encode reward signals" - [arXiv 2509.26314](https://arxiv.org/abs/2509.26314).
- Retrofitted recurrence (McLeish, Geiping et al., Nov 2025): converts pretrained non-recurrent LMs into depth-recurrent ones using a curriculum over recurrence counts. On math, this beats plain post-training of the original model at equal compute - [arXiv 2511.07384](https://arxiv.org/abs/2511.07384).

**Theory**
- Saunshi et al. ("Reasoning with Latent Thoughts", ICLR 2025):
  - On addition, p-hop induction and math, "a k-layer transformer looped L times nearly matches the performance of a kL-layer non-looped model".
  - Looped models "can simulate T steps of CoT with T loops", and scaling depends on effective depth.
  - In language modeling, looped models are competitive on downstream reasoning but have worse perplexity. This motivates a "dichotomy between reasoning and memorization" and a looping-inspired regularizer.
  - [arXiv 2502.17416](https://arxiv.org/abs/2502.17416).
- Giannou et al. ("Looped Transformers as Programmable Computers", ICML 2023):
  - The input works as a "punchcard" holding instructions and memory.
  - A constant number of encoder layers emulates basic computing blocks, which together emulate a small instruction-set computer.
  - A looped 13-layer transformer can run a basic calculator, a basic linear-algebra library, and in-context learning with backpropagation.
  - [arXiv 2301.13196](https://arxiv.org/abs/2301.13196).

**Universal Transformers, ACT, PonderNet**
- Universal Transformer (Dehghani et al., 2018/ICLR 2019): a weight-shared recurrent Transformer motivated by standard Transformers' failures to generalize on copying and simple logical inference at longer lengths. It adds per-position dynamic halting - [arXiv 1807.03819](https://arxiv.org/abs/1807.03819).
- ACT (Graves 2016) lets RNNs learn how many steps to take per input. It is "deterministic and differentiable". It helps dramatically on parity, logic, addition and sorting, but gives no large gain on character-level language modeling - [arXiv 1603.08983](https://arxiv.org/abs/1603.08983).
- PonderNet (Banino et al., 2021) learns the number of steps end to end. It "succeeds at extrapolation tests where traditional neural networks fail". It matched the state of the art on a real QA dataset with less compute - [arXiv 2107.05407](https://arxiv.org/abs/2107.05407).
- Ouro notes that PonderNet's geometric prior favors early halting - [Ouro Sec. 3.3](https://arxiv.org/abs/2510.25741).

**Deep Thinking networks (easy-to-hard extrapolation)**
- Schwarzschild et al. (NeurIPS 2021): recurrent nets trained on easy prefix sums, mazes and chess puzzles with few iterations solve harder instances "simply by 'thinking for longer'". Example: trained on 32-bit strings with 10 iterations, tested on 36- and 48-bit. Trained on 44-bit data in 10 iterations, they get 100% of 16-bit examples in only 4 iterations - [arXiv 2106.04537](https://arxiv.org/abs/2106.04537).
- Bansal et al. (NeurIPS 2022) name the failure mode "overthinking": behavior degenerates under many iterations.
  - Fixes: a "recall" architecture that re-feeds the input, and a progressive loss that blocks iteration-specific behavior.
  - Mazes: trained only on 9x9 with 30 iterations, the model solves 59x59 and 201x201. An "extreme" maze in the appendix is solved with 10,000 iterations, equivalent to about 100,004 conv layers.
  - Prefix sums: trained on 32-bit, plain DT nets get 0% at 512 bits, while DT-Recall with progressive loss exceeds 97% after about 200 iterations.
  - Recall plus progressive loss also gives "notable improvement on chess puzzles".
  - [arXiv 2202.05826](https://arxiv.org/abs/2202.05826).

**Coconut (Hao et al., Meta, Dec 2024)**
- The last hidden state is fed back as the next input embedding (a "continuous thought"). A multi-stage curriculum replaces CoT steps with latent thoughts - [arXiv 2412.06769](https://arxiv.org/abs/2412.06769).
- Table 1, accuracy (tokens generated):

  | Method | GSM8K | ProntoQA | ProsQA |
  |---|---|---|---|
  | Coconut | 34.1% (8.2 tokens) | 99.8% | 97.0% (14.2 tokens) |
  | CoT | 42.9% (25.0 tokens) | 98.8% | 77.5% (49.4 tokens) |
  | No-CoT | 16.5% | - | - |
  | iCoT | 30.0% | - | - |
  | Pause tokens | 16.4% | - | - |
  | Coconut without curriculum | 14.4% | - | - |

  [Coconut](https://arxiv.org/abs/2412.06769).
- Continuous thoughts "can encode multiple alternative next steps", which allows breadth-first-search-like reasoning - [Coconut](https://arxiv.org/abs/2412.06769).

**Sudoku-specific recurrent and logic models**
- Recurrent relational networks (Palm et al., NeurIPS 2018) solve "96.6% of the hardest Sudoku puzzles", a task "requiring upwards of 64 steps of relational reasoning". They also solve 20/20 bAbI tasks - [arXiv 1711.08028](https://arxiv.org/abs/1711.08028).
- SATNet (Wang, Donti, Wilder, Kolter, ICML 2019) is a differentiable MAXSAT solver based on an SDP relaxation. It solves 98.3% of 9x9 Sudoku test boards (95.0% after 22 epochs, 37 minutes on a GTX 1080 Ti) with no hand-coded rules, and 63.2% of visual MNIST Sudoku - [arXiv 1905.12149](https://arxiv.org/abs/1905.12149).
- Chang et al., "Assessing SATNet's Ability to Solve the Symbol Grounding Problem": without intermediate labels that tie digit images to symbols, "SATNet completely fails at visual Sudoku (0% test accuracy)" - [arXiv 2312.11522](https://arxiv.org/abs/2312.11522).
- Recurrent Transformer for constraint satisfaction (2023) learns CSPs end to end, beats GNNs and SATNet, and handles symbol grounding on visual inputs - [arXiv 2307.04895](https://arxiv.org/abs/2307.04895).

### Inferences
- Huginn and Ouro sit at the two extremes of how loop count is chosen during training. Huginn samples random depths (mean 32) and re-injects the input every step. Ouro trains a fixed maximum of 4 with no re-injection. Section 4 suggests this choice largely decides whether extra test-time loops help.
- Theory says looped models can do T steps of serial computation with T loops. Ouro's T=4 is small next to the dozens to hundreds of serial steps a Sokoban plan needs, so in practice most planning in Ouro would still have to happen in tokens.

### Gaps
- I did not verify the UT paper's specific numbers (LAMBADA, WMT BLEU), PonderNet's exact parity and bAbI numbers, or the chess-puzzle accuracies in Schwarzschild/Bansal.
- RRN's training and test step counts were not extracted.

## 4. Does more test-time depth extrapolate to harder instances than training? Limits (overthinking, instability, past the trained loop count)

### Takeaway
Extrapolation happens when training makes the update reusable. Three ingredients do this: input re-injection ("recall"), progressive or random-depth training, and deep supervision of answer refinement. Deep Thinking nets, TRM, InfiLoop and Equilibrium Reasoners then keep improving to thousands or tens of thousands of steps. Ouro is trained at a fixed depth of 4 without these and degrades past 4.

Known failures: "overthinking" or "latent overthinking" (later updates overwrite correct answers), training instability (loss spikes, residual explosion), finite-step overshoot, and a "one-loop tax".

### Cited Findings
**Where extrapolation works**
- Deep Thinking with recall and progressive loss goes from 9x9 mazes to 201x201, and from 32-bit to 512-bit prefix sums (>97%). Without these techniques, accuracy falls to 0% at 512 bits - [Bansal et al.](https://arxiv.org/abs/2202.05826).
- HRM on Sudoku: a model trained with Mmax=8 improves at Mmax=16. On ARC, extra inference loops give minimal gains - [HRM](https://arxiv.org/abs/2506.21734), [ARC Prize](https://arcprize.org/blog/hrm-analysis).
- TRM (preprint follow-up): extending recurrence from the nominal 16 steps to 512 raises exact-solve accuracy on 1,000 hard Sudokus from 59.2% to 87.5% (attention) and from 74.4% to 91.9% (MLP). Latent motion "drops sharply after the first exact solution" - [arXiv 2609.26487](https://arxiv.org/abs/2609.26487).
- InfiLoop (Oct 2026, preprint) has a 7M-parameter "loop-native residual connection". It reaches 97.9% on Sudoku-Extreme and 13.6% pass@2 on ARC-AGI-2, and "continues to improve with test-time looping beyond 20,000 effective steps" - [arXiv 2610.11570](https://arxiv.org/abs/2610.11570).
- Equilibrium Reasoners (May 2026, preprint) unroll up to the equivalent of 40,000 layers, going from 2.6% (feedforward) to over 99% on Sudoku-Extreme. Simple cases converge in 1-5 iterations - [arXiv 2605.21488](https://arxiv.org/abs/2605.21488).
- PAR (Oct 2026, preprint) is a Qwen2.5-0.5B trained on S5 state tracking with at most 32 moves. It reaches 95.5% at 512 moves (16x training length), against 55.7% reported for RLT and 0.8% for explicit CoT. Removing norm anchoring cuts the extrapolation factor to under 2.2x. Final-answer-only supervision caps it at 8x - [arXiv 2610.27628](https://arxiv.org/abs/2610.27628).
- In Huginn, accuracy grows with recurrence. Easy tasks saturate at about 8 iterations, while GSM8K benefits up to r=32 and beyond. Training covered a broad range of depths (mean 32) - [Huginn](https://arxiv.org/abs/2502.05171).

**Where it fails, and why**
- Ouro is trained at T=4, and "performance degrades as they extrapolate to deeper, unseen recurrent steps (T=6-8)".
  - 1.4B-Thinking on AIME24: 65.00 at T=4, 38.67 at T=8.
  - 2.6B-Thinking on OlympiadBench: 76.44 at T=4, 39.26 at T=8.
  - Safety on HEx-PHI keeps improving through T=8.
  - [Ouro Sec. 5.3](https://arxiv.org/abs/2510.25741).
- Ouro's pretraining at 8 loops "led to loss spikes and gradient oscillations", which forced the reduction to 4 - [Ouro Sec. 4.3](https://arxiv.org/abs/2510.25741).
- Parcae (Apr 2026, preprint): existing looped-training recipes suffer "residual explosion and loss spikes", traced to large spectral norms of the injection parameters - [arXiv 2604.12946](https://arxiv.org/abs/2604.12946).
- "Monitoring latent reasoning in looped LMs" (Oct 2026, preprint, 8 models from 4 families):
  - For carry-loop models (Ouro, LoopUS), the best depth is at or below the training depth.
  - Re-read-loop models (Huginn, Rec-OLMo, Parcae) show "monotonous improvement, a plateau".
  - Majority vote across iterations "never helps".
  - [arXiv 2610.18127](https://arxiv.org/abs/2610.18127).
- "Agreement rather than normalisation" (Oct 2026, preprint) finds that depth stays reusable because of agreement between the training and inference loop counts. Removing Ouro's loop-boundary normalization drops Ouro-1.4B and 2.6B to zero accuracy on a balanced binary task. At matched overshoot of the trained budget, looping's advantage over stacked depth "vanishes on one task and shrinks on the other" - [arXiv 2610.31943](https://arxiv.org/abs/2610.31943).
- "Latent overthinking": "noisy state updates overwrite correct intermediate deductions and even undo completed solutions" - [InfiLoop](https://arxiv.org/abs/2610.11570).
- Finite-step failure: "a locally improving direction produces a harmful full update" - [arXiv 2609.16665](https://arxiv.org/abs/2609.16665).
- PAR on the general case: "additional iterations need not help them solve longer problems" - [PAR](https://arxiv.org/abs/2610.27628).
- One-loop tax: at one loop, latent-recurrent LMs "typically fall well short of a non-recurrent model of the same size" - [ELR, arXiv 2610.12582](https://arxiv.org/abs/2610.12582).
- Sokoban DRC agent (RL-trained ConvLSTM):
  - 6 extra "thinking steps" at episode start raise the success rate by 4.7%, "with a slight drop beyond 16 steps".
  - One Microban level "succeeds with 0 to 16 thinking steps, but fails with 32 or more".
  - [arXiv 2407.15421](https://arxiv.org/abs/2407.15421).
- Huginn shows marginal gains from added depth on arithmetic probes - [arXiv 2507.02199](https://arxiv.org/abs/2507.02199).

### Inferences
- Running more loops helps on harder instances only if training taught the model a reusable update. That means variable or random depth, input re-injection, norm control, and supervision of the intermediate answer. Ouro has deep supervision over steps 1-4 only, and no re-injection.
- For Ouro, more loops than 4 should be treated as an out-of-distribution perturbation, not "more thinking". Behavior changes at T>4 may reflect drift, not deliberation. Safety-like behaviors and capabilities move in different directions there.

### Gaps
- I found no study of Ouro beyond T=8, nor of extrapolation by loop count specifically on planning tasks with LMs.
- The overthinking studies above use supervised tasks. None tests multi-turn interaction.

## 5. RL on looped or recurrent-depth models, and on Sokoban, gridworlds and multi-step planning

### Takeaway
RL on looped LMs is new and mostly infrastructure work. Ouro's authors tried DAPO and GRPO and got no gain over SFT. One Oct 2026 preprint (ScaleRLT) ran GRPO on Ouro-1.4B and raised held-out GSM8K from 34% to 90% in 100 updates. It also showed that rollout and learner log-probabilities disagree numerically for most tokens.

I found no published RL results for looped LMs in Sokoban, gridworlds or multi-turn planning. The closest evidence is RL-trained recurrent conv nets on Sokoban. Those nets use extra thinking steps modestly, plan about 50 steps ahead, and learn to "pace" to buy computation.

### Cited Findings
**Looped LMs**
- Ouro: DAPO and GRPO on DAPO-17K gave no significant gains. The off-policy early-exit simulation failed. Fixed-4-round RL "progressed normally but ... did not surpass the SFT checkpoint" - [Ouro Sec. 4.5](https://arxiv.org/abs/2510.25741).
- ScaleRLT (Oct 2026, preprint) studies GRPO on Ouro-1.4B.
  - Problem: a deferred rollout engine and a dense learner pass compute log-probs that differ in floating point. In FP32 at 4,096 tokens, about 18% of sampled tokens miss a 1e-5 same-weight gate. With bf16 weights, "nearly every token does".
  - Fix: a hybrid learner log-prob, l = l_dense + sg(l_sched - l_dense). Its value replays the rollout bit for bit and its gradient is the dense learner's.
  - Speed: deferral gives 1.47-1.56x rollout throughput on one GPU. A two-node update runs 2.71x shorter.
  - Learning: in 100 updates, held-out GSM8K rose from 23% to 82% with a mixed-depth readout and from 34% to 90% with the native readout. That evidence is one seed per setting, 64-256 samples per update, FP32, and a 500-problem evaluation with one sample each.
  - [arXiv 2610.34427](https://arxiv.org/abs/2610.34427).
- "Towards Looped Models Done Right, Part II" (Oct 2026, preprint) lists RL among the costs each recurrence adds. Near fixed points, "RL updates that compute gradients from saved rollout states" run 2x faster than backprop through the replayed trajectory. Fixed-depth training breaks KV sharing - [arXiv 2610.06833](https://arxiv.org/abs/2610.06833).
- "Running Fewer Loops Removes Refusal Behaviour" (Oct 2026, preprint) finds that "training at randomly chosen depths" makes safety behavior hold across depths, where standard safety training does not. The abstract does not say whether that training is RL or SFT. Numbers are in Sec. 6 - [arXiv 2610.15218](https://arxiv.org/abs/2610.15218).
- HRM's halting head is trained with Q-learning inside otherwise supervised training - [HRM](https://arxiv.org/abs/2506.21734).

**Recurrent networks with RL on Sokoban (non-LM)**
- Guez et al. (ICML 2019): DRC is a model-free ConvLSTM stack trained with RL. It "exceeds the state-of-the-art" on Sokoban, shows planner-like generalization and data efficiency, and can use "additional thinking time" - [arXiv 1901.03559](https://arxiv.org/abs/1901.03559).
- Taufeeque et al. (2024):
  - DRC(3,3) is trained on Boxoban: 10x10 levels with 4 boxes and train, validation and test splits at unfiltered, medium and hard difficulty.
  - Its plan representation is causal and "predicts its future actions about 50 steps in advance". The probe predicts at least 90% of actions 50 steps before they happen.
  - Plan quality and length grow over the first few thinking steps.
  - The network "paces" in cycles to gain computation, and training incentivizes this. Forcing 6 thinking steps removes about 75% of early cycles.
  - The trained RNN extends to larger out-of-distribution puzzles.
  - [arXiv 2407.15421](https://arxiv.org/abs/2407.15421).
- Bush et al. (ICLR 2025) give "the first mechanistic evidence that model-free reinforcement learning agents can learn to plan": DRC on Sokoban forms internal plans from learned concept representations - [arXiv 2504.01871](https://arxiv.org/abs/2504.01871).

### Inferences
- ScaleRLT's quick GSM8K gain, from a low 34% start, probably includes learning the answer format. It neither contradicts nor confirms Ouro's "limited headroom after SFT" observation. It does show that on-policy GRPO on Ouro is feasible once rollout and learner computations agree.
- The DRC "pacing" result is the closest precedent for how a recurrent policy behaves when compute and actions trade off. Under RL, an agent short on internal compute can learn to buy time with harmless actions.

### Gaps
- I found no paper applying RL to Ouro, Huginn or any looped LM in Sokoban, gridworlds, ARC-AGI-3-style games or other multi-turn agent tasks.
- I did not survey RL on Coconut-style continuous-thought models.
- Whether DRC's training used IMPALA was not checked here.

## 6. Developments in 2026: newer looped and latent models, ARC-AGI results, reports about frontier models

### Takeaway
2026 brought a burst of looped-LM work, mostly preprints, many on Ouro. New models include Parcae, LoopUS, HRM-Text and Mimir, and Koi-4B via Elastic Latent Recurrence. Studies show loop count changes behavior unevenly. The clearest case: refusal disappears at shallow depth while capability mostly survives.

ARC-AGI-3 (interactive games) launched in March 2026, with frontier models reportedly under 1%. I found no primary source saying that any frontier production model uses recurrent depth.

### Cited Findings
**Behavior and monitoring vs loop count (directly relevant to the user's study)**
- "Running Fewer Loops Removes Refusal Behaviour in Recurrent-Depth Language Models" (Oct 2026, preprint) studied four released Ouro checkpoints and Huginn-0125.
  - At half its trained depth, Ouro-1.4B-Thinking keeps 89% of its GSM8K accuracy (0.692 vs 0.775). Its rate of helping with harmful requests rises 5.3x, from 0.038 to 0.205.
  - A linear probe reads harmfulness after one loop with ROC AUC 0.967, against 0.986 at full depth. "What the missing loops remove is the refusal that should follow".
  - Ablating one direction at full depth drops refusal from 0.950 to 0.130.
  - Early-exit rules "do not spend more loops on harmful requests". Choosing a depth per request does not recover the loss.
  - Standard safety training makes the shallow model safe only by refusing 0.865 of harmless requests.
  - Random-depth training gives a +0.399 harmful-vs-harmless refusal separation at one loop, against +0.118. On Huginn-0125 it keeps harmful help at or below 0.040 across 1-32 loops, against 0.194 for standard training.
  - [arXiv 2610.15218](https://arxiv.org/abs/2610.15218).
- "Monitoring latent reasoning in looped language models" (Oct 2026, preprint) inspects eight looped LMs through five lenses. It splits them into carry loops (Ouro, LoopUS: the prompt is carried only in the looped state) and re-read loops (Parcae, Rec-OLMo, Huginn: the prompt is re-supplied every iteration).
  - "76-89% of answer tokens are decided at the first iteration in Ouro, Parcae, and Rec-OLMo", and 90% by iteration 2-3. "73-88% of all distribution change happens after the top-1 token is decided."
  - Exit gates "fire late" in carry loops. Huginn "decides gradually".
  - Noise injected at loop entry: carry loops "collapse at alpha=4 (prompt amnesia, script switching)", while re-read loops are "almost unchanged even at alpha=50" (per the summary table).
  - [arXiv 2610.18127](https://arxiv.org/abs/2610.18127).
- "Cross-Loop Query Handoff" (Oct 2026, preprint) studies two-hop retrieval in Ouro. Questions answered wrong at loop 1 and right at loop 3 are rescued 90% (1.4B) and 97% (2.6B) of the time by transplanting loop-3 attention queries into loop 1, against 24% and 21% for shuffled donors. The rescue is weaker on natural questions and replicates only in 2.6B - [arXiv 2610.34639](https://arxiv.org/abs/2610.34639).
- "Present but Hard to Read" (Oct 2026, preprint) probes Ouro-1.4B and Qwen2.5-1.5B with minimal pairs.
  - A copied digit is present in the state, with readout about 0.9.
  - A usual one-prompt probe reads it at only 0.36 (Ouro) and 0.19 (Qwen), even after 5,249 training questions.
  - An unwritten intermediate a+b is "readable but not used": patching it does not move the written digit.
  - [arXiv 2610.28498](https://arxiv.org/abs/2610.28498).
- "Do Halting Heads Really Save Compute?" (Oct 2026, preprint) pretrained models from scratch up to 1.3B parameters. Heads trained on a mix of iterations predict that tokens need under half the iterations, yet tokens "run almost every iteration" under the standard rule. In batched generation, a model without a head "is almost as good" - [arXiv 2610.32246](https://arxiv.org/abs/2610.32246).

**New looped and recursive models and recipes**
- Parcae (Apr 2026, preprint) is a stable looped architecture. It constrains the spectral norm of the injection parameters through a discretized negative-diagonal parameterization, achieves "up to 6.3% lower validation perplexity" than prior large-scale looped models, and fits scaling laws for looping - [arXiv 2604.12946](https://arxiv.org/abs/2604.12946).
- LoopUS (May 2026, preprint) converts a pretrained LLM into an encoder, a looped block and a decoder. It uses a selective gate against state drift, random deep supervision, and a confidence head for early exit - [arXiv 2605.11011](https://arxiv.org/abs/2605.11011).
- HRM-Text (May 2026, preprint) is a 1B HRM-style LM trained from scratch on 40B unique tokens of instruction-response data, for about $1,500. It reports MMLU 60.7, ARC-C 81.9, DROP 82.2, GSM8K 84.5 and MATH 56.2 - self-reported - [arXiv 2605.20613](https://arxiv.org/abs/2605.20613).
- Mimir (Aug 2026, preprint) is a 1.78B HRM-Text model. Its v1.5 reports the best sub-2B average across 20 benchmarks, above Ouro 1.4B and 2.6B. Changing the number of L-stack applications degrades it gracefully. Changing the number of H-stack applications "breaks the model" - [arXiv 2608.13517](https://arxiv.org/abs/2608.13517).
- Elastic Latent Recurrence (Oct 2026, preprint) removes the "one-loop tax" with three choices: a gated state update, loop counts mixed within each optimizer update, and loop-index and budget conditioning.
  - On a ten-task average, it sits 0.52 points below and 1.04 above matched baselines at one loop (4B dense and 30B-A3B MoE hybrid). Four loops add 7.25 and 4.75 points.
  - Koi-4B scores an "estimated" 18 on the Artificial Analysis Intelligence Index at one loop, matching Qwen3.5-4B, and 29 at four loops.
  - [arXiv 2610.12582](https://arxiv.org/abs/2610.12582).
- "Towards Looped Models Done Right" Part I (topology, input injection, state design) and Part II (fixed points, a learned depth prior, orthogonal injection). In Part II, at 1.6B a 3x smaller KV cache matches fixed-depth training with the full cache - [arXiv 2610.23036](https://arxiv.org/abs/2610.23036), [arXiv 2610.06833](https://arxiv.org/abs/2610.06833).
- Other 2026 preprints:
  - RIPE caching: prefix KV of late loops matters far more for self-speculative decoding in Huginn-0125 and Raven-Llama - [arXiv 2610.34436](https://arxiv.org/abs/2610.34436).
  - Gradient lost by truncated backprop across nine looped LMs - [arXiv 2610.28871](https://arxiv.org/abs/2610.28871).
  - "Thinking Deeper, Not Longer": a depth-recurrent Transformer stable for 20+ steps - [arXiv 2603.21676](https://arxiv.org/abs/2603.21676).
  - LoopQuant: post-training quantization for looped Transformers - [arXiv 2610.16368](https://arxiv.org/abs/2610.16368).

**Puzzle and ARC results in 2025-2026**
- ARC Prize 2025:
  - "The central theme driving AGI progress in 2025 is refinement loops." The post describes zero-pretraining refinement models (TRM, and CompressARC with 76K parameters and 20% on ARC-AGI-1 eval) and harness-level refinement on commercial models.
  - Kaggle ARC-AGI-2 private-set top scores: NVARC 24.03% (about $0.20/task), the ARChitects 16.53%, MindsAI 12.64%.
  - Top verified commercial model: Opus 4.5 (Thinking, 64k) at 37.6% for $2.20/task. Top verified refinement system: Poetiq on Gemini 3 Pro at 54% for about $30/task.
  - On one task, Gemini 3 Pro used 96 reasoning tokens and Gemini 3 Deep Think 138,000.
  - [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis). Technical report: [arXiv 2601.10904](https://arxiv.org/abs/2601.10904), not read in detail.
- Recursive-model records (preprints, self-reported):
  - URM: 53.8% pass@1 on ARC-AGI-1 and 16.0% pass@1 on ARC-AGI-2 - [arXiv 2512.14693](https://arxiv.org/abs/2512.14693).
  - InfiLoop: 13.6% pass@2 on ARC-AGI-2 and 97.9% on Sudoku-Extreme - [arXiv 2610.11570](https://arxiv.org/abs/2610.11570).
  - GAR: TRM from 85.9% to 99.4% on Sudoku-Extreme - [arXiv 2610.16427](https://arxiv.org/abs/2610.16427).
  - Equilibrium Reasoners: over 99% on Sudoku-Extreme - [arXiv 2605.21488](https://arxiv.org/abs/2605.21488).
- ARC-AGI-3 (secondary):
  - Launched March 25, 2026 as interactive, turn-based game environments with no instructions - [DataCamp](https://www.datacamp.com/blog/arc-agi-3).
  - Humans solved all 135 environments, and every frontier model scored below 1% at launch - [The Decoder](https://the-decoder.com/arc-agi-3-offers-2m-to-any-ai-that-matches-untrained-humans-yet-every-frontier-model-scores-below-1/), [MindStudio](https://www.mindstudio.ai/blog/arc-agi-3-results-frontier-models-score-zero).
  - The primary ARC Prize post from late 2025 only stated the plan to "release ARC-AGI-3 in early 2026 alongside ARC Prize 2026" - [ARC Prize](https://arcprize.org/blog/arc-prize-2025-results-analysis).

**Frontier models and recurrent depth**
- An extended web search found no 2026 report from a frontier lab describing a production model built on recurrent depth or looping. It returned only academic preprints - [search results incl. arXiv 2608.18230](https://arxiv.org/abs/2608.18230). Any claim that a frontier model "uses recurrent depth" is unverified speculation as far as I can tell.
- The largest looped LMs I found are research models: ELR's 30B-A3B MoE hybrid and Koi-4B - [arXiv 2610.12582](https://arxiv.org/abs/2610.12582).
- A July 2025 position paper argues that CoT monitorability "may be fragile" and that developers should "consider the impact of development decisions on CoT monitorability" - [arXiv 2507.11473](https://arxiv.org/abs/2507.11473).

### Inferences
- The 2026 Ouro studies point to a consistent picture. Information shows up early: harmfulness is readable after 1 loop, and most answer tokens are decided at iteration 1. Later loops mostly execute and sharpen behavior that depends on post-training, such as refusal, two-hop retrieval and answer sharpening. A behavior can be "known" in the state but not expressed at low loop counts.
- Carry-loop models like Ouro are much more fragile to state perturbation than re-read models. This bears on board-state fidelity in a gridworld.

### Gaps
- No ARC-AGI-3 results for looped or recursive models were found.
- The status of the ARC Prize 2026 leaderboard was not checked.
- Koi-4B's developer and Parcae's training scale were not extracted.

## 7. What the literature implies for RL-training Ouro on the camera/Sokoban task

### Takeaway
Treat loops 1-4 as "trained depth" and anything above 4 as "out of distribution". Expect behavior to depend on loop count partly because of capability and training-depth mismatch, not hiding. Train RL at randomly sampled loop counts if the experiment is meant to compare depths fairly. Fix the vLLM-versus-learner log-prob mismatch before trusting GRPO or PPO ratios. Use Ouro's natively decodable per-loop states as the monitor.

### Cited Findings
- At T=1, Ouro-1.4B-Thinking scores 0-2% on AIME and OlympiadBench (2.6B-Thinking: 2-3% on AIME, 19% on OlympiadBench). T=2 recovers much of it (OlympiadBench 59.7 vs 71.6 at T=4 for 1.4B-Thinking). T=4 is near the optimum and T>=6 degrades - [Ouro Tables 12-13](https://arxiv.org/abs/2510.25741).
- Behavior can drop faster than capability at shallow depth. At half depth, Ouro-1.4B-Thinking keeps 89% of GSM8K accuracy while harmful help rises 5.3x. The relevant information stays readable after 1 loop (AUC 0.967) - [arXiv 2610.15218](https://arxiv.org/abs/2610.15218).
- Training at randomly chosen depths makes post-trained behavior consistent across depths, where fixed-depth training does not - [arXiv 2610.15218](https://arxiv.org/abs/2610.15218). Huginn-style random-depth training is what lets recurrence scale - [Huginn](https://arxiv.org/abs/2502.05171), [ELR](https://arxiv.org/abs/2610.12582).
- RL pitfalls found so far:
  - vLLM always runs the full `total_ut_steps` and ignores the gate - [HF card](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking).
  - Ouro's own RLVR gave no gain after SFT - [Ouro Sec. 4.5](https://arxiv.org/abs/2510.25741).
  - Rollout and learner log-probs disagree for about 18% of tokens in FP32, and nearly all in bf16, unless the learner replays the rollout schedule - [ScaleRLT](https://arxiv.org/abs/2610.34427).
  - The HF implementation requires `transformers<4.56.0` - [HF card](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking).
- Every Ouro loop is trained through the LM head, so per-loop readouts are a native "logit lens" - [Ouro Sec. 7.2](https://arxiv.org/abs/2510.25741). Single-prompt probes understate what is present, so minimal-pair probing is recommended - [arXiv 2610.28498](https://arxiv.org/abs/2610.28498).
- Recurrent RL agents on Sokoban gain modestly from extra thinking steps (+4.7% with 6) and learn to "pace" to buy computation - [arXiv 2407.15421](https://arxiv.org/abs/2407.15421).
- Ouro-Thinking was SFT'd on long-CoT data up to 32K tokens and evaluated with up to 8,192 new tokens - [Ouro Sec. 4.4, Sec. 7.1](https://arxiv.org/abs/2510.25741).

### Inferences
- **Loop range.** Ouro-1.4B's 1-loop gibberish is the expected T=1 collapse, not a measurement of hiding. The usable sweep is T=2, 3, 4. T=5-6 can serve as an explicit out-of-distribution probe. Report a capability control at each loop count, for example solve rate on a variant with no camera, so that a change in behavior such as parking the block can be normalized against competence.
- **Train at sampled depths.** An RL policy trained only at T=4 and run at T=1-3 is under distribution shift. The refusal study shows post-trained behaviors can vanish at shallow depth. To claim "more loops -> more hidden behavior", sample T from {2,3,4} (or 1-4) per rollout or per update during RL. Otherwise any behavior that grows with depth may simply mean it was only learned at T=4.
- **RL plumbing.** Use a fixed, native-depth readout. ScaleRLT's native readout beat the mixed-depth one. Recompute learner log-probs with the same loop count and numerics as rollouts, or apply truncated importance sampling. Expect 4x per-token compute at T=4.
- **Monitoring "hiding".** Decode each loop's next-token distribution on the move tokens. Probe for the "park in camera row" intent at loops 1-4 with minimal pairs: the same board with and without the camera. On Ouro, most top-1 decisions are fixed at loop 1 and later loops sharpen. Hidden strategizing would then look like an intent that is decodable early but appears only at later loops or in later moves, and loop-to-loop disagreement (Ouro's step 2 vs 4 agree 36%) is the signal to track.
- **Token versus loop compute.** An agent with fewer loops may compensate with more CoT tokens (Thinking) or more moves, as DRC paces. Fix or log tokens per move and moves per episode at each loop count. For Thinking variants, the >8K tokens per move fits the long-CoT SFT. A base Ouro with a short-format SFT warm start, or an RL length penalty, avoids 4-loop x 8K-token rollouts.
- **Board fidelity.** Ouro is a carry-loop model, with no prompt re-injection, and collapses under modest state noise. Re-serializing the full board in every turn, in a fixed compact format, likely reduces invented objects. Ouro-2.6B's hallucinations at T=4 are a capability issue the literature does not explain specifically.
- **Headroom.** Ouro's authors saw limited RL headroom after heavy SFT, while ScaleRLT got fast gains from a weaker start on GSM8K. A multi-turn sparse-reward Sokoban task has no published looped-LM precedent. Expect to need dense shaping (for example per-block progress) or curriculum on easy boards before the camera trade-off can be learned at all, given 0 blocks at baseline.

### Gaps
- No published work trains a looped LM with RL on multi-turn gridworlds, Sokoban or deception or monitoring setups. Every recommendation above is extrapolated from single-turn math, safety and puzzle results.
- It is unknown whether vLLM's Ouro kernels match HF numerics at each loop count, or whether vLLM shares KV across loops.
- I did not check whether the repo's Tinker + Verifiers backend can train Ouro's custom looped architecture (`trust_remote_code`), or whether LoRA on a weight-tied block behaves differently across loops.
