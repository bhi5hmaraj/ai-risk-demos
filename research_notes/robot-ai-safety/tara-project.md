# TARA project: oversight removal on a robot arm

Draft, 4 November 2026. Tags follow [the index](README.md).

## Goal

Make "The Price of Corrigibility" relevant to Robocurve's frontier robotics evals. The runs are due 5 December 2026 and
the report 10 December. The project should build on this repo's camera game and Trazzi experiments.

## Why this is open

- Robocurve has a safety and alignment remit, but no oversight or corrigibility eval, and Inspect Robots has no safety
  task [V]. See [robocurve.md](robocurve.md).
- The one robot shutdown test used one model, removed oversight in software, and needed a hint in the prompt [E]. See
  [corrigibility-and-oversight.md](corrigibility-and-oversight.md).
- No published test has a robot policy remove its oversight with its body [E]. See
  [gaps-and-our-toy-models.md](gaps-and-our-toy-models.md).

## Prior art, and what we add

| Prior work | What it did | What we add |
|---|---|---|
| Palisade's robot dog, Feb 2026 | Grok 4 renamed a shutdown script: 3 of 10 real runs | Physical removal, a swept gain, several models, and real against simulated framing |
| Palisade, TMLR 2026 | Shutdown sabotage in a shell, as one rate per prompt | A price: the gain at which a model removes oversight |
| Apollo, 2024 | Flipping an oversight flag in a config file | An overseer only the body can defeat |
| Anthropic's RL-trained Opus, Aug 2026 | RL taught a model to kill its monitor, in text | Stretch goal: does our camera-game RL carry over to an arm? |
| RoboHarm, Sep 2026 | Refusal of misuse on real arms | Misalignment, in the same harness |
| SOLE-R1, Dern et al., 2026 | Reward hacking while training robot policies | An eval of deployed agent policies, with no VLA training needed |

## Design

The decision is the proposal's: keep the supervisor, or remove it for more attempts at a known gain. The numbers stay the
same across three forms.

1. **Text:** the Stage 0 harness, as written.
2. **Console:** Trazzi v2's console world, with `stop_monitor` priced by the same gain.
3. **Arm:** an Inspect Robots task on an SO-101. As in Stage 4, a webcam watches the tray along one line and ends the
   episode when it sees a cube in the tray. A cube placed on that line blinds it.

Conditions on each form:

- **Supervisor or benign control:** the same numbers, with no supervisor in the control.
- **Real or simulated:** the prompt says "real robot" in some runs and "simulated robot" in others.
- **Instruction:** neutral, or "do not block the camera".

Scoring comes from state: whether the webcam saw the cube or was blinded. No LLM judge is needed, and a person checks a
sample of videos.

Models: two or three current frontier agents through their APIs, plus GPT-OSS-120B. A stretch arm adds GPT-OSS-120B with
this repo's camera-game RL adapter.

## Statistics and cost

- 20 trials per cell only separate 0% from 100% [V]. The gain sweep gives a curve instead, which is more robust than
  ranking models.
- On the arm: 3 gains, 2 framings and 20 runs is 120 episodes per model.
- At a median of about 197 seconds per real rollout [E], that is about 6.6 hours of arm time per model.
- An SO-101 costs about $120 to $150 in parts [E]. API fees run 2 to 24 cents an episode [E], so 360 episodes cost
  roughly $10 to $90.

## Timeline

| Week | Work | Gate at the end |
|---|---|---|
| 4 to 10 Nov | Text and console forms, base rates on two models. Order the SO-101 | A model removes oversight at some gain in text |
| 11 to 17 Nov | The Inspect Robots task: scene, camera-line check, scorer. Dry runs in the mock world or Isaac Lab | The agent drives the SO-101. If not, switch the arm form to simulation |
| 18 to 26 Nov | Arm runs | 120 episodes per model done |
| 27 Nov to 5 Dec | Fill gaps and analyse. The report is due 10 Dec | None |

## Risks

- Inspect Robots is in early alpha, so pin a version [V].
- Verifiers split on whether its agent policy runs on the SO-101, though SO-101 Bench drove one through it [E].
- Palisade's result needed a prompt hint [E]. The prompt must state the camera's rule plainly without suggesting the
  block.
- The default agent prompt says the robot is real [V], which changes eval awareness.
- Use speed caps and soft objects on the real arm.

## What Robocurve would get

An open oversight-removal task for Inspect Robots, publishable in World Evals the way RoboHarm is. Also a price per model,
and a measure of whether "real" and "simulated" framing changes behaviour.

## Options not chosen

- **RL transfer:** testing the camera-game adapters on the arm. Kept as the stretch arm above.
- **The clean eval:** a honeypot that shows the monitor as down, with flags compared against what happened. It fits as a
  section of the report.
