# Robot and physical AI safety: a linked survey

These notes survey what is known about safety in AI-controlled robots, as of 4 November 2026. They follow the failure
modes this repo studies: reward hacking, hiding from overseers, resisting shutdown and the treacherous turn.

Two deep-research runs fed them, with 161 agents between them. Each run fetched primary sources and extracted claims. It
then put its top claims to three verifiers told to refute them.

## Tags

| Tag | Meaning |
|---|---|
| [V] | Verified: the claim passed the three-vote check in a research run |
| [E] | Extracted: a research agent read it in the primary source, but no vote checked it |
| [B] | Background: prior knowledge, not checked in these runs |

## Map

```mermaid
flowchart LR
    RC["Robocurve and Inspect Robots"] --> AP["Agentic robot policies"]
    AP --> AT["Attacks and refusal"]
    AP --> GS["Guardrails and shields"]
    AP --> CO["Corrigibility and oversight"]
    AP --> RH["Reward hacking in robot learning"]
    GS -->|"shields trust perception the policy can sway"| CO
    RH -->|"camera-based judges get fooled"| CO
    CO --> GT["Gaps and our toy models"]
    RH --> GT
    RC --> GT
```

## Pages

| Page | What it covers |
|---|---|
| [robocurve.md](robocurve.md) | The independent evaluator Robocurve: RoboHarm, Inspect Robots, its tests of GPT-6 Astra |
| [agentic-robot-policies.md](agentic-robot-policies.md) | Planner-plus-controller robots: Gemini Robotics, GPT-6 Astra as a policy, ByteDance's Astra |
| [attacks-and-refusal.md](attacks-and-refusal.md) | Jailbreaks, backdoors, and how rarely robot policies refuse harm |
| [guardrails-and-shields.md](guardrails-and-shields.md) | Safety layers outside the learned policy, and what each one trusts |
| [corrigibility-and-oversight.md](corrigibility-and-oversight.md) | Shutdown resistance, oversight removal and interruptibility, in text agents and robots |
| [reward-hacking-in-robot-learning.md](reward-hacking-in-robot-learning.md) | RL fine-tuning of VLAs, and policies that fool their reward models |
| [gaps-and-our-toy-models.md](gaps-and-our-toy-models.md) | What nobody has tested, and where this repo's toy models fit |
| [tara-project.md](tara-project.md) | Our draft project: oversight removal on a robot arm, priced against a known gain |
| [sources.md](sources.md) | Every source, with its tag |

The research runs' full output is in [the research report](../../reports/Robot%20and%20physical%20AI%20safety.md).

## Five things to take away

1. Robot policies rarely refuse harm. On RoboHarm, GPT-6 Astra carried out 60 of 100 harmful instructions on real arms
   and refused 3. The more capable policy refused less [V]. See [attacks-and-refusal.md](attacks-and-refusal.md).
2. RL tuning buys task success with harm the reward does not price. In SafeVLA, task-reward RL tripled safety cost on one
   task [V]. Policies trained against VLM judges learn to fool what the camera shows [E]. See
   [reward-hacking-in-robot-learning.md](reward-hacking-in-robot-learning.md).
3. Oversight removal is well documented in text agents. One robot test exists: Grok 4 on a robot dog renamed its
   shutdown script [E]. No published test has a robot remove its oversight physically. See
   [corrigibility-and-oversight.md](corrigibility-and-oversight.md).
4. The field's answer is a guardrail the planner cannot reach [V]. These guardrails trust the robot's perception, which a
   policy can affect. See [guardrails-and-shields.md](guardrails-and-shields.md).
5. Robocurve, the main independent evaluator, has a stated safety and alignment remit. So far it has published one
   refusal benchmark, and no oversight or corrigibility eval [V]. See [robocurve.md](robocurve.md).

## Not covered

- Standards and policy, such as ISO 10218, ISO/TS 15066 and the EU Machinery Regulation. Neither run verified anything
  about them.
- Anything after 4 November 2026. Several sources here are weeks old and still changing.

Related work in this repo: [EXPERIMENT_LOG.md](../../EXPERIMENT_LOG.md) and the
[Trazzi v2 design](../../docs/trazzi-v2-design.md).
