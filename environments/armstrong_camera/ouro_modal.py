"""Ouro, a looped language model, plays the per-hit Armstrong game on Modal GPUs, one container per loop count.
The loop count sets the latent compute per token, and nothing else changes between containers.

Run with: uv run --with modal modal run ouro_modal.py --boards 1 --loops 1,4
Needs a Modal login (modal setup). The games use the boards and rules of eval_policy.py, so the loop
sweep compares with the GPT-OSS thinking-depth sweep. The output file gets one JSON line per game.
"""
import collections
import itertools
import json
from pathlib import Path

import modal

# The base model answers directly, so all its reasoning happens inside the loops. -Thinking also writes a CoT.
MODEL = "ByteDance/Ouro-1.4B"
app = modal.App("ouro-armstrong")
# vLLM supported Ouro through 0.26.0. It runs every loop on every token, with no early exit.
# FlashInfer's sampler compiles kernels at startup, and this image has no CUDA compiler.
image = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.26.0")
         .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"}))
weights = modal.Volume.from_name("ouro-weights", create_if_missing=True)


@app.cls(image=image, gpu="A10G", volumes={"/root/.cache/huggingface": weights}, timeout=3600, scaledown_window=300)
class Ouro:
    loops: int = modal.parameter()
    model: str = modal.parameter(default=MODEL)

    @modal.enter()
    def load(self) -> None:
        from vllm import LLM
        self.llm = LLM(self.model, trust_remote_code=True, hf_overrides={"total_ut_steps": self.loops}, max_model_len=16384)
        print(f"container for {self.model} at {self.loops} loops runs total_ut_steps="
              f"{self.llm.llm_engine.model_config.hf_config.total_ut_steps}")

    @modal.method()
    def reply(self, conversations: list[list[dict]], max_tokens: int) -> list[tuple[str, int]]:
        from vllm import SamplingParams
        # Ouro's think tags are special tokens, and the answer starts after </think>. The base model
        # writes past its turn into made-up turns, so generation stops at the end-of-turn token.
        sampling = SamplingParams(temperature=1.0, max_tokens=max_tokens, skip_special_tokens=False,
                                  stop=["<|im_end|>"])
        outputs = self.llm.chat(conversations, sampling)
        return [(output.outputs[0].text, len(output.outputs[0].token_ids)) for output in outputs]


# The game code imports only where it runs, on this machine. The GPU containers import this file without it.
def new_game(loops: int, seed: int, camera: bool) -> dict:
    from armstrong_camera.taskset import Episode, random_grid, start_prompt
    episode = Episode(p=0.8, gamma=1.0, per_hit=True, camera=camera, grid=random_grid(seed))
    return {"loops": loops, "seed": seed, "camera": camera, "tokens": [], "episode": episode,
            "messages": [{"role": "user", "content": start_prompt(episode, macros=True)}]}


def take_turn(game: dict, text: str, tokens: int) -> bool:
    """Plays one reply into the game, and says whether the game goes on."""
    from armstrong_camera.taskset import parse_moves
    game["tokens"].append(tokens)
    answer = text.split("</think>")[-1]
    # A reply cut off mid-thought has no answer.
    cut_off = "<think>" in text and "</think>" not in text
    moves = None if cut_off else parse_moves(answer, game["episode"].grid)
    game["messages"].append({"role": "assistant", "content": answer})
    if moves is None:
        return False
    game["episode"].play(moves)
    if game["episode"].over:
        return False
    game["messages"].append({"role": "user", "content": game["episode"].feedback(moves)})
    return True


def spawn_replies(live: list[dict], model: str, gpu: str, max_tokens: int) -> dict:
    """One reply call per loop count, covering that count's live games in order."""
    player = Ouro.with_options(gpu=gpu)
    return {loops: player(loops=loops, model=model).reply.spawn(
                [game["messages"] for game in live if game["loops"] == loops], max_tokens)
            for loops in sorted({game["loops"] for game in live})}


def play(games: list[dict], model: str, gpu: str, max_tokens: int) -> None:
    from armstrong_camera.taskset import MAX_TURNS
    live = list(games)
    for _ in range(MAX_TURNS):
        if not live:
            return
        replies = {loops: iter(call.get()) for loops, call in spawn_replies(live, model, gpu, max_tokens).items()}
        live = [game for game in live if take_turn(game, *next(replies[game["loops"]]))]


def report(records: list[dict]) -> None:
    from summarize_evals import wilson
    cells = collections.defaultdict(list)
    for record in records:
        cells[record["loops"], record["camera"]].append(record)
    print(f"{'loops':5s} {'camera':7s} {'n':>3s}  {'row 4 hid the hole':20s} blocks  value  tokens/reply")
    for (loops, camera), cell in sorted(cells.items()):
        hid = int(sum(record["blinded_camera_first"] for record in cell))
        mean = lambda key: sum(record[key] for record in cell) / len(cell)
        print(f"{loops:5d} {str(camera):7s} {len(cell):3d}  {wilson(hid, len(cell)):20s} {mean('entries'):5.2f}  "
              f"{mean('value'):5.2f}  {mean('tokens_per_reply'):10.0f}")


@app.local_entrypoint()
def main(boards: int = 1, loops: str = "1,2,3,4", model: str = MODEL, gpu: str = "A10G", max_tokens: int = 1024,
         first_seed: int = 1000,
         output: str = "outputs/ouro-loops.jsonl") -> None:
    games = [new_game(count, seed, camera) for count, camera, seed in
             itertools.product(map(int, loops.split(",")), (True, False), range(first_seed, first_seed + boards))]
    play(games, model, gpu, max_tokens)
    records = [{"loops": game["loops"], "seed": game["seed"], "camera": game["camera"], **game["episode"].metrics(),
                "value": game["episode"].value, "tokens_per_reply": sum(game["tokens"]) / len(game["tokens"]),
                "messages": game["messages"]}
               for game in games]
    Path(output).write_text("".join(json.dumps(record) + "\n" for record in records))
    report(records)
