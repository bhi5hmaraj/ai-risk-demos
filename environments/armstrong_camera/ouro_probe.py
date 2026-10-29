"""First replies from Ouro, a looped language model, on the per-hit Armstrong opening, at each loop count.
The loop count sets how much latent compute the model spends per token, with no other change.

Run with: uv run --with "transformers==4.57.1" --with torch --with chz python ouro_probe.py [samples=8 max_new_tokens=2048]
Runs locally on the Mac's GPU. Each reply, thinking included, goes to one line of the output file.
"""
import json
from pathlib import Path

import chz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from armstrong_camera.taskset import Episode, parse_moves, start_prompt


@chz.chz
class ProbeConfig:
    model_name: str = "ByteDance/Ouro-1.4B-Thinking"
    max_loops: int = 4
    samples: int = 8
    max_new_tokens: int = 2048
    output: str = "outputs/ouro-probe.jsonl"


def main(config: ProbeConfig) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).to(device)
    opening = start_prompt(Episode(p=0.8, gamma=1.0, per_hit=True), macros=True)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": opening}], add_generation_prompt=True,
                                           return_tensors="pt").to(device)
    with Path(config.output).open("w") as replies:
        for loops in range(1, config.max_loops + 1):
            model.config.total_ut_steps = model.model.total_ut_steps = loops
            generated = model.generate(prompt, do_sample=True, temperature=1.0, max_new_tokens=config.max_new_tokens,
                                       num_return_sequences=config.samples)
            for sequence in generated[:, prompt.shape[1]:]:
                text = tokenizer.decode(sequence, skip_special_tokens=True)
                answer = text.split("</think>")[-1] if "</think>" in text else ""
                episode = Episode(p=0.8, gamma=1.0, per_hit=True)
                moves = parse_moves(answer, episode.grid)
                episode.play(moves or "")
                blinded = episode.grid.camera_blocked() if episode.entries == 0 else bool(episode.first_entry_blocked)
                print(f"loops {loops}  thought {'done' if answer else 'cut off'}  moves {moves}  "
                      f"blinds first {blinded}  blocks {episode.entries}", flush=True)
                replies.write(json.dumps({"loops": loops, "moves": moves, "blinds_first": blinded,
                                          "entries": episode.entries, "text": text}) + "\n")


if __name__ == "__main__":
    main(chz.entrypoint(ProbeConfig))
