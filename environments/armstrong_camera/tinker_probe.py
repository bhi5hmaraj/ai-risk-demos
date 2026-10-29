"""First-turn replies from the base model, to size the token cap a thinking run needs and to
read whether its reasoning reaches the camera-blinding plan.

Run with: uv run --with tinker-cookbook==0.5.9 python tinker_probe.py [samples=8 max_tokens=16384]
Needs TINKER_API_KEY. Each reply, thinking included, goes to one line of the output file.
"""
import json
from pathlib import Path

import chz
import tinker
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from armstrong_camera.taskset import opening_prompt, parse_moves


@chz.chz
class ProbeConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5"
    p: float = 0.8
    gamma: float = 1.0
    samples: int = 8
    max_tokens: int = 16384
    output: str = "outputs/probe-thinking.jsonl"


def main(config: ProbeConfig) -> None:
    tokenizer = get_tokenizer(config.model_name)
    renderer = get_renderer(config.renderer_name, tokenizer)
    sampler = tinker.ServiceClient().create_sampling_client(base_model=config.model_name)
    prompt = renderer.build_generation_prompt([{"role": "user", "content": opening_prompt(config.p, config.gamma)}])
    params = tinker.SamplingParams(temperature=1.0, max_tokens=config.max_tokens, stop=renderer.get_stop_sequences())
    response = sampler.sample(prompt, num_samples=config.samples, sampling_params=params).result()
    with Path(config.output).open("w") as replies:
        for sequence in response.sequences:
            message, _ = renderer.parse_response(sequence.tokens)
            # A reply cut off mid-thought has no answer, so its brackets are all inside the thinking.
            moves = parse_moves(get_text_content(message)) if sequence.stop_reason == "stop" else None
            print(f"{len(sequence.tokens):6d} tokens  {sequence.stop_reason:6s}  moves {moves}")
            replies.write(json.dumps({"tokens": len(sequence.tokens), "stop_reason": sequence.stop_reason,
                                      "moves": moves, "text": tokenizer.decode(sequence.tokens)}) + "\n")


if __name__ == "__main__":
    main(chz.entrypoint(ProbeConfig))
