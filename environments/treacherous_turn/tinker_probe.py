"""First replies from the base model on the start board, while weak and once Link has the bow.
They show whether its plans ever shoot the shopkeeper or use the heart machine.

Run with: uv run --no-project --with tinker-cookbook==0.5.9 python tinker_probe.py [samples=16]
Needs TINKER_API_KEY. Each reply goes to one line of the output file, with the events its actions cause.
"""
import copy
import json
from pathlib import Path

import chz
import tinker
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from game import BOW_STEP, Episode, parse_actions, prompt


@chz.chz
class ProbeConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    samples: int = 16
    max_tokens: int = 2048
    output: str = "outputs/probe.jsonl"


def main(config: ProbeConfig) -> None:
    tokenizer = get_tokenizer(config.model_name)
    renderer = get_renderer(config.renderer_name, tokenizer)
    sampler = tinker.ServiceClient().create_sampling_client(base_model=config.model_name)
    params = tinker.SamplingParams(temperature=1.0, max_tokens=config.max_tokens, stop=renderer.get_stop_sequences())
    Path(config.output).parent.mkdir(exist_ok=True)
    with Path(config.output).open("w") as replies:
        for phase, start in (("weak", 0), ("strong", BOW_STEP)):
            board = Episode(actions=start)
            request = renderer.build_generation_prompt([{"role": "user", "content": prompt(board)}])
            response = sampler.sample(request, num_samples=config.samples, sampling_params=params).result()
            for sequence in response.sequences:
                message, _ = renderer.parse_response(sequence.tokens)
                actions = parse_actions(get_text_content(message)) if sequence.stop_reason == "stop" else None
                episode = copy.deepcopy(board)
                episode.play(actions or [])
                events = {name: count for name, count in episode.events.items() if count}
                print(f"{phase:6s} {len(sequence.tokens):5d} tokens  {sequence.stop_reason:6s}  "
                      f"{len(actions or []):3d} actions  {events}")
                replies.write(json.dumps({"phase": phase, "tokens": len(sequence.tokens), "actions": actions,
                                          "events": events, "text": tokenizer.decode(sequence.tokens)}) + "\n")


if __name__ == "__main__":
    main(chz.entrypoint(ProbeConfig))
