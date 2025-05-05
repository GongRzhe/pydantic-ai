from dataclasses import dataclass, field

from .graph import Node


@dataclass
class Prompt[InputT, OutputT](Node[InputT, OutputT]):
    prompt: str
    input_type: type[InputT] = field(init=False)
    output_type: type[OutputT] = field(init=False)
