from typing import Callable, List, Any
import random
from torch.utils.data import TensorDataset
import torch

from visualization import visualize_datasets


SequenceTransformer = Callable[[List[Any]], List[Any]]


def compose_transformers(transformers: List[SequenceTransformer]) -> SequenceTransformer:
    """ Compose multiple sequence transformers into a single transformer. """
    def composed_transformer(seq: List[int]) -> List[int]:
        for transformer in transformers:
            seq = transformer(seq)
        return seq
    return composed_transformer


def create_transformer(func: Callable[[List[int], int], int]) -> SequenceTransformer:
    """ Create a sequence transformer from a function that takes a sequence and an index as input. """
    def transformer(seq: List[int]) -> List[int]:
        return [func(seq, i) for i in range(len(seq))]
    return transformer


def translate(n: int) -> SequenceTransformer:
    """ Translate the sequence by n positions. """
    return create_transformer(lambda seq, i: seq[(i - n) % len(seq)])


def reflect(i_pivot: int) -> SequenceTransformer:
    """ Reflect the sequence about the index i_pivot. """
    return create_transformer(lambda seq, i: seq[(-(i - i_pivot) + i_pivot) % len(seq)])


def colorshift(n: int) -> SequenceTransformer:
    """ Shift the color of each element in the sequence by n. """
    return create_transformer(lambda seq, i: seq[i] + n if seq[i] != 0 else seq[i])


def shrink(seq: List[int]) -> List[int]:
    """For each span of consecutive instances of a specific value, map the
    midpoint of the range to that value and the rest of the range to 0."""
    new_seq = [0] * len(seq)
    spans = []
    current_span = {"start": 0, "val": seq[0], "len": 1}

    for i, v in enumerate(seq[1:], 1):
        if v == current_span["val"]:
            current_span["len"] += 1
        else:
            spans.append(current_span)
            current_span = {"start": i, "val": v, "len": 1}
    spans.append(current_span)

    for span in spans:
        mid = span["start"] + span["len"] // 2
        new_seq[mid % len(seq)] = span["val"]

    return new_seq


def expand(n: int) -> SequenceTransformer:
    """Expand each value in the input sequence to fill each pixel within `n` of it
    in the output sequence. """
    def transformer(seq: List[int]) -> List[int]:
        def mode_non_bg(s: List[int]):
            counts = [(x, s.count(x)) for x in set(s) if x != 0]
            counts.sort(key=lambda x_count: x_count[1], reverse=True)
            return counts[0][0] if counts else 0

        return [
            mode_non_bg([seq[j % len(seq)] for j in range(i - n, i + n + 1)])
            for i in range(len(seq))
        ]

    return transformer


def endpoints(seq: List[int]) -> List[int]:
    ''' Identify the start/end of each block '''
    new_seq = [0] * len(seq)
    spans = []
    current_span = {"start": 0, "val": seq[0], "len": 1}
    for i, v in enumerate(seq[1:]):
        if v == current_span["val"]:
            current_span["len"] += 1
        else:
            spans.append(current_span)
            current_span = {"start": i + 1, "val": v, "len": 1}
    spans.append(current_span)
    for span in spans:
        end0 = span["start"]
        new_seq[end0 % len(seq)] = span["val"]
        end1 = span["start"] + span["len"] - 1
        new_seq[end1 % len(seq)] = span["val"]
    return new_seq


def gen_some_blocks(colors: List[int]) -> List[int]:
    """ Generate a sequence of blocks with random colors. """
    t = 48 // 6
    n = random.randrange(t, 2 * t)
    return translate(n)([
        colors[0 % len(colors)] if i in range(0 * t, 1 * t) else
        colors[1 % len(colors)] if i in range(3 * t, 4 * t) else
        0
        for i in range(48)
    ])


def gen_one_block(colors: int) -> List[int]:
    """ Generate a sequence of a single block with a random color. """
    t = 48 // 3
    n = random.randrange(1 * t, 1.5 * t)
    color = random.choice(colors)
    return translate(n)([
        color if i in range(0 * t, 1 * t) else
        0
        for i in range(48)
    ])


##########
# Demo

random.seed(42)

colors = [1, 2, 3, 4, 6, 7, 8, 9]

puzzles = [
    ('translate(4)', translate(4), gen_some_blocks),
    ('reflect(seq_len//2)', reflect(24), gen_some_blocks),
    ('colorshift(2)', colorshift(2), gen_some_blocks),
    ('translate(4) + reflect(seq_len//2)', compose_transformers([translate(4), reflect(24)]), gen_some_blocks),
    ('translate(4) + colorshift(2)', compose_transformers([translate(4), colorshift(2)]), gen_some_blocks),
    ('expand(1)', expand(1), gen_some_blocks),
    ('expand(1) expand(1)', compose_transformers([expand(1), expand(1)]), gen_some_blocks),
    ('expand(1) + colorshift(2)', compose_transformers([expand(1), colorshift(2)]), gen_some_blocks),
    ('expand(1) + translate(1)', compose_transformers([expand(1), translate(1)]), gen_some_blocks),
    ('shrink', shrink, gen_some_blocks),
    ('shrink + expand(2)', compose_transformers([shrink, expand(2)]), gen_some_blocks),
    ('endpoints', endpoints, gen_some_blocks),
    ('expand(1) + endpoints', compose_transformers([expand(1), endpoints]), gen_one_block),
    ('endpoints + expand(1)', compose_transformers([endpoints, expand(1)]), gen_one_block),
    ('endpoints + expand(4) + endpoints + expand(1)', compose_transformers([endpoints, expand(4), endpoints, expand(1)]), gen_one_block)
]

datasets = {}
num_samples = 10
grid_width = 4
grid_height = 5
for (name, transformer, gen) in puzzles:
    inputs, outputs = [], []
    for _ in range(num_samples):
        input = gen(colors)
        output = transformer(input)
        inputs.append(input)
        outputs.append(output)
        inputs_tensor, outputs_tensor = torch.tensor(inputs), torch.tensor(outputs)
        datasets[name] = TensorDataset(inputs_tensor, outputs_tensor)

visualize_datasets(datasets, grid_width=grid_width, grid_height=grid_height, num_samples=num_samples)
