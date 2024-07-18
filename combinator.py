import random
from random import randrange, choice, randint, shuffle, sample

from visualization import visualize_datasets
random.seed(777)

from typing import Any, Callable, Dict

import torch
from torch.utils.data import TensorDataset

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

'''
parameters
    Val: type
    bg: Val # background
    seq_len: int
definitions
    Seq = list[val, seq_len]
'''

Val = int
bg: Val = 0
seq_len: int = 16
Seq = list[Val]
Puz = Callable[[Seq], Seq]

'''
An idea for algorithmically creating new puzzles: define a collection of
combinators that, when combinated in well-formed way, yield a puzzle.

A *puzzle* in this context is a mapping `Seq -> Seq`. So if we have the
combinators
    - `x : A -> Seq -> Seq`,
    - `y : B -> Seq -> Seq`, and
    - `z : C -> Seq -> Seq`
then we can construct the puzzle `lambda seq: z(c, y(b, x(a, seq)))`.

I'm imagining that the combinators will be common sorts of transformations. But
is of coursean interesting question of how to classify terms of `Seq -> Seq`
into "good puzzles" and "bad puzzles".
'''

# ==============================================================================
# utilities
# ==============================================================================

def bg_seq() -> Seq:
    return [bg] * seq_len

def thn(g: Callable, f: Callable) -> Callable:
    """
    f then g
    """
    return lambda x: f(g(x))

def fold_thn(fs: list[ Callable[[Any], Any] ]) -> Callable[[Any], Any]:
    if len(fs) == 0:
        return lambda x: x
    else:
        return thn(fs[0], fold_thn(fs[1:]))

# ==============================================================================
# combinators
# ==============================================================================

def translate(n: int) -> Puz:
    """
    Translates the sequence to the right by `n`.
    Wraps.
    """
    def puz(s: Seq) -> Seq:
        s_new = bg_seq()
        for i in range(len(s)):
            s_new[i] = s[i - n % len(s)]
        return s_new
    return puz

def reflect(i_pivot: int) -> Puz:
    """
    Reflects the sequence about the index `i_pivot`.
    Wraps.
    """
    def puz(s: Seq) -> Seq:
        s_new = bg_seq()
        # shift s so that index i is at 0, then negate, then shift back, and
        # make sure to mod by len to properly wrap
        for i in range(len(s)):
            s_new[i] = s[(-(i - i_pivot) + i_pivot) % len(s)]
        return s_new
    return puz

def colorshift(n: int) -> Puz:
    """
    Adds `n` to the color of each non-background element.
    """
    def puz(s: Seq) -> Seq:
        s_new = bg_seq()
        for i in range(len(s)):
            s_new[i] = s[i] + n if s[i] != bg else bg
        return s_new
    return puz

# ==============================================================================
# main
# ==============================================================================

if __name__ == "__main__":
    num_samples = 10

    def generate_input() -> Seq:
        """
        Randomly generate a sequence to use as input to a `Puz`.
        """
        s = bg_seq()
        n = random.randrange(0, seq_len)
        for i in range(n, n + 4):
            s[i % seq_len] = 1
        for i in range(n + 8, n + 12):
            s[i % seq_len] = 2
        return s

    puzzles: dict[str, Puz] = {}
    puzzles["translate"] = fold_thn([translate(4)])
    puzzles["reflect"] = fold_thn([reflect(seq_len//2)])
    puzzles["colorshift"] = fold_thn([colorshift(2)])
    puzzles["translate; reflect"] = fold_thn([translate(4), reflect(seq_len//2)])
    puzzles["translate; colorshift"] = fold_thn([translate(4), colorshift(2)])

    datasets = {}
    for name, puz in puzzles.items():
        inputs, outputs = [], []
        for _ in range(num_samples):
            input = generate_input()
            output = puz(input)
            inputs.append(input)
            outputs.append(output)
        inputs_tensor, outputs_tensor = torch.tensor(inputs), torch.tensor(outputs)
        datasets[name] = TensorDataset(inputs_tensor, outputs_tensor)

    visualize_datasets(datasets, grid_width=4, grid_height=7, num_samples=num_samples)
