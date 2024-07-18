import torch
from torch.utils.data import TensorDataset
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random
from random import randrange, choice, randint, shuffle, sample
from typing import Any, Callable

from visualization import visualize_datasets

random.seed(777)

# ==============================================================================
# parameters
# ==============================================================================

Val = int
bg: Val = 0 # background value

Seq = list[Val] # type of sequences
seq_len: int = 16

Puz = Callable[[Seq], Seq] # type of puzzles

'''
An idea for algorithmically creating new puzzles: define a collection of
combinators that, when combinated in well-formed way, yield a puzzle.

A *puzzle* in this context is a mapping `Seq -> Seq`. So if we have the
combinators
    - `x : A -> Seq -> Seq`,
    - `y : B -> Seq -> Seq`, and
    - `z : C -> Seq -> Seq`
then we can construct the puzzle `lambda s: z(c, y(b, x(a, s)))`.

I'm imagining that the combinators will be common sorts of transformations. But
is of course an interesting question of how to classify terms of `Seq -> Seq`
into "good puzzles" and "bad puzzles".
'''

# ==============================================================================
# utilities
# ==============================================================================

def bg_seq() -> Seq:
    return [bg] * seq_len

def thn(g: Callable, f: Callable) -> Callable:
    """
    g then f
    """
    return lambda x: f(g(x))

def fold_thn(fs: list[ Callable[[Any], Any] ]) -> Callable[[Any], Any]:
    """
    f_1 then ... then f_n i.e.
        fold_thn([f_1, ..., f_n]) = lambda x: f_n(... f_1(x))

    Note that
        fold_thn([]) = lambda x: x
    """
    if len(fs) == 0:
        return lambda x: x
    else:
        return thn(fs[0], fold_thn(fs[1:]))

def bijective_puz(f: Callable[[Seq, int], Val]) -> Puz:
    return lambda s: [ f(s, i) for i in range(seq_len) ]

# ==============================================================================
# combinators
# ==============================================================================

def translate(n: int) -> Puz:
    """
    Translates the sequence to the right by `n`.
    Wraps.
    """
    return bijective_puz(lambda s, i: s[(i - n) % seq_len])

def reflect(i_pivot: int) -> Puz:
    """
    Reflects the sequence about the index `i_pivot`.
    Wraps.
    """
    return bijective_puz(lambda s, i: s[(-(i - i_pivot) + i_pivot) % seq_len])

def colorshift(n: int) -> Puz:
    """
    Adds `n` to the color of each non-background element.
    """
    return bijective_puz(lambda s, i: s[i] + n if s[i] != bg else s[i])

# ==============================================================================
# main
# ==============================================================================

if __name__ == "__main__":
    num_samples = 10

    def gen_two_blocks() -> Seq:
        """
        Randomly generate a sequence to use as input to a `Puz`.
        """
        n = random.randrange(0, seq_len)
        return translate(n)([
            1   if i in range(0, 4) else 
            2   if i in range(8, 12) else 
            bg
            
            for i in range(seq_len)
        ])

    puzzles: dict[str, tuple[Puz, Callable[[], Seq]]] = {
        "translate": (fold_thn([translate(4)]), gen_two_blocks),
        "reflect": (fold_thn([reflect(seq_len//2)]), gen_two_blocks),
        "colorshift": (fold_thn([colorshift(2)]), gen_two_blocks),
        "translate; reflect": (fold_thn([translate(4), reflect(seq_len//2)]), gen_two_blocks),
        "translate; colorshift": (fold_thn([translate(4), colorshift(2)]), gen_two_blocks),
    }

    datasets = {}
    for name, (puz, gen) in puzzles.items():
        inputs, outputs = [], []
        for _ in range(num_samples):
            input = gen()
            output = puz(input)
            inputs.append(input)
            outputs.append(output)
        inputs_tensor, outputs_tensor = torch.tensor(inputs), torch.tensor(outputs)
        datasets[name] = TensorDataset(inputs_tensor, outputs_tensor)

    visualize_datasets(datasets, grid_width=4, grid_height=7, num_samples=num_samples)
