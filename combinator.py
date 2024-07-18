from abc import ABC, abstractmethod
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
seq_len: int = 48

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

class Combinator(ABC):
    description: str
    
    def __init__(self, name: str, apply: Puz):
        self.name_ = name
        self.apply = apply

    @property
    def name(self):
        return self.name_

    def __call__(self, seq: Seq) -> Any:
        self.apply(seq)
        
    @classmethod
    def sample(cls): 
        raise Exception()

class Composition(Combinator):
    description = """
    Composition of a sequence of combinators.
    """.strip()
    
    def __init__(self, cs: list[Combinator]):
        super().__init__(
            "; ".join([ c.name for c in cs ]),
            fold_thn([ c.apply for c in cs ])
        )
        
    @classmethod
    def sample(cls):
        return Composition([])

class Translate(Combinator):
    description = """
    Translates the sequence to the right by `n`. Wraps.
    """.strip()

    def __init__(self, n: int):
        super().__init__(
            f"translate(n={n})",
            translate(n)
        )
        
    @classmethod
    def sample(cls):
        return cls(random.randint(-2, 2))

def translate(n: int) -> Puz:
    return bijective_puz(lambda s, i: s[(i - n) % seq_len])

class Reflect(Combinator):
    description = """
    Reflects the sequence about the index `i_pivot`. Wraps.
    """.strip()
    
    def __init__(self, i_pivot: int):
        super().__init__(
            f"reflect(i_pivot={i_pivot})",
            bijective_puz(lambda s, i: s[(-(i - i_pivot) + i_pivot) % seq_len])
        )
        
    @classmethod
    def sample(cls):
        return cls(seq_len//2)

class Colorshift(Combinator):
    description = """
    Adds `n` to the color of each non-background element.
    """.strip()
    
    def __init__(self, n: int):
        super().__init__(
            f"colorshift(n={n})",
            bijective_puz(lambda s, i: s[i] + n if s[i] != bg else s[i])
        )

    @classmethod        
    def sample(cls):
        return cls(random.sample([-1, 2, 3], 1)[0])

class Expand(Combinator):
    description = """
    Expand each value in the input sequence to fill each pixel within `n` of it
    in the output sequence.
    """.strip()
    
    def __init__(self, n: int):
        def mode_non_bg(s: Seq):
            """
            Returns the most common non-`bg` value in `s`.
            If `s` is all `bg`, then return `bg`.
            """
            counts: list[tuple[Val, int]] = [ (x, s.count(s)) for x in set(s) if x != bg ]
            counts.sort(key=lambda x_count: x_count[1])
            return counts[0][0] if len(counts) != 0 else bg
        super().__init__(
            f"expand(n={n})",
            lambda s: [
                mode_non_bg([ s[j % seq_len] for j in range(i - n, i + n + 1) ])
                for i in range(seq_len)
            ]
        )
        
    @classmethod
    def sample(cls):
        return cls(random.randint(1, 4))

class Shrink(Combinator):
    description = """
    For each span of consecutive instances of a specific value `v`, map the
    midpoint of the range to `v` and the rest of the range to `bg`. rest of the
    range mapped to `bg`.
    """

    def __init__(self):
        def puz(s: Seq) -> Seq:
            s_new = bg_seq()
            # collect spans
            spans: list[dict[str, Any]] = []
            current_span = { "start": 0, "val": s[0], "len": 1 }
            for (i, v) in enumerate(s[1:]):
                if v == current_span["val"]:
                    current_span["len"] += 1
                else:
                    spans.append(current_span)
                    current_span = { "start": i, "val": v, "len": 1 }
            # in `s_new`, set midpoint of each span to its val
            for span in spans:
                mid = span["start"] + span["len"]//2 + 1
                s_new[mid % seq_len] = span["val"]
            return s_new

        super().__init__("shrink", puz)

    @classmethod
    def sample(cls):
        return cls()

class Endpoints(Combinator):
    def __init__(self):
        def puz(s: Seq) -> Seq:
            s_new = bg_seq()
            # collect spans
            spans: list[dict[str, Any]] = []
            current_span = { "start": 0, "val": s[0], "len": 1 }
            for (i, v) in enumerate(s[1:]):
                if v == current_span["val"]:
                    current_span["len"] += 1
                else:
                    spans.append(current_span)
                    current_span = { "start": i, "val": v, "len": 1 }
            # in `s_new`, set endpoints of each span to its val
            for span in spans:
                end0 = span["start"]
                s_new[end0 + 1 % seq_len] = span["val"]
                end1 = span["start"] + span["len"]
                s_new[end1 % seq_len] = span["val"]
            return s_new
        super().__init__("endpoints", puz)

    @classmethod
    def sample(cls):
        return cls()

# ==============================================================================
# main
# ==============================================================================

if __name__ == "__main__":
    num_samples = 10

    def gen_some_pixels(colors=[1,2,3,4]) -> Seq:
        random.shuffle(colors)
        t = seq_len//4
        n = random.randrange(t//2, t)
        return translate(n)([
            colors[0 % len(colors)] if i == 0*t else
            colors[1 % len(colors)] if i == 1*t else
            colors[2 % len(colors)] if i == 2*t else
            colors[3 % len(colors)] if i == 3*t  else
            bg
            for i in range(seq_len)
        ])

    def gen_some_blocks(colors=[1,4]) -> Seq:
        """
        Randomly generate a sequence to use as input to a `Puz`.
        """
        random.shuffle(colors)
        t = seq_len//6
        n = random.randrange(t, 2*t)
        return translate(n)([
            colors[0 % len(colors)]   if i in range(0*t, 1*t) else
            colors[1 % len(colors)]   if i in range(3*t, 4*t) else
            bg
            for i in range(seq_len)
        ])
        
    def gen_one_block(color=2) -> Callable[[], Seq]:
        def go():
            t = seq_len//3
            n = random.randrange(1*t, 1.5*t)
            return translate(n)([
                color   if i in range(0*t, 1*t) else
                bg
                for i in range(seq_len)
            ])
        return go

    grid_width = 4
    grid_height = 5

    puzzles: list[Puz, Callable[[], Seq]] = []

    autogen = True
    if autogen:
        for i in range(grid_width * grid_height):
            c_options: list[Combinator] = [ 
                Translate,
                Reflect,
                Colorshift,
                Expand,
                Shrink,
                Endpoints
            ]
            random.shuffle(c_options)
            cs = [ c.sample() for c in c_options[0:3] ]
            # puzzles.append((Composition(cs), gen_some_blocks))
            puzzles.append((Composition(cs), gen_one_block(color=random.randint(2, 6))))
            
    else:
        puzzles = [
            (   Composition([ Translate(4) ]), 
                gen_some_blocks
            ),
            (   Composition([ Reflect(seq_len//2) ]), 
                gen_some_blocks
            ),
            (   Composition([ Colorshift(2) ]),
                gen_some_blocks
            ),
            (   Composition([ Translate(4), Reflect(seq_len//2) ]),
                gen_some_blocks
            ),
            (   Composition([ Translate(4), Colorshift(2) ]),
                gen_some_blocks
            ),
            (   Composition([ Expand(1)]),
                gen_some_pixels
            ),
            (   Composition([ Expand(1), Colorshift(2) ]),
                gen_some_pixels
            ),
            (   Composition([ Expand(1), Translate(1) ]),
                gen_some_pixels
            ),
            (   Composition([ Shrink() ]),
                gen_some_blocks
            ),
            (   Composition([ Shrink(), Expand(2) ]),
                gen_some_blocks
            ),
            (   Composition([ Endpoints() ]),
                gen_some_blocks
            ),
            (   Composition([ Endpoints(), Expand(1) ]),
                gen_one_block()
            ),
            (   Composition([ Endpoints(), Expand(4), Endpoints(), Expand(1) ]),
                gen_one_block()
            ),
        ]

    datasets = {}
    for (comb, gen) in puzzles:
        inputs, outputs = [], []
        for _ in range(num_samples):
            input = gen()
            output = comb.apply(input)
            inputs.append(input)
            outputs.append(output)
        inputs_tensor, outputs_tensor = torch.tensor(inputs), torch.tensor(outputs)
        datasets[comb.name] = TensorDataset(inputs_tensor, outputs_tensor)

    visualize_datasets(datasets, grid_width=grid_width, grid_height=grid_height, num_samples=num_samples)
