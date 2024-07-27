from typing import Dict

import torch
from torch.utils.data import TensorDataset

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# custom color map
custom_colors = {
    -1: '#222222',  # blank lines
    0: '#000000',   # Black
    1: '#e122a1',   # Pink
    2: '#22e1a1',   # Mint
    3: '#2261e1',   # Blue
    4: '#e16122',   # Orange
    5: '#61e122',   # Lime
    6: '#a122e1',   # Purple
    7: '#e1a122',   # Gold
    8: '#22a1e1',   # Sky Blue
    9: '#c12261',   # Red
    10: '#22c161',
}

custom_cmap = mcolors.ListedColormap([custom_colors[i] for i in sorted(custom_colors.keys())])
norm = mcolors.Normalize(vmin=min(custom_colors.keys()), vmax=max(custom_colors.keys()))


def visualize_datasets(datasets: Dict[str, TensorDataset], grid_width: int, grid_height: int, num_samples: int = 10):
    num_datasets = len(datasets)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(5 * grid_width, 5 * grid_height))
    fig.tight_layout(pad=3.0)

    for i, (dataset_name, dataset) in enumerate(datasets.items()):
        if i >= grid_width * grid_height:
            print("Warning: Not all datasets are displayed. Increase grid size to show all.")
            break

        row = i // grid_width
        col = i % grid_width
        ax = axs[row, col] if grid_height > 1 else axs[col]

        samples = [dataset[j] for j in range(min(num_samples, len(dataset)))]

        # interleave input output pairs for display
        interleaved_data = []
        for input_seq, output_seq in samples:
            blank_line = torch.full_like(input_seq, -1)
            interleaved_data.extend([input_seq, output_seq, blank_line])

        # rm last blank line
        interleaved_data = interleaved_data[:-1]
        interleaved_data = torch.stack(interleaved_data).numpy()

        im = ax.imshow(interleaved_data, cmap=custom_cmap, aspect='auto', norm=norm, interpolation='nearest')
        ax.set_title(f'{dataset_name}', fontsize=10)
        ax.set_yticks(range(1, len(interleaved_data), 3))
        ax.set_yticklabels([f'S{j+1}' for j in range(num_samples)], fontsize=8)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Remove any unused subplots
    for i in range(num_datasets, grid_width * grid_height):
        row = i // grid_width
        col = i % grid_width
        fig.delaxes(axs[row, col] if grid_height > 1 else axs[col])

    plt.colorbar(im, ax=axs, label='Digit Value', aspect=30, ticks=range(-1, 10)) # type: ignore
    plt.show()
