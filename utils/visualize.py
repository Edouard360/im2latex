import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mathtext import MathTextParser

vocab = open('data/latex_vocab.txt').read().split('\n')
vocab_to_idx = dict([(vocab[i], i) for i in range(len(vocab))])


def convert_to_formula(labels):
    return ' '.join([vocab[i - 4] for i in labels if i >= 4])


def display_result(img, gt_labels, predicted_labels):
    # Usage: print_result(images[0],labels[0],predicted_labels[0])

    fig, axes = plt.subplots(3)
    for ax in axes:
        ax.set_xticks(());
        ax.set_yticks(())
    axes[0].set_title("Input image")
    axes[0].imshow(img[:, :, 0], cmap='gray')

    check_parser = MathTextParser('MacOSX')

    kwargs = {'x': 0.5, 'y': 0.5, 'size': 30, 'verticalalignment': 'center', 'horizontalalignment': 'center'}

    axes[1].set_title("Prediction")
    predicted_formula = "$" + convert_to_formula(predicted_labels) + "$"
    try:
        check_parser.parse(predicted_formula)
        axes[1].text(s=predicted_formula, **kwargs)
    except ValueError:
        kwargs['size'] = 15
        axes[1].text(s=predicted_formula[1:-1].replace(' ', ''), **kwargs)

    axes[2].set_title("[ Ground Truth ]")
    gt_formula = "$" + convert_to_formula(gt_labels) + "$"
    try:
        check_parser.parse(gt_formula)
        axes[2].text(s=gt_formula, **kwargs)
    except ValueError:
        axes[2].text(s=gt_formula[1:-1].replace(' ', ''), **kwargs)
    plt.show()


def plot_bbox(ax, bbox, score):
    """Plot bounding-box on matplotlib ax"""
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=2)
    )
    if score > 0.5:
        ax.text(bbox[0], bbox[1] - 2,
                '{:.3f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')


def plot_attention(image, predicted_labels, rf_coords, alignment_history, time=0):
    """Plot the highest attention region for image at time t in the decoding"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].set_title("Highest alignment score")
    axes[0].imshow(np.squeeze(image), cmap='gray')
    rf_coords_flat = np.reshape(rf_coords, (-1, 4))

    idx = alignment_history.argmax(axis=1)
    i, score = idx[time], alignment_history[np.arange(len(alignment_history)), idx][time]
    bbox = rf_coords_flat[i]
    plot_bbox(axes[0], bbox, score)

    check_parser = MathTextParser('MacOSX')

    kwargs = {'x': 0.0, 'y': 0.5, 'size': 30, 'verticalalignment': 'center', 'horizontalalignment': 'left'}

    axes[1].set_title("Prediction so far")
    predicted_formula = "$" + convert_to_formula(predicted_labels[0][:time]) + "$"
    try:
        check_parser.parse(predicted_formula)
        axes[1].text(s=predicted_formula, **kwargs)
    except ValueError:
        kwargs['size'] = 15
        axes[1].text(s=predicted_formula[1:-1].replace(' ', ''), **kwargs)
