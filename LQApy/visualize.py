import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np


def plot_annotations(image, annotations):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    for anno in annotations:
        # Draw Masks
        polygon = anno['segmentation'][0]
        polygon = np.asarray(polygon).reshape(-1, 2)
        patch = Polygon(
            polygon,
            closed=True,
            facecolor="blue",
            edgecolor="none",
            alpha=0.5,
        )
        ax.add_patch(patch)
        # Draw Boxes
        box = anno['box']
        x1, y1, x2, y2 = box
        rect = Rectangle(
            (x1, y1),
            x2-x1,
            y2-y1,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.axis("off")
    plt.show()
