from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_masks(segments: List[np.ndarray], fig = None, color = [0,0,1], alpha = 1):
    if fig:
        fig = fig
        ax = fig.gca()
    else:
        fig = plt.figure(figsize = (10,10))
        ax = fig.gca()
        
    for i,label in enumerate(segments):
        polygon = Polygon([(x,y) for x,y in zip(label[0::2],label[1::2])], alpha)
        polygon.set_color(color)
        polygon.set_alpha(alpha)
        ax.add_patch(polygon)
        plt.ylim(0,1024)
        plt.xlim(0,1024)
    return fig