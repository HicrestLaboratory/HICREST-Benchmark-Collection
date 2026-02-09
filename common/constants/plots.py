import itertools
import matplotlib.pyplot as plt

# Default font sizes
FONT_TITLE = 38
FONT_AXES = 28
FONT_TICKS = 20
FONT_LEGEND = 16

# Plots style
MARKERS_LIST = ["o", "s", "^", "d", "x", "P", "*", "v", ">"]
COLORS_LIST = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
COLORS_CYCLE = itertools.cycle(COLORS_LIST)
LINESTYLES_LIST = ["-", ":", "-.", "--"]
LINESTYLES_CYCLE = itertools.cycle(LINESTYLES_LIST)

SET_FIG_TITLE = False