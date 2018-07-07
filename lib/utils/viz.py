#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_im(im):
    plt.figure()
    plt.imshow(im)

def draw_bbx(ax, x, y, size):
    rect = patches.Rectangle(
        (x, y), size, size, edgecolor='r', facecolor='none', linewidth=2)
    ax.add_patch(rect)