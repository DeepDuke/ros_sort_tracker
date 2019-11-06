#!/usr/bin/env python
from __future__ import print_function
import random


def get_colors(n):
    """
    @param n: number of OpenCV format colors to generate
    """
    random.seed(1)
    colors = []
    R = int(random.random() * 256)
    G = int(random.random() * 256)
    B = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        R += step
        G += step
        B += step
        R = int(R) % 256
        G = int(G) % 256
        B = int(B) % 256
        # colors.append((R, G, B))
        colors.append((B, G, R))
    return colors


if __name__ == '__main__':
    random.seed(1)
    colors = get_colors(80)
    for idx, color in enumerate(colors):
        print('{}\t{}'.format(idx, color))