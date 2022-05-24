from RacingRewards.RewardSignals.DistanceReward import DistanceReward

import numpy as np
from matplotlib import pyplot as plt
from argparse import Namespace
from PIL import Image


def get_pt(x, y):
    i = (x  - 11.6)
    j = y - 26.52

    return np.array([i, j])


map_name = "berlin"



reward = DistanceReward(Namespace(**{"map_name": map_name}))

map_img_path = "maps/" + map_name + ".png"
map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
map_img = map_img.astype(np.float64)

x_size, y_size = 100, 120
xs = np.linspace(0, 25, x_size)
ys = np.linspace(0, 30, y_size)
grid = np.meshgrid(xs, ys)

grid = np.zeros((x_size, y_size))

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        pt = get_pt(x, y)
        reward_val = reward.find_s(pt)
        grid[i, j] = reward_val

plt.figure(1)
# plt.imshow(map_img)
plt.imshow(grid)

plt.show()




