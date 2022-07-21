import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import csv

path = "Data/Vehicles/BaselineSSS/BaselineSSS_levine_blocked_4_0_0/"

def moving_average(data, period):
    ret = np.convolve(data, np.ones(period), 'same') / period
    for i in range(period):
        # t = np.mean
        t = np.convolve(data, np.ones(i+1), 'valid') / (i+1)
        ret[i] = t[0]
        ret[-i-1] = t[-1]
    return ret

def load_csv_reward():
    rewards, lengths = [], []
    with open(f"{path}training_data.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if float(row[2]) > 0:
                rewards.append(float(row[1]))
                lengths.append(float(row[2]))
    return rewards, lengths

from matplotlib.ticker import MultipleLocator
def plot_reward_steps():
    rewards, lengths = load_csv_reward()

    plt.figure(1)
    steps = np.cumsum(lengths) / 100
    plt.plot(steps, rewards, '.', color='darkblue', markersize=4)
    plt.plot(steps, moving_average(rewards, 20), linewidth='4', color='r')
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(1))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Reward per Episode")

    plt.tight_layout()
    plt.grid()

    tikzplotlib.save(path + "baseline_reward_plot.tex", strict=True, extra_axis_parameters=['height=4cm', 'width=0.5\\textwidth', 'clip mode=individual'])

    plt.show()


    plt.show()

plot_reward_steps()