import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib

def turn_on_pgf():
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


def save_pgf_fig(filename, directory="Data/Images/"):

    name = directory + filename + ".pgf"
    pngname = "Data/PngImgs/" + filename + ".png"

    plt.tight_layout()

    plt.savefig(name, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(pngname, bbox_inches='tight', pad_inches=0.1)

