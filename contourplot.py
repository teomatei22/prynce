import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d, interp2d, dfitpack

def plot_neff_yp():
    def make_neff_yp(disp, color, window_size=300, disp_label=None):
        data = pd.read_csv(disp, sep=" ")
        # print(data.dtypes)
        STD_NEFF = 3.046
        yps = data["Yp(CMB)"]
        neffs = data["Neff"]
        # print(data)
        yps = np.array([float(x) for x in yps])
        neffs = np.array([float(x) for x in neffs])
        sorted_indices = np.argsort(yps)
        yps = yps[sorted_indices]
        neffs = neffs[sorted_indices]
        
        means = []
        x_means = []
        stds = []
        for i in range(0, len(yps), window_size):
            means.append(np.mean(neffs[i:i+window_size]))
            stds.append(np.std(neffs[i:i+window_size]))
            x_means.append(np.mean(yps[i:i+window_size]))

        mu_of_yp = interp1d(x_means, means, kind="cubic", bounds_error=False, fill_value="extrapolate")
        std_of_yp = interp1d(x_means, stds, kind="cubic", bounds_error=False, fill_value="extrapolate")
        xs = np.linspace(0.245, 0.248, 1000) 
        ys = mu_of_yp(xs) 
        std_ys = std_of_yp(xs)

        line = plt.plot(xs, ys - STD_NEFF, color=color, label=disp_label)
        plt.fill_between(xs, ys - std_ys - STD_NEFF, ys + std_ys - STD_NEFF, alpha=0.3, color=line[0].get_color())

    make_neff_yp("dispersion_1.csv", "red", window_size=200, disp_label="Dispersion 1")
    make_neff_yp("dispersion_2.csv", "blue", disp_label="Dispersion 2")
    make_neff_yp("dispersion_3.csv", "green", window_size=200, disp_label="Dispersion 3")
    plt.xlabel("$Y_p$")
    plt.ylabel("$\Delta N_{\\mathrm{eff}}$")
    plt.title("$\Delta N_{\\mathrm{eff}}$ vs $Y_p$")
    plt.hlines(0, 0.245, 0.248, colors="black", linestyles="dashed",label="Standard BBN")

    plt.tight_layout()
    plt.legend()

    plt.savefig("neff_yp.pdf") 

def plot_neff_doh():

    def make_neff_yp(disp, color, window_size=300, disp_label=None):
        data = pd.read_csv(disp, sep=" ")
        # print(data.dtypes)
        STD_NEFF = 3.046
        yps = data["DoH"]
        neffs = data["Neff"]
        # print(data)
        yps = np.array([float(x) for x in yps])
        neffs = np.array([float(x) for x in neffs])
        sorted_indices = np.argsort(yps)
        yps = yps[sorted_indices]
        neffs = neffs[sorted_indices]
        
        means = []
        x_means = []
        stds = []
        for i in range(0, len(yps), window_size):
            means.append(np.mean(neffs[i:i+window_size]))
            stds.append(np.std(neffs[i:i+window_size]))
            x_means.append(np.mean(yps[i:i+window_size]))

        mu_of_yp = interp1d(x_means, means, kind="cubic", bounds_error=False, fill_value="extrapolate")
        std_of_yp = interp1d(x_means, stds, kind="cubic", bounds_error=False, fill_value="extrapolate")
        xs = np.linspace(2.518, 2.576, 1000) 
        ys = mu_of_yp(xs) 
        std_ys = std_of_yp(xs)

        line = plt.plot(xs, ys - STD_NEFF, color=color, label=disp_label)
        plt.fill_between(xs, ys - std_ys - STD_NEFF, ys + std_ys - STD_NEFF, alpha=0.3, color=line[0].get_color())

    make_neff_yp("dispersion_1.csv", "red", window_size=100, disp_label="Dispersion 1")
    make_neff_yp("dispersion_2.csv", "blue", disp_label="Dispersion 2")
    make_neff_yp("dispersion_3.csv", "green", window_size=100, disp_label="Dispersion 3")
    plt.xlabel("$D/H$")
    plt.ylabel("$\Delta N_{\\mathrm{eff}}$")
    plt.title("$\Delta N_{\\mathrm{eff}}$ vs $D/H$")
    # plt.hlines(0, 0.245, 0.248, colors="black", linestyles="dashed",label="Standard BBN")

    plt.tight_layout()
    plt.legend()

    plt.savefig("neff_doh.png") 

plot_neff_yp()
# plot_neff_doh()