"""
MCMC driver for BBN abundances with PRyM
"""

import traceback
import warnings, sys, numpy as np, multiprocessing as mp
from scipy.special import zeta
import PRyM.PRyM_main as PRyMmain
import PRyM.PRyM_init as PRyMini

warnings.filterwarnings("ignore")


class DispersionModel:
    DISP_LAM = 1
    DISP_BET = 2
    DISP_ALP = 3

    FREEZEOUT_T = 0.5
    FREEZEOUT_RHO = 0.00135

    def __init__(self):
        PRyMini.NP_e_flag = PRyMini.numba_flag = True
        PRyMini.nacreii_flag = PRyMini.aTid_flag = PRyMini.smallnet_flag = True
        PRyMini.compute_nTOp_flag = False

    @staticmethod
    def rho_np(Tg, disp, param):
        if disp == DispersionModel.DISP_LAM:
            return -(96 / np.pi**2) * zeta(5) * param * Tg**5
        elif disp == DispersionModel.DISP_BET:
            return (600 / np.pi**2) * zeta(6) * param * Tg**6
        elif disp == DispersionModel.DISP_ALP:
            return +(120 / np.pi**2) * zeta(5) * param * Tg**5
        return 0

    @staticmethod
    def p_np(Tg, disp, param):
        if disp == DispersionModel.DISP_LAM:
            return -(72 / (3 * np.pi**2)) * zeta(5) * param * Tg**5
        elif disp == DispersionModel.DISP_BET:
            return (360 / (3 * np.pi**2)) * zeta(6) * param * Tg**6
        elif disp == DispersionModel.DISP_ALP:
            return +(72 / (3 * np.pi**2)) * zeta(5) * param * Tg**5
        return 0

    @staticmethod
    def drho_np_dT(Tg, disp, param):
        if disp == DispersionModel.DISP_LAM:
            return -(96 * 5 / np.pi**2) * zeta(5) * param * Tg**4
        elif disp == DispersionModel.DISP_BET:
            return (600 * 6 / np.pi**2) * zeta(6) * param * Tg**5
        elif disp == DispersionModel.DISP_ALP:
            return -(600 * 5 / np.pi**2) * zeta(5) * param * Tg**4
        return 0

    def freezeout_conditions(self, disp, param):
        Tf = self.FREEZEOUT_T
        rhoF = self.rho_np(Tf, disp, param)

        wF = self.p_np(Tf, disp, param) / rhoF
        if not (-1.0 < wF < 1.0):
            return False
        if rhoF >= self.FREEZEOUT_RHO:
            return False

        return True

    def abundances(self, disp, param):
        rho_w = lambda Tg: self.rho_np(Tg, disp, param)
        p_w = lambda Tg: self.p_np(Tg, disp, param)
        drho_dt_w = lambda Tg: self.drho_np_dT(Tg, disp, param)

        PRyMini.tau_n = np.random.normal(PRyMini.tau_n, 0.5)
        PRyMini.Omegabh2 = np.random.normal(PRyMini.Omegabh2, 2e-4)
        PRyMini.eta0b = PRyMini.Omegabh2_to_eta0b * PRyMini.Omegabh2
        (
            PRyMini.p_npdg,
            PRyMini.p_dpHe3g,
            PRyMini.p_ddHe3n,
            PRyMini.p_ddtp,
            PRyMini.p_tpag,
            PRyMini.p_tdan,
            PRyMini.p_taLi7g,
            PRyMini.p_He3ntp,
            PRyMini.p_He3dap,
            PRyMini.p_He3aBe7g,
            PRyMini.p_Be7nLi7p,
            PRyMini.p_Li7paa,
        ) = np.random.normal(0, 1, 12)
        PRyMini.ReloadKeyRates()

        try:
            PRyMini.numba_flag = True
            PRyMini.compute_nTOp_flag = False
            PRyMini.compute_dTp_flag = False

            solver = PRyMmain.PRyMclass(rho_w, p_w, drho_dt_w)

            prym_results = solver.PRyMresults()
            return prym_results
        except Exception as err:
            traceback.print_exc()

            return np.array([np.nan] * 8)

    def prym_results(self, disp, param):
        """Return only the abundance results (Yp, D/H, ³He/H, ⁷Li/H) for backward compatibility"""
        full_results = self.abundances(disp, param)

        return full_results


OBS = np.array([0.245, 2.547])
SIG2 = np.square([0.003, 0.029])


outfiles = {}

PRIORS = {1: (1e-10, 5e-3), 2: (1e-9, 1e-3), 3: (1e-10, 2e-4)}


def ln_prior(model, disp, param):
    lo, hi = PRIORS[disp]

    if not (lo < param < hi):
        return -np.inf
    if not model.freezeout_conditions(disp, param):
        return -np.inf
    return 0.0


def ln_likelihood(model, disp, param):
    try:
        np.random.seed()
        Neff, Omegabh2, Omegabh2m1, Yp_cmb, Yp, DoH, He3, Li7 = model.prym_results(
            disp, param
        )

        if np.any(np.isnan([Neff, Omegabh2, Omegabh2m1, Yp_cmb, Yp, DoH, He3, Li7])):
            return -np.inf
        theo = np.array([Yp_cmb, DoH])
        chi2 = np.sum((theo - OBS) ** 2 / SIG2)

        if disp not in outfiles:
            outfiles[disp] = open(f"dispersion_{disp}.csv", "w")

            outfiles[disp].write(
                "Neff Omegabh2 1/Omegabh2 Yp(CMB) Yp DoH He3 Li7 param log_likelihood\n"
            )
            outfiles[disp].flush()

        outfiles[disp].write(
            f"{Neff:.15e} {Omegabh2:.15e} {1 / Omegabh2:.15e} {Yp_cmb:.15e} {Yp:.15e} {DoH:.15e} {He3:.15e} {Li7:.15e} {param:.15e} {-0.5 * chi2:.15e}\n"
        )
        outfiles[disp].flush()

        return -0.5 * chi2
    except:
        return -np.inf


def ln_posterior(theta, model, disp):
    if type(theta) == list or type(theta) == np.ndarray:
        param = theta[0]
    else:
        param = theta
    lp = ln_prior(model, disp, param)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(model, disp, param)


def close_output_files():
    """Close all output file handles."""
    for disp, file_handle in outfiles.items():
        file_handle.close()
