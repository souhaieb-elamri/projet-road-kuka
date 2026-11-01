r"""Bloc FMDS - DONNÉES SIMULÉES (pas issues de RoAD).

Étude de cas "banc d'essai virtuel" du guide pédagogique :
200 temps de défaillance Weibull (beta=2.5, eta=1500 h), censure à droite à 1200 h.

Étapes :
- non paramétrique : Kaplan-Meier, Nelson-Aalen, estimateur à noyau (hasard lissé) ;
- paramétrique : MLE avec censure, moindres carrés (LS), méthode des moments (MoM) ;
- comparaison de 4 lois (Weibull, exponentielle, log-normale, gamma) par AIC ;
- intervalles de confiance (bootstrap), MTTF, MTTR, MTBF, disponibilité, B10.

Lancer :  python run_fmds_simule.py      Sorties : dossier fmds_results/
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import gamma, log
from scipy import stats
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import minimize, brentq

SEED = 42
N = 200
BETA_TRUE, ETA_TRUE = 2.5, 1500.0     # forme, échelle (heures)
T_CENSOR = 1200.0                      # arrêt de l'essai (censure à droite)
REPAIR_MEDIAN_H, REPAIR_SIGMA = 6.0, 0.4   # temps de réparation SIMULÉS (log-normale)
N_BOOT = 500
OUT = Path("fmds_results")


# ------------------------------------------------------------- simulation
def simulate(rng):
    t_true = ETA_TRUE * rng.weibull(BETA_TRUE, N)
    t_obs = np.minimum(t_true, T_CENSOR)
    event = (t_true <= T_CENSOR).astype(int)       # 1 = défaillance, 0 = censure
    return t_obs, event


# ------------------------------------------------- estimateurs non paramétriques
def kaplan_meier(t, e):
    times = np.unique(t[e == 1])
    s, surv = 1.0, []
    for u in times:
        at_risk = np.sum(t >= u)
        d = np.sum((t == u) & (e == 1))
        s *= 1 - d / at_risk
        surv.append(s)
    return times, np.array(surv)


def nelson_aalen(t, e):
    times = np.unique(t[e == 1])
    h, cum = 0.0, []
    for u in times:
        h += np.sum((t == u) & (e == 1)) / np.sum(t >= u)
        cum.append(h)
    return times, np.array(cum)


def km_on_grid(tk, sk, grid):
    """Valeurs de la fonction en escalier Kaplan-Meier sur une grille."""
    idx = np.searchsorted(tk, grid, side="right")
    return np.r_[1.0, sk][idx]


def kernel_hazard(t, e, grid, bw=None):
    """Estimateur à noyau gaussien du hasard (lissage des sauts de Nelson-Aalen).

    Tient compte de la censure (contrairement à un simple ksdensity sur les
    défaillances). R(t) = exp(-H(t)) avec H l'intégrale du hasard lissé.
    Biais de bord possible près de 0 et près de la censure (1200 h).
    """
    u = np.unique(t[e == 1])
    inc = np.array([np.sum((t == x) & (e == 1)) / np.sum(t >= x) for x in u])
    if bw is None:
        bw = 1.06 * np.std(t[e == 1], ddof=1) * len(u) ** (-1 / 5)   # règle de Silverman
    z = (grid[:, None] - u[None, :]) / bw
    h = (inc[None, :] * np.exp(-0.5 * z ** 2) / (bw * np.sqrt(2 * np.pi))).sum(axis=1)
    H = cumulative_trapezoid(h, grid, initial=0.0)
    return h, H, np.exp(-H), bw


# ------------------------------------------------------ estimateurs paramétriques
def weibull_negloglik(params, t, e):
    b, eta = np.exp(params)
    z = (t / eta) ** b
    ll_fail = np.log(b / eta) + (b - 1) * np.log(t / eta) - z
    return -(np.sum(ll_fail[e == 1]) + np.sum(-z[e == 0]))


def fit_weibull_ls(t, e):
    """Moindres carrés sur papier de Weibull, rangs ajustés (Johnson) + Bernard."""
    order = np.argsort(t)
    t, e = t[order], e[order]
    n = len(t)
    prev, ranks, tf = 0.0, [], []
    for i in range(n):
        if e[i] == 1:
            prev = prev + (n + 1 - prev) / (1 + (n - i))
            ranks.append(prev)
            tf.append(t[i])
    f = (np.array(ranks) - 0.3) / (n + 0.4)
    x, y = np.log(tf), np.log(-np.log(1 - f))
    slope, intercept = np.polyfit(x, y, 1)
    return slope, float(np.exp(-intercept / slope))


def fit_weibull_mom(t):
    """Méthode des moments sur les temps observés.

    IGNORE la censure : les unités censurées à 1200 h sont traitées comme des
    défaillances, ce qui biaise l'estimation (à montrer dans le rapport).
    """
    m, s = np.mean(t), np.std(t, ddof=1)
    cv2 = (s / m) ** 2
    f = lambda b: gamma(1 + 2 / b) / gamma(1 + 1 / b) ** 2 - 1 - cv2
    b = brentq(f, 0.2, 100.0)
    return float(b), float(m / gamma(1 + 1 / b))


def fit_weibull_mle(t, e, start=None):
    b0, e0 = start if start else (1.5, np.mean(t))
    res = minimize(weibull_negloglik, np.log([b0, e0]), args=(t, e), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    b, eta = np.exp(res.x)
    return float(b), float(eta), float(res.fun)


def fit_exponential_mle(t, e):
    rate = e.sum() / t.sum()
    ll = e.sum() * np.log(rate) - rate * t.sum()
    return float(rate), float(-ll)


def _nll(dist, t, e):
    return -(np.sum(dist.logpdf(t[e == 1])) + np.sum(dist.logsf(t[e == 0])))


def fit_lognormal_mle(t, e):
    def f(p):
        return _nll(stats.lognorm(s=np.exp(p[1]), scale=np.exp(p[0])), t, e)
    res = minimize(f, [np.mean(np.log(t)), np.log(np.std(np.log(t)))], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    mu, sig = res.x[0], float(np.exp(res.x[1]))
    return float(mu), sig, float(res.fun)


def fit_gamma_mle(t, e):
    def f(p):
        return _nll(stats.gamma(a=np.exp(p[0]), scale=np.exp(p[1])), t, e)
    res = minimize(f, np.log([2.0, np.mean(t) / 2]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    k, theta = np.exp(res.x)
    return float(k), float(theta), float(res.fun)


def mttf_weibull(b, eta):
    return eta * gamma(1 + 1 / b)


def b_life(b, eta, p=0.10):
    """Temps pour lequel une proportion p de la population est défaillante (B10 si p=0.1)."""
    return eta * (-log(1 - p)) ** (1 / b)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    t, e = simulate(rng)
    print("DONNÉES SIMULÉES (pas RoAD)")
    print(f"n={N}, défaillances={e.sum()}, censures={N - e.sum()} (censure à {T_CENSOR:.0f} h)")

    # non paramétrique
    tk, sk = kaplan_meier(t, e)
    tn, hn = nelson_aalen(t, e)
    grid = np.linspace(0, T_CENSOR, 600)
    h_ker, H_ker, R_ker, bw = kernel_hazard(t, e, grid)

    # paramétrique
    b_ls, eta_ls = fit_weibull_ls(t, e)
    b_mom, eta_mom = fit_weibull_mom(t)
    b_ml, eta_ml, nll_w = fit_weibull_mle(t, e, start=(b_ls, eta_ls))
    rate, nll_e = fit_exponential_mle(t, e)
    mu_ln, sig_ln, nll_ln = fit_lognormal_mle(t, e)
    k_g, th_g, nll_g = fit_gamma_mle(t, e)
    aic = {"Weibull": 2 * 2 + 2 * nll_w, "Exponentielle": 2 * 1 + 2 * nll_e,
           "Log-normale": 2 * 2 + 2 * nll_ln, "Gamma": 2 * 2 + 2 * nll_g}
    mttf_law = {"Weibull": mttf_weibull(b_ml, eta_ml), "Exponentielle": 1 / rate,
                "Log-normale": float(np.exp(mu_ln + sig_ln ** 2 / 2)), "Gamma": k_g * th_g}
    params_law = {"Weibull": f"beta={b_ml:.3f}; eta={eta_ml:.1f}", "Exponentielle": f"lambda={rate:.6f}",
                  "Log-normale": f"mu={mu_ln:.3f}; sigma={sig_ln:.3f}", "Gamma": f"k={k_g:.3f}; theta={th_g:.1f}"}
    nll_law = {"Weibull": nll_w, "Exponentielle": nll_e, "Log-normale": nll_ln, "Gamma": nll_g}
    best_law = min(aic, key=aic.get)

    # temps moyen restreint à la durée de l'essai (comparable entre estimateurs)
    rmst_km = float(trapezoid(km_on_grid(tk, sk, grid), grid))
    rmst_ker = float(trapezoid(R_ker, grid))
    rmst_w = float(trapezoid(np.exp(-(grid / eta_ml) ** b_ml), grid))

    # bootstrap (intervalles de confiance à 95 %)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, N, N)
        try:
            bb, ee, _ = fit_weibull_mle(t[idx], e[idx], start=(b_ml, eta_ml))
            boots.append((bb, ee, mttf_weibull(bb, ee)))
        except Exception:
            continue
    boots = np.array(boots)
    ci = np.percentile(boots, [2.5, 97.5], axis=0)

    # maintenabilité (temps de réparation SIMULÉS) et disponibilité
    repairs = rng.lognormal(mean=np.log(REPAIR_MEDIAN_H), sigma=REPAIR_SIGMA, size=int(e.sum()))
    mttr = float(repairs.mean())
    mu = 1 / mttr
    mttf = mttf_weibull(b_ml, eta_ml)
    mtbf = mttf + mttr                       # cycle complet : fonctionnement + réparation
    availability = mttf / (mttf + mttr)
    b10 = b_life(b_ml, eta_ml, 0.10)
    r_at = {x: float(np.exp(-(x / eta_ml) ** b_ml)) for x in (500, 1000, 1200)}
    m_at = {x: float(np.mean(repairs <= x)) for x in (4, 8, 12)}

    # résultats
    rows = [
        {"estimateur": "Vraie loi (simulation)", "beta": BETA_TRUE, "eta_h": ETA_TRUE,
         "MTTF_h": mttf_weibull(BETA_TRUE, ETA_TRUE)},
        {"estimateur": "MLE avec censure", "beta": b_ml, "eta_h": eta_ml, "MTTF_h": mttf},
        {"estimateur": "Moindres carres (LS)", "beta": b_ls, "eta_h": eta_ls,
         "MTTF_h": mttf_weibull(b_ls, eta_ls)},
        {"estimateur": "Moments (MoM, ignore la censure)", "beta": b_mom, "eta_h": eta_mom,
         "MTTF_h": mttf_weibull(b_mom, eta_mom)},
    ]
    with open(OUT / "estimateurs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(OUT / "lois_aic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["loi", "parametres", "neg_loglik", "AIC", "MTTF_h"])
        for k in sorted(aic, key=aic.get):
            w.writerow([k, params_law[k], f"{nll_law[k]:.2f}", f"{aic[k]:.1f}", f"{mttf_law[k]:.1f}"])
    indic = {"MTTF_h": mttf, "MTTR_h": mttr, "MTBF_h": mtbf, "mu_reparation_par_h": mu,
             "lambda_exponentielle_par_h": rate, "Disponibilite": availability,
             "B10_h": b10, "R(500h)": r_at[500], "R(1000h)": r_at[1000], "R(1200h)": r_at[1200],
             "M(4h)": m_at[4], "M(8h)": m_at[8], "M(12h)": m_at[12],
             "AIC_Weibull": aic["Weibull"], "AIC_Exponentielle": aic["Exponentielle"],
             "AIC_Lognormale": aic["Log-normale"], "AIC_Gamma": aic["Gamma"],
             "temps_moyen_restreint_1200h_KM": rmst_km, "temps_moyen_restreint_1200h_noyau": rmst_ker,
             "temps_moyen_restreint_1200h_Weibull": rmst_w, "largeur_noyau_h": bw,
             "beta_IC95_bas": ci[0, 0], "beta_IC95_haut": ci[1, 0],
             "eta_IC95_bas": ci[0, 1], "eta_IC95_haut": ci[1, 1],
             "MTTF_IC95_bas": ci[0, 2], "MTTF_IC95_haut": ci[1, 2]}
    with open(OUT / "indicateurs_fmds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["indicateur", "valeur"])
        for k, v in indic.items():
            w.writerow([k, f"{v:.6g}"])

    print("\n=== Estimation de la loi de Weibull ===")
    for r in rows:
        print(f"{r['estimateur']:34s} beta={r['beta']:.3f}  eta={r['eta_h']:.1f} h  MTTF={r['MTTF_h']:.1f} h")
    print(f"IC95% bootstrap : beta [{ci[0,0]:.2f}; {ci[1,0]:.2f}]  eta [{ci[0,1]:.0f}; {ci[1,1]:.0f}]  "
          f"MTTF [{ci[0,2]:.0f}; {ci[1,2]:.0f}]")
    print("\n=== Comparaison des lois (MLE avec censure, critère AIC) ===")
    for k in sorted(aic, key=aic.get):
        print(f"{k:14s} AIC={aic[k]:.1f}  MTTF={mttf_law[k]:.1f} h  ({params_law[k]})")
    print(f"-> loi préférée : {best_law}")
    print(f"\nbeta estimé = {b_ml:.2f} {'> 1 : taux de défaillance croissant (usure)' if b_ml > 1 else '<= 1'}")
    print(f"\n=== Estimateur à noyau (largeur {bw:.0f} h) ===")
    print(f"Temps moyen restreint à {T_CENSOR:.0f} h : KM={rmst_km:.1f}  noyau={rmst_ker:.1f}  Weibull MLE={rmst_w:.1f}")
    print("\n=== Indicateurs FMDS (MTTR simulé) ===")
    print(f"MTTF = {mttf:.1f} h | MTTR = {mttr:.2f} h | MTBF = MTTF + MTTR = {mtbf:.1f} h")
    print(f"Taux de réparation mu = {mu:.3f} /h | Disponibilité = {availability*100:.2f} %")
    print(f"R(500)={r_at[500]:.3f}  R(1000)={r_at[1000]:.3f}  R(1200)={r_at[1200]:.3f}")
    print(f"M(4h)={m_at[4]:.2f}  M(8h)={m_at[8]:.2f}  M(12h)={m_at[12]:.2f}")
    print(f"B10 = {b10:.0f} h -> maintenance préventive proposée avant ~{b10:.0f} h")

    # figure 1 : courbes principales
        # figure 0 : histogramme des temps de défaillance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mask_fail = (e == 1)
    mask_cens = (e == 0)
    bins = np.linspace(0, 2500, 40)
    ax.hist(t[mask_fail], bins=bins, color="#2E75B6", edgecolor="black", alpha=0.85, label="Défaillance observée")
    ax.hist(t[mask_cens], bins=bins, color="#B0B0B0", edgecolor="black", alpha=0.85, label="Censuré à 1200 h")
    ax.axvline(T_CENSOR, color="red", ls="--", lw=1.5, label=f"Censure = {T_CENSOR:.0f} h")
    ax.axvline(mttf_weibull(BETA_TRUE, ETA_TRUE), color="green", ls=":", lw=1.5, label=f"MTTF vrai = {mttf_weibull(BETA_TRUE, ETA_TRUE):.0f} h")
    ax.set_xlabel("Temps (heures)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Histogramme des temps de défaillance simulés (Weibull β=2.5, η=1500 h)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "donnees_fmds.png", dpi=150)
    plt.close(fig)
    tt = np.linspace(1, T_CENSOR * 1.2, 400)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].step(np.r_[0, tk], np.r_[1, sk], where="post", label="Kaplan-Meier")
    axs[0].plot(grid, R_ker, "-.", label="noyau")
    axs[0].plot(tt, np.exp(-(tt / eta_ml) ** b_ml), "--", label="Weibull MLE")
    axs[0].plot(tt, np.exp(-(tt / ETA_TRUE) ** BETA_TRUE), ":", label="vraie loi")
    axs[0].axvline(T_CENSOR, color="gray", lw=0.8); axs[0].set_title("Fiabilité R(t)")
    axs[0].set_xlabel("temps (h)"); axs[0].legend()
    axs[1].step(np.r_[0, tn], np.r_[0, hn], where="post", label="Nelson-Aalen")
    axs[1].plot(grid, H_ker, "-.", label="noyau")
    axs[1].plot(tt, (tt / eta_ml) ** b_ml, "--", label="H(t) Weibull MLE")
    axs[1].set_title("Hasard cumulé H(t)"); axs[1].set_xlabel("temps (h)"); axs[1].legend()
    axs[2].plot(tt, (b_ml / eta_ml) * (tt / eta_ml) ** (b_ml - 1), color="g", label="Weibull MLE")
    axs[2].plot(grid, h_ker, "-.", color="m", label="noyau")
    axs[2].plot(tt, (BETA_TRUE / ETA_TRUE) * (tt / ETA_TRUE) ** (BETA_TRUE - 1), ":", color="k", label="vraie loi")
    axs[2].set_title("Taux de défaillance λ(t) (croissant si β>1)"); axs[2].set_xlabel("temps (h)"); axs[2].legend()
    fig.suptitle("FMDS - données SIMULÉES (pas RoAD)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fmds_courbes.png", dpi=150); plt.close(fig)

    # figure 2 : comparaison des estimateurs paramétriques
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(np.r_[0, tk], np.r_[1, sk], where="post", color="k", lw=1.5, label="Kaplan-Meier")
    ax.plot(tt, np.exp(-(tt / ETA_TRUE) ** BETA_TRUE), ":", color="gray", lw=2, label="vraie loi")
    ax.plot(tt, np.exp(-(tt / eta_ml) ** b_ml), "--", label=f"MLE (β={b_ml:.2f})")
    ax.plot(tt, np.exp(-(tt / eta_ls) ** b_ls), "--", label=f"LS (β={b_ls:.2f})")
    ax.plot(tt, np.exp(-(tt / eta_mom) ** b_mom), "--", label=f"MoM (β={b_mom:.2f})")
    ax.axvline(T_CENSOR, color="gray", lw=0.8)
    ax.set_xlabel("temps (h)"); ax.set_ylabel("R(t)"); ax.set_title("Estimateurs de la loi de Weibull (SIMULÉ)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fmds_estimateurs.png", dpi=150); plt.close(fig)
    print("\nRésultats dans :", OUT.resolve())


if __name__ == "__main__":
    main()