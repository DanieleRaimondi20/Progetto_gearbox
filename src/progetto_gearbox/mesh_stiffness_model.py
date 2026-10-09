"""Rigidezza flessionale + a taglio di un dente danneggiato.

Paper: Liang, Zuo, Pandey (2014), par. 5.1, condition 1-4.
Le formule sono le stesse di rigidezza_dente_danneggiato.py, ma riorganizzate
in una funzione senza grafici, cosi' si puo' chiamare variando tooth_damage e
alpha1 (angolo della forza) nei test.

Convenzione (come nel paper): alpha1 > 0, il limite inferiore degli integrali
e' -alpha1, alpha_a e alpha_c restituiti da _find_alpha_bisection sono > 0.
"""
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from progetto_gearbox.gearbox import _find_alpha_bisection


def default_find_alpha(gear):
    """Restituisce find(h_target) -> alpha_k, usando _find_alpha_bisection."""

    def find(h_target):
        # _find_alpha_bisection trova il raggio rho con h(rho) = h_target, ma il
        # suo primo valore di ritorno (alpha2 - psi) e' l'angolo polare del
        # punto, NON l'alpha del paper. Nel paper (eq. 2.6-2.8, 5.x)
        #     h(alpha) = Rb[(alpha2-alpha)cos(alpha) + sin(alpha)]
        # coincide con la semi-spessore reale solo per alpha = alpha2 - phi(rho)
        # (verificato numericamente: h e x coincidono a 1e-16).
        # alpha_a/alpha_c del paper sono positivi: alpha_k = -(alpha2 - phi).
        _, _, phi, _ = _find_alpha_bisection(
            min_radius=gear.radiuses["base"],
            max_radius=gear.radiuses["addendum"],
            base_radius=gear.radiuses["base"],
            alpha2=gear.alpha[2],
            h_target=h_target,
        )
        return phi - gear.alpha[2]

    return find


def alpha_r(gear, q, v, kind):
    """alpha_r: angolo polare del punto del cerchio di radice sulla verticale
    di A (kind="q1", condizioni 1-2) o di C (kind="q2", condizioni 3-4), cioe'
    il punto x = 0 di Fig. 10:  Rr cos(alpha_r) = x_A (o x_C).
      q1: cos(ar) = cos(a3) - (q1/Rr) cos(v)
      q2: cos(ar) = cos(a3) - (sin(a3)/sin(v) - q2/Rr) cos(v)
    """
    Rr, a3 = gear.radiuses["root"], gear.alpha[3]
    if kind == "q1":
        c = np.cos(a3) - (q / Rr) * np.cos(v)
    else:
        c = np.cos(a3) - (np.sin(a3) / np.sin(v) - q / Rr) * np.cos(v)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def alpha_of_x(gear, x, alphar, tol=1e-12):
    """Angolo del profilo in funzione di x (x = 0 sulla verticale di A/C).

    Ritorna (zona, angolo):
      "root"  0 <= x <= d1   : angolo polare gamma sul cerchio di radice
      "line"  d1 < x <= d2   : tratto NN' (hx = Rb sin a2), angolo = alpha3
      "evol"  x > d2         : alpha del paper = alpha2 - phi(rho), con rho
                               tale che rho*cos(alpha2-psi(rho)) = Rr cos(alphar) + x
    Il punto di tangenza con l'evolvente e' a x = Rb cos(a2) - Rr cos(ar) = d2.
    """
    Rb, Rr, Ra = gear.radiuses["base"], gear.radiuses["root"], gear.radiuses["addendum"]
    a2, a3 = gear.alpha[2], gear.alpha[3]
    X = Rr * np.cos(alphar) + x            # ascissa da O
    d1 = Rr * (np.cos(a3) - np.cos(alphar))
    d2 = Rb * np.cos(a2) - Rr * np.cos(alphar)
    if x <= d1:
        return "root", float(np.arccos(X / Rr))
    if x <= d2:
        return "line", a3
    f = lambda rho: rho * np.cos(a2 - gear._compute_psi_angle(2 * rho)) - X
    lo, hi = Rb, Ra
    if f(hi) < 0:
        raise ValueError("x oltre la punta del dente")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    rho = 0.5 * (lo + hi)
    return "evol", float(a2 - gear._compute_phi_angle(2 * rho))


def geometry_thresholds(gear):
    """hr, alpha0 e h0 calcolati come nello script originale."""
    hr = 2 * gear.radiuses["root"] * np.sin(gear.alpha[3])
    alpha0 = gear.alpha[2] - gear._compute_phi_angle(gear.diameters["addendum"])
    h0 = gear.radiuses["base"] * (
        np.sin(alpha0) + (gear.alpha[2] - alpha0) * np.cos(alpha0)
    )
    return hr, alpha0, h0


def _quad(f, lo, hi):
    # epsabs=0 e' essenziale: gli integrali valgono ~1e-9..1e-10 (1/(E*L) ~ 1e-10),
    # molto meno dell'epsabs di default di quad (1.49e-8), che farebbe fermare
    # l'algoritmo alla prima stima senza controllare l'accuratezza.
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", IntegrationWarning)
        val, err = quad(f, lo, hi, epsabs=0.0, epsrel=1e-10, limit=500)
    return val, err


def compute_Kt(gear, tooth_damage, alphar=None, alpha1=None, v=np.pi / 6, find_alpha=None):
    """Kt = 1/(Ib+Is) per un dato tooth_damage e una data posizione alpha1.

    Restituisce un dizionario con Kt, Ib, Is, case ("1","2a","2b","3a","3b","4"),
    alpha_thr (alpha_a o alpha_c, se esiste) e parts (ogni integrale con
    valore, errore stimato ed estremi).
    """
    cos, sin = np.cos, np.sin
    Rb, Rr = gear.radiuses["base"], gear.radiuses["root"]
    a2, a3 = gear.alpha[2], gear.alpha[3]
    E, L = gear.young, gear.thickness
    N, pa, nu = gear.teeth_number, gear.pressure_angle, gear.poisson

    hr, alpha0, h0 = geometry_thresholds(gear)
    a1 = abs(alpha0) if alpha1 is None else alpha1
    find_alpha = find_alpha or default_find_alpha(gear)

    hq = tooth_damage * hr
    alpha_thr = None
    if tooth_damage <= 0.5:
        q = hq / sin(v)
        ha = hr / 2 - hq
        kind = "q1"
        if ha >= h0:
            alpha_thr = find_alpha(ha)  # alpha_a
            case = "1" if a1 > alpha_thr else "2a"
        else:
            case = "2b"
    else:
        hc = hq - hr / 2
        q = hc / sin(v)
        kind = "q2"
        if hc >= h0:
            alpha_thr = find_alpha(hc)  # alpha_c del paper (nel tuo script alphac = -alpha_k)
            case = "3a" if a1 <= alpha_thr else "4"
        else:
            case = "3b"

    if alphar is None:
        alphar = alpha_r(gear, q, v, kind)
    sv, cv = sin(v), cos(v)
    nx = N * cos(pa) / (N - 2.5)
    two = (N - 2.5) / (N * cos(pa))
    num_b1 = 4 * (1 - two * cos(a1) * cos(a3)) ** 3 - 4 * (1 - cos(a1) * cos(a2)) ** 3
    k_s = 2.4 * (1 + nu) * cos(a1) ** 2

    def mom(a):  # {1 + cos(a1)[(a2-a) sin a - cos a]}
        return 1 + cos(a1) * ((a2 - a) * sin(a) - cos(a))

    if kind == "q1":  # condition 1 e 2 (eq. 5.5, 5.6, 5.9, 5.10)
        def Ib0(a):
            return (12 * sin(a) * (nx - (cos(a) + cos(a3) - cos(alphar) - (q / Rr) * cv) * cos(a1)) ** 2
                    / (E * L * (sin(a3) + sin(a) - (q / Rr) * sv) ** 3))
        Ib1 = num_b1 / (E * L * cos(a1) * (2 * sin(a2) - (q / Rb) * sv) ** 3)

        def Ib2(a):
            return (12 * mom(a) ** 2 * (a2 - a) * cos(a)
                    / (E * L * (sin(a2) - (q / Rb) * sv + sin(a) + (a2 - a) * cos(a)) ** 3))

        def Is0(a):
            return k_s * sin(a) / (E * L * (sin(a3) - (q / Rr) * sv + sin(a)) ** 3)
        Is1 = k_s * (cos(a2) - two * cos(a3)) / (E * L * (2 * sin(a2) - (q / Rb) * sv))

        def Is2(a):
            return (k_s * (a2 - a) * cos(a)
                    / (E * L * (sin(a2) - (q / Rb) * sv + sin(a) + (a2 - a) * cos(a))))
    else:  # condition 3 e 4 (eq. 5.13-5.16)
        def Ib0(a):
            return (12 * sin(a) * (nx - (cos(a) + cos(a3) - cos(alphar)
                                         - (sin(a3) / sv - q / Rr) * cv) * cos(a1)) ** 2
                    / (E * L * (sin(a) - (q / Rr) * sv) ** 3))
        Ib1 = num_b1 / (E * L * cos(a1) * (sin(a2) - (q / Rb) * sv) ** 3)

        def Ib2(a):
            return (12 * mom(a) ** 2 * (a2 - a) * cos(a)
                    / (E * L * (sin(a) + (a2 - a) * cos(a) - (q / Rb) * sv) ** 3))

        def Is0(a):
            return k_s * sin(a) / (E * L * (sin(a) - (q / Rr) * sv) ** 3)
        Is1 = k_s * (cos(a2) - two * cos(a3)) / (E * L * (sin(a2) - (q / Rb) * sv))

        def Is2(a):
            return (k_s * (a2 - a) * cos(a)
                    / (E * L * (sin(a) + (a2 - a) * cos(a) - (q / Rb) * sv)))

    def Ib3(a):  # solo condition 1
        return 3 * mom(a) ** 2 * (a2 - a) * cos(a) / (2 * E * L * (sin(a) + (a2 - a) * cos(a)) ** 3)

    def Is3(a):  # solo condition 1
        return 1.2 * (1 + nu) * (a2 - a) * cos(a) * cos(a1) ** 2 / (E * L * (sin(a) + (a2 - a) * cos(a)))

    # estremi dell'integrale "2" (e "3" nella condition 1)
    if case == "1":
        lim2, lim3 = (-alpha_thr, a2), (-a1, -alpha_thr)
    elif case == "4":
        lim2, lim3 = (-alpha_thr, a2), None  # nel tuo script: alphac = -alpha_k
    else:
        lim2, lim3 = (-a1, a2), None

    parts = {}

    def integ(name, f, lim):
        val, err = _quad(f, *lim)
        parts[name] = dict(value=val, err=err, lo=lim[0], hi=lim[1])
        return val

    Ib = integ("Ib0", Ib0, (a3, alphar)) + Ib1 + integ("Ib2", Ib2, lim2)
    Is = integ("Is0", Is0, (a3, alphar)) + Is1 + integ("Is2", Is2, lim2)
    parts["Ib1"], parts["Is1"] = dict(value=Ib1), dict(value=Is1)
    if lim3 is not None:
        Ib += integ("Ib3", Ib3, lim3)
        Is += integ("Is3", Is3, lim3)

    return dict(Kt=1.0 / (Ib + Is), Ib=Ib, Is=Is, case=case,
                alpha_thr=alpha_thr, parts=parts)
