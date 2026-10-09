"""Test di continuita' e di correttezza dei casi per la rigidezza del dente danneggiato.

Uso:   python test_mesh_stiffness.py        (stampa un riepilogo e salva i grafici
                                             in test_outputs/)
       pytest test_mesh_stiffness.py -v      (solo gli assert)

ALPHAR=None: alpha_r e' calcolato dalla geometria (alpha_r() nel modello).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from progetto_gearbox.mesh_stiffness_model import compute_Kt, default_find_alpha, geometry_thresholds

ALPHAR = None          # None = calcolato da alpha_r()
V = np.pi / 6          # angolo della cricca (come nello script)

TOL_CONT = 1e-3        # salto ammesso a cavallo di una transizione, rispetto a K_ref
EPS_TD = 1e-7          # distanza dalla soglia in tooth_damage
EPS_A1 = 1e-7          # distanza dalla soglia in alpha1 [rad]
JUMP_MAX = 0.05        # salto relativo massimo tra due punti vicini (controllo di regolarita')
A1_FRACS = (1.0, 0.5, 0.1)   # alpha1 come frazione di |alpha0| (punta del dente)
OUT = Path("test_outputs")
REPORT = []

_GEAR = None


def get_gear():
    global _GEAR
    if _GEAR is None:
        from progetto_gearbox.demos.gearbox_demo import create_gear
        _GEAR = create_gear()
    return _GEAR


def K(td, a1):
    return compute_Kt(get_gear(), td, alphar=ALPHAR, alpha1=a1, v=V)


def thresholds():
    hr, alpha0, h0 = geometry_thresholds(get_gear())
    return dict(hr=hr, h0=h0, a_tip=abs(alpha0),
                td1=0.5 - h0 / hr, td2=0.5, td3=0.5 + h0 / hr)


def family(case):
    return {"1": 0, "2a": 0, "2b": 1, "3b": 2, "3a": 3, "4": 3}[case]


# ---------------------------------------------------------------- correttezza
def test_alpha_k_matches_h0():
    """alpha_k(h0) deve coincidere con |alpha0| e avere il segno del paper."""
    g, t = get_gear(), thresholds()
    a = default_find_alpha(g)(t["h0"])
    assert a > 0, f"alpha_k={a} non positivo: convenzione di segno diversa dal paper"
    assert np.isclose(a, t["a_tip"], rtol=1e-4), f"alpha_k(h0)={a} ma |alpha0|={t['a_tip']}"
    REPORT.append(f"alpha_k(h0) = {a:.6f}   |alpha0| = {t['a_tip']:.6f}")


def test_units():
    """Le formule sono in unita' SI: lunghezze in m, E in Pa, K in N/m.
    I test di continuita' sono invarianti per scala, quindi non vedrebbero un errore di unita'."""
    g = get_gear()
    assert 1e-3 < g.radiuses["base"] < 1.0, f"raggio base = {g.radiuses['base']}: non sembra in metri"
    assert g.young > 1e9, (f"young = {g.young:.3e}: per acciaio in Pa serve ~2e11 "
                           "(sembra MPa o un'altra unita': Kt verrebbe sbagliata di ~1e6)")


def test_thresholds_are_ordered():
    t = thresholds()
    assert 0 < t["td1"] < 0.5 < t["td3"] < 1, f"soglie fuori ordine: {t}"


def test_case_labels():
    """Per ogni zona deve attivarsi il caso del paper (1, 2a, 2b, 3b, 3a, 4)."""
    g, t = get_gear(), thresholds()
    find = default_find_alpha(g)
    td_A = 0.5 * t["td1"]
    td_B = 0.5 - 0.5 * t["h0"] / t["hr"]
    td_C = 0.5 + 0.5 * t["h0"] / t["hr"]
    td_D = t["td3"] + 0.5 * (1 - t["td3"])
    aa = find(t["hr"] / 2 - td_A * t["hr"])
    ac = find(td_D * t["hr"] - t["hr"] / 2)
    expected = [
        (td_A, t["a_tip"], "1"), (td_A, 0.5 * aa, "2a"),
        (td_B, t["a_tip"], "2b"), (td_B, 0.3 * t["a_tip"], "2b"),
        (td_C, t["a_tip"], "3b"), (td_C, 0.3 * t["a_tip"], "3b"),
        (td_D, 0.5 * ac, "3a"), (td_D, t["a_tip"], "4"),
    ]
    for td, a1, want in expected:
        got = K(td, a1)["case"]
        assert got == want, f"td={td:.4f}, alpha1={a1:.5f}: atteso caso {want}, ottenuto {got}"


def test_case_sequence_vs_tooth_damage():
    """Al crescere di tooth_damage la famiglia di casi non puo' tornare indietro."""
    t = thresholds()
    for frac in A1_FRACS:
        a1 = frac * t["a_tip"]
        fams = [family(K(td, a1)["case"]) for td in np.linspace(0.002, 0.998, 300)]
        assert all(b >= a for a, b in zip(fams, fams[1:])), f"sequenza non monotona (alpha1={frac}|alpha0|)"


def test_case_switch_vs_alpha1():
    """Caso 1<->2a a alpha_a (td<=0.5-h0/hr) e 3a<->4 a alpha_c (td>=0.5+h0/hr);
    nessun cambio di caso nelle zone 2b e 3b."""
    g, t = get_gear(), thresholds()
    find = default_find_alpha(g)
    grid = np.linspace(0.02, 1.0, 120) * t["a_tip"]
    td_A, td_B = 0.5 * t["td1"], 0.5 - 0.5 * t["h0"] / t["hr"]
    td_C = 0.5 + 0.5 * t["h0"] / t["hr"]
    td_D = t["td3"] + 0.5 * (1 - t["td3"])
    aa = find(t["hr"] / 2 - td_A * t["hr"])
    ac = find(td_D * t["hr"] - t["hr"] / 2)
    for td, thr, low, high in [(td_A, aa, "2a", "1"), (td_D, ac, "3a", "4"),
                               (td_B, None, "2b", "2b"), (td_C, None, "3b", "3b")]:
        for a1 in grid:
            want = high if (thr is None or a1 > thr) else low
            got = K(td, a1)["case"]
            assert got == want, f"td={td:.4f} alpha1={a1:.5f}: atteso {want}, ottenuto {got}"


def test_integration_sanity():
    """Estremi ordinati, valori finiti, Ib e Is positivi, errore di quad piccolo
    (il caso 4 e' escluso: l'integrale diverge in -alpha_c, vedi test_case4)."""
    g, t = get_gear(), thresholds()
    find = default_find_alpha(g)
    td_A, td_D = 0.5 * t["td1"], t["td3"] + 0.5 * (1 - t["td3"])
    aa = find(t["hr"] / 2 - td_A * t["hr"])
    ac = find(td_D * t["hr"] - t["hr"] / 2)
    combos = [(td_A, t["a_tip"]), (td_A, 0.5 * aa), (0.5 - 0.3 * t["h0"] / t["hr"], t["a_tip"]),
              (0.5 + 0.3 * t["h0"] / t["hr"], 0.5 * t["a_tip"]), (td_D, 0.5 * ac)]
    for td, a1 in combos:
        r = K(td, a1)
        for name, p in r["parts"].items():
            if "lo" in p:
                assert p["lo"] < p["hi"], f"caso {r['case']} {name}: estremi invertiti [{p['lo']:.4f}, {p['hi']:.4f}]"
                assert np.isfinite(p["value"]), f"caso {r['case']} {name}: valore non finito"
                assert abs(p["err"]) <= 1e-8 * abs(r["Ib"] + r["Is"]) + 1e-30, \
                    f"caso {r['case']} {name}: errore quad {p['err']:.2e} troppo grande"
        assert r["Ib"] > 0 and r["Is"] > 0 and r["Kt"] > 0, f"caso {r['case']}: Ib={r['Ib']}, Is={r['Is']}"


def test_case4_stiffness_vanishes():
    """Nel caso 4 sezioni con hx<hc vengono ignorate e l'integrale in -alpha_c diverge
    (hx-hc -> 0 al cubo): ci si aspetta Kt ~ 0, e Kt -> 0 anche per alpha1 -> alpha_c da 3a."""
    g, t = get_gear(), thresholds()
    find = default_find_alpha(g)
    td_D = t["td3"] + 0.5 * (1 - t["td3"])
    ac = find(td_D * t["hr"] - t["hr"] / 2)
    k_ref = K(td_D, 0.3 * ac)["Kt"]
    k_seq = [K(td_D, ac - d)["Kt"] for d in (1e-2, 1e-3, 1e-4)]
    k_right = K(td_D, t["a_tip"])["Kt"]
    REPORT.append(f"caso 4: K_ref(3a, 0.3 alpha_c)={k_ref:.3e}  K(3a, alpha_c-1e-2/-3/-4)="
                  + " ".join(f"{k:.3e}" for k in k_seq) + f"  K(4)={k_right:.3e}")
    assert all(k > 0 for k in k_seq), "Kt negativo vicino ad alpha_c (quad su integrale quasi singolare)"
    assert k_seq[0] > k_seq[1] > k_seq[2], "Kt non decresce avvicinandosi ad alpha_c da sinistra (3a)"
    assert k_seq[2] < 0.05 * k_ref, "Kt non tende a 0 avvicinandosi a alpha_c da sinistra"
    assert not np.isfinite(k_right) or k_right < 0.05 * k_ref, (
        f"caso 4: Kt={k_right:.3e} non e' ~0. L'integrale diverge in -alpha_c (hx-hc -> 0, elevato al cubo): "
        "quad restituisce un valore arbitrario. Nel caso 4 conviene imporre Kt = 0 (o troncare l'integrale).")


# ------------------------------------------------------------------ continuita'
def test_continuity_tooth_damage():
    """Continuita' a td1=0.5-h0/hr, td=0.5, td3=0.5+h0/hr, per tre posizioni alpha1."""
    t = thresholds()
    for frac in A1_FRACS:
        a1 = frac * t["a_tip"]
        k_ref = K(0.01, a1)["Kt"]
        for name in ("td1", "td2", "td3"):
            kl, kr = K(t[name] - EPS_TD, a1), K(t[name] + EPS_TD, a1)
            if name == "td3" and frac == 1.0:
                # con alpha1 = punta si passa da 3b a 4: Kt -> 0 a sinistra (integrale singolare);
                # il lato destro (caso 4) e' controllato in test_case4_stiffness_vanishes
                REPORT.append(f"td: td3 alpha1=1.0|a0|  3b->4  Kt sinistra = {kl['Kt']:.3e}  (K_ref={k_ref:.3e})")
                assert kl["Kt"] < 0.05 * k_ref, "Kt non tende a 0 per td -> td3 con alpha1 = punta"
                continue
            jump = abs(kl["Kt"] - kr["Kt"]) / k_ref
            REPORT.append(f"td: {name}={t[name]:.5f} alpha1={frac}|a0|  {kl['case']:>2}->{kr['case']:<2} "
                          f"Kt {kl['Kt']:.6e} -> {kr['Kt']:.6e}  salto/K_ref = {jump:.2e}")
            assert jump < TOL_CONT, f"discontinuita' in tooth_damage a {name} (alpha1={frac}|alpha0|): {jump:.2e}"


def test_continuity_alpha1():
    """Continuita' in alpha1 a alpha_a (caso 1<->2a). La transizione a alpha_c (3a<->4)
    e' singolare (Kt -> 0) ed e' controllata in test_case4_stiffness_vanishes."""
    g, t = get_gear(), thresholds()
    find = default_find_alpha(g)
    td_A = 0.5 * t["td1"]
    for td, thr, label in [(td_A, find(t["hr"] / 2 - td_A * t["hr"]), "alpha_a")]:
        k_ref = K(td, 0.3 * thr)["Kt"]
        kl, kr = K(td, thr - EPS_A1), K(td, thr + EPS_A1)
        jump = abs(kl["Kt"] - kr["Kt"]) / k_ref
        REPORT.append(f"alpha1: {label}={thr:.5f} td={td:.4f}  {kl['case']:>2}->{kr['case']:<2} "
                      f"Kt {kl['Kt']:.6e} -> {kr['Kt']:.6e}  salto/K_ref = {jump:.2e}")
        assert jump < TOL_CONT, f"discontinuita' in alpha1 a {label}: {jump:.2e}"


def test_smoothness_alpha1():
    """Nessun salto > JUMP_MAX tra punti vicini (esclusa la zona del caso 4)."""
    g, t = get_gear(), thresholds()
    find = default_find_alpha(g)
    grid = np.linspace(0.05, 1.0, 400) * t["a_tip"]
    for td in (0.5 * t["td1"], 0.5 - 0.5 * t["h0"] / t["hr"], 0.5 + 0.5 * t["h0"] / t["hr"]):
        ks = np.array([K(td, a1)["Kt"] for a1 in grid])
        rel = np.abs(np.diff(ks)) / ks[:-1]
        assert rel.max() < JUMP_MAX, f"td={td:.4f}: salto relativo {rel.max():.3f} a alpha1={grid[rel.argmax()]:.4f}"


def test_monotonic_tooth_damage():
    """Controllo di plausibilita': piu' danno => rigidezza non maggiore (escluso il caso 4)."""
    t = thresholds()
    tds = np.linspace(0.002, 0.998, 400)
    for frac in A1_FRACS:
        a1 = frac * t["a_tip"]
        res = [K(td, a1) for td in tds]
        ks = np.array([r["Kt"] for r in res if r["case"] != "4"])
        td_ok = tds[[r["case"] != "4" for r in res]]
        bad = np.where(np.diff(ks) > 1e-6 * ks[:-1])[0]
        assert bad.size == 0, f"Kt cresce con il danno a td={td_ok[bad[0]]:.4f} (alpha1={frac}|alpha0|)"


# ----------------------------------------------------------------------- grafici
def make_plots():
    OUT.mkdir(exist_ok=True)
    g, t = get_gear(), thresholds()
    find = default_find_alpha(g)

    fig, ax = plt.subplots(figsize=(8, 5))
    tds = np.linspace(0.002, 0.998, 400)
    for frac in A1_FRACS:
        ax.plot(tds, [K(td, frac * t["a_tip"])["Kt"] for td in tds], label=f"alpha1 = {frac}|alpha0|")
    for key in ("td1", "td2", "td3"):
        ax.axvline(t[key], color="gray", ls="--", lw=0.8)
    ax.set_xlabel("tooth_damage"); ax.set_ylabel("Kt = 1/(Ib+Is)"); ax.set_yscale("log"); ax.legend(); ax.grid(True)
    ax.set_title("Kt vs tooth_damage (linee: soglie td1, 0.5, td3)")
    fig.savefig(OUT / "Kt_vs_tooth_damage.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    grid = np.linspace(0.05, 1.0, 400) * t["a_tip"]
    td_A, td_D = 0.5 * t["td1"], t["td3"] + 0.5 * (1 - t["td3"])
    cases = {"A (caso 1/2a)": td_A, "B (2b)": 0.5 - 0.5 * t["h0"] / t["hr"],
             "C (3b)": 0.5 + 0.5 * t["h0"] / t["hr"], "D (3a/4)": td_D}
    for lab, td in cases.items():
        ax.plot(grid, [K(td, a1)["Kt"] for a1 in grid], label=f"td={td:.3f} {lab}")
    ax.axvline(find(t["hr"] / 2 - td_A * t["hr"]), color="gray", ls="--", lw=0.8)
    ax.axvline(find(td_D * t["hr"] - t["hr"] / 2), color="gray", ls="--", lw=0.8)
    ax.set_xlabel("alpha1 [rad]"); ax.set_ylabel("Kt = 1/(Ib+Is)"); ax.set_yscale("log"); ax.legend(); ax.grid(True)
    ax.set_title("Kt vs alpha1 (linee: alpha_a e alpha_c)")
    fig.savefig(OUT / "Kt_vs_alpha1.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for f in tests:
        try:
            f()
            print(f"PASS  {f.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {f.__name__}: {exc}")
    print("\n".join(REPORT))
    try:
        make_plots()
        print(f"grafici salvati in {OUT}/")
    except Exception as exc:  # noqa: BLE001
        print(f"grafici non creati: {exc}")
    sys.exit(1 if failed else 0)
