"""
Physics-constrained sparse symbolic regression (SISSO-style), shared by
scripts/symbolic/fit_symbolic.py and scripts/validation/symbolic_split_comparison.py.

A model here is an explicit equation  y = c0 + c1*f1 + ... + cd*fd  with d <= 3,
where each f_i is picked from a generated library of candidate descriptors.

Library construction (dimensional analysis enforced at generation time):
  * Primaries are built only from the information XGBoost/NN already get (offsets,
    shell thickness, core radius, trap density, eps, Vmax, Type I indicator) -- band
    gaps are deliberately not used, so the comparison is fair on inputs.
  * Units are tracked as (eV, nm) exponents. Vmax enters as q*Vmax (eV). log10(Nt)
    is log10(Nt / 1 cm^-3), i.e. dimensionless, as is standard.
  * Sums/differences only between equal units; exp/log only of dimensionless
    quantities; inverses/logs/sqrts only where the *physical* domain guarantees a
    finite result (strictly positive / non-negative), not just where the training
    rows happen to be -- so a selected descriptor can't go non-finite on new data.
  * The only exponential is exp(-x) for x >= 0, which is bounded in (0, 1] on any
    input (see _unary for the exp(+x) failure that motivated this).

Fitting (SIS + exhaustive L0, as in SISSO, Ouyang et al. PRM 2, 083802 (2018)):
  for d = 1..max_terms, sure-independence-screen the n_sis descriptors most
  correlated with the current residual into a growing pool, then exhaustively
  least-squares-fit every d-subset of the pool. Among the lowest-RSS subsets, the
  best one that satisfies every physics constraint is kept.

Physics constraints are sign requirements on partial derivatives (the expected
signs from physics_sanity_checks.py), checked numerically on the training rows by
perturbing the raw input and re-evaluating the equation.

The number of terms d is chosen by grouped inner CV on the training set only
(one-standard-error rule), never by looking at the test set.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# 2*sqrt(2*m_e)/hbar in nm^-1 eV^-1/2: WKB decay exponent of a rectangular barrier,
# kappa * t * sqrt(dE) is the (dimensionless) tunneling exponent for thickness t (nm)
# and barrier height dE (eV), free-electron mass.
WKB_KAPPA = 10.25
EXP_CLIP = 50.0

TYPE_I_COLS = ["band_align_Type_I_straddling_core_confines_shell",
               "band_align_Type_I_straddling_shell_confines_core"]

# domain codes: "pos" strictly > 0, "nonneg" >= 0, "any" sign-changing
_DOMAIN_RANK = {"pos": 2, "nonneg": 1, "any": 0}


def primaries(df):
    """Physically meaningful primary descriptors. Returns {name: (values, units, domain)}."""
    dEc = df["dE_LUMO_eV"].to_numpy(float)
    dEv = df["dE_HOMO_eV"].to_numpy(float)
    t = df["shell_thick_nm"].to_numpy(float)
    Bc, Bv = np.abs(dEc), np.abs(dEv)
    Bmax, Bmin = np.maximum(Bc, Bv), np.minimum(Bc, Bv)
    type_I = df[[c for c in TYPE_I_COLS if c in df.columns]].sum(axis=1).to_numpy(float)
    E, L, D = (1, 0), (0, 1), (0, 0)
    return {
        "dEc": (dEc, E, "any"),
        "dEv": (dEv, E, "any"),
        "|dEc|": (Bc, E, "nonneg"),
        "|dEv|": (Bv, E, "nonneg"),
        "Bmax": (Bmax, E, "nonneg"),
        "Bmin": (Bmin, E, "nonneg"),
        "qV": (df["Vmax_V"].to_numpy(float), E, "pos"),
        "t": (t, L, "pos"),
        "R": (df["core_radius_nm"].to_numpy(float), L, "pos"),
        "eps": (df["eps_shell"].to_numpy(float), D, "pos"),
        "lgNt": (np.log10(df["Nt_cm3"].to_numpy(float).clip(min=1e10)), D, "pos"),
        "typeI": (type_I, D, "nonneg"),
        "wkb_max": (WKB_KAPPA * t * np.sqrt(Bmax), D, "nonneg"),
        "wkb_min": (WKB_KAPPA * t * np.sqrt(Bmin), D, "nonneg"),
    }


@dataclass(frozen=True)
class Feature:
    name: str
    expr: tuple          # ("p", name) | ("u", op, child) | ("b", op, a, b)
    units: tuple
    domain: str
    complexity: int

    def evaluate(self, prims):
        return _eval(self.expr, prims)


def _eval(expr, prims):
    kind = expr[0]
    if kind == "p":
        return prims[expr[1]][0]
    if kind == "u":
        x = _eval(expr[2], prims)
        op = expr[1]
        if op == "sq":
            return x * x
        if op == "sqrt":
            return np.sqrt(x)
        if op == "inv":
            return 1.0 / x
        if op == "log":
            return np.log(x)
        if op == "exp":
            return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))
        if op == "nexp":
            return np.exp(-np.clip(x, -EXP_CLIP, EXP_CLIP))
    if kind == "b":
        a, b = _eval(expr[2], prims), _eval(expr[3], prims)
        op = expr[1]
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "+":
            return a + b
        if op == "-":
            return a - b
    raise ValueError(f"bad expression {expr}")


def _unary(f):
    out = []
    dimless = f.units == (0, 0)
    mk = lambda name, op, units, domain: Feature(name, ("u", op, f.expr), units, domain, f.complexity + 1)
    if f.name != "typeI":  # the indicator squared/rooted is itself
        out.append(mk(f"({f.name})^2", "sq", (2 * f.units[0], 2 * f.units[1]),
                      "pos" if f.domain == "pos" else "nonneg"))
        if f.domain != "any":
            out.append(mk(f"sqrt({f.name})", "sqrt", (f.units[0] / 2, f.units[1] / 2), f.domain))
    if f.domain == "pos":
        out.append(mk(f"1/({f.name})", "inv", (-f.units[0], -f.units[1]), "pos"))
        if dimless:
            out.append(mk(f"ln({f.name})", "log", (0, 0), "any"))
    # Only the bounded exponential: exp(-x) with x >= 0 lies in (0, 1] for ANY input, so it
    # can't blow up on new data. exp(+x) was in the first version of this library and
    # inner CV picked exp(lgNt/eps) on 3 of 30 comparison fits; test shells with eps
    # near 1.3 sent it to ~1e8 and R^2 to ~-2e4. For the log-scale targets exp(+x) is
    # redundant anyway (a linear term in log space already is an exponential).
    if dimless and f.name != "typeI" and f.domain != "any":
        out.append(mk(f"exp(-{f.name})", "nexp", (0, 0), "pos"))
    return out


def _binary(a, b):
    out = []
    mk = lambda name, op, x, y, units, domain: Feature(name, ("b", op, x.expr, y.expr), units, domain,
                                                        x.complexity + y.complexity + 1)
    prod_domain = min(a.domain, b.domain, key=_DOMAIN_RANK.get)
    out.append(mk(f"{a.name}*{b.name}", "*", a, b, (a.units[0] + b.units[0], a.units[1] + b.units[1]),
                  prod_domain))
    if b.domain == "pos":
        out.append(mk(f"{a.name}/{b.name}", "/", a, b, (a.units[0] - b.units[0], a.units[1] - b.units[1]),
                      a.domain))
    if a.domain == "pos":
        out.append(mk(f"{b.name}/{a.name}", "/", b, a, (b.units[0] - a.units[0], b.units[1] - a.units[1]),
                      b.domain))
    if a.units == b.units and "typeI" not in (a.name, b.name):
        out.append(mk(f"({a.name}+{b.name})", "+", a, b, a.units, prod_domain if prod_domain != "any" else "any"))
        out.append(mk(f"({a.name}-{b.name})", "-", a, b, a.units, "any"))
    return out


def build_library(prims):
    """All tier-1/tier-2 descriptors. Deterministic given the primary set."""
    base = [Feature(n, ("p", n), u, d, 0) for n, (_, u, d) in prims.items()]
    lib = list(base)
    for f in base:
        lib.extend(_unary(f))
    for a, b in itertools.combinations(base, 2):
        for g in _binary(a, b):
            lib.append(g)
            # one more nonlinearity on dimensionless combinations (V/B, t/R, ...)
            if g.units == (0, 0) and g.expr[1] in ("*", "/"):
                lib.extend(u for u in _unary(g) if u.expr[1] in ("log", "nexp", "sqrt"))
    return lib


@dataclass
class Constraint:
    """Expected sign of d(prediction)/d(driver), checked on rows where `mask` is True."""
    name: str
    perturb: callable    # df -> df with the driver increased slightly
    sign: int
    mask: callable = None
    min_frac: float = 0.99


def _bump(col, fn):
    def perturb(df):
        out = df.copy()
        out[col] = fn(out[col])
        return out
    return perturb


def _type_I_mask(df):
    return df[[c for c in TYPE_I_COLS if c in df.columns]].sum(axis=1).to_numpy() > 0


# Same expected signs as physics_sanity_checks.py's CHECKS.
CONSTRAINTS = {
    "hysteresis_window_V": [
        Constraint("d/d(Nt) >= 0", _bump("Nt_cm3", lambda s: s * 1.01), +1),
        Constraint("d/d|dE_LUMO| >= 0 (Type I)", _bump("dE_LUMO_eV", lambda s: s + 0.01 * np.sign(s)), +1,
                   _type_I_mask),
        Constraint("d/d|dE_HOMO| >= 0 (Type I)", _bump("dE_HOMO_eV", lambda s: s + 0.01 * np.sign(s)), +1,
                   _type_I_mask),
    ],
    "log10_on_off_ratio": [
        Constraint("d/d(t_shell) <= 0", _bump("shell_thick_nm", lambda s: s + 0.01), -1),
    ],
    "log10_retention_tau_s": [
        Constraint("d/d(t_shell) >= 0", _bump("shell_thick_nm", lambda s: s + 0.01), +1),
    ],
}


class SparseSymbolicRegressor:
    def __init__(self, max_terms=3, n_sis=40, n_check=500, constraints=(), ridge=1e-8, corr_dedupe=0.9999):
        self.max_terms = max_terms
        self.n_sis = n_sis
        self.n_check = n_check
        self.constraints = list(constraints)
        self.ridge = ridge
        self.corr_dedupe = corr_dedupe

    # ---------- library on a given training set ----------
    def _prepare(self, df):
        prims = primaries(df)
        lib = build_library(prims)
        F = np.column_stack([f.evaluate(prims) for f in lib])
        keep = np.isfinite(F).all(axis=0) & (F.std(axis=0) > 1e-12)
        lib = [f for f, k in zip(lib, keep) if k]
        F = F[:, keep]
        mu, sd = F.mean(axis=0), F.std(axis=0)
        Z = (F - mu) / sd
        # drop near-duplicates, keeping the simplest representative
        order = np.argsort([f.complexity for f in lib], kind="stable")
        Z, F, mu, sd = Z[:, order], F[:, order], mu[order], sd[order]
        lib = [lib[i] for i in order]
        C = np.abs(Z.T @ Z) / len(Z)
        kept = []
        for j in range(len(lib)):
            if not kept or C[j, kept].max() < self.corr_dedupe:
                kept.append(j)
        return [lib[j] for j in kept], F[:, kept], Z[:, kept], mu[kept], sd[kept]

    def _constraint_deltas(self, df, lib):
        """Per constraint: (feature-value change under the perturbation, row mask, sign, min_frac)."""
        out = []
        for c in self.constraints:
            dfp = c.perturb(df)
            p0, p1 = primaries(df), primaries(dfp)
            dF = np.column_stack([f.evaluate(p1) - f.evaluate(p0) for f in lib])
            mask = np.ones(len(df), bool) if c.mask is None else c.mask(df)
            out.append((c, dF[mask]))
        return out

    def _satisfies(self, idx, coef_raw, deltas):
        report = {}
        ok = True
        for c, dF in deltas:
            dpred = dF[:, idx] @ coef_raw
            frac = np.mean(c.sign * dpred >= -1e-9)
            # a zero derivative satisfies a sign constraint trivially; record whether the
            # driver actually enters the equation so "passes" isn't mistaken for "captures"
            report[c.name] = (frac, bool(np.any(np.abs(dpred) > 1e-12)))
            ok &= frac >= c.min_frac
        return ok, report

    # ---------- SIS + L0 for every d up to max_terms ----------
    def _search(self, Z, y, sd, deltas):
        yc = y - y.mean()
        yy = yc @ yc
        n = len(y)
        pool = []
        best = {}
        resid = yc
        for d in range(1, self.max_terms + 1):
            score = np.abs(Z.T @ resid)
            score[pool] = -1
            pool.extend(np.argsort(-score)[:self.n_sis].tolist())
            P = np.array(pool)
            G = Z[:, P].T @ Z[:, P] + self.ridge * n * np.eye(len(P))
            b = Z[:, P].T @ yc
            combos = np.array(list(itertools.combinations(range(len(P)), d)))
            Gs = G[combos[:, :, None], combos[:, None, :]]
            bs = b[combos]
            beta = np.linalg.solve(Gs, bs[..., None])[..., 0]
            rss = yy - (bs * beta).sum(axis=1)
            # Linearly dependent subsets (e.g. {a-b, a, b}) are degenerate, not d-term models:
            # pairwise dedupe can't catch them. Gs/n is a correlation matrix (+ the ridge), so its
            # determinant is ~0 exactly when the subset is (near-)dependent. Found on noise-free
            # labels in recovery_test.py, where such a subset ties the true 2-term model on RSS.
            # The threshold must sit well above the ridge's own floor: an exactly dependent triple
            # measured det = 2e-8 because of the ridge (a first version used 1e-8 and missed it);
            # healthy subsets measured ~0.5.
            if d > 1:
                rss[np.linalg.det(Gs / n) < 1e-6] = np.inf
            top = np.argsort(rss)[:self.n_check]
            chosen = None
            for rank, k in enumerate(top):
                idx = P[combos[k]]
                coef_raw = beta[k] / sd[idx]
                ok, report = self._satisfies(idx, coef_raw, deltas)
                if ok:
                    chosen = (idx, beta[k], rss[k], rank, report)
                    break
            unconstrained = (P[combos[top[0]]], beta[top[0]], rss[top[0]])
            if chosen is None:  # nothing in the top n_check passes: keep best, flag it
                idx, bk, r = unconstrained
                _, report = self._satisfies(idx, bk / sd[idx], deltas)
                chosen = (idx, bk, r, -1, report)
            best[d] = {"idx": chosen[0], "beta_std": chosen[1], "rss": chosen[2],
                       "constraint_rank": chosen[3], "constraint_report": chosen[4],
                       "unconstrained_idx": unconstrained[0]}
            resid = yc - Z[:, chosen[0]] @ chosen[1]
        return best

    def _fit_all_d(self, df, y):
        lib, F, Z, mu, sd = self._prepare(df)
        deltas = self._constraint_deltas(df, lib)
        return lib, mu, sd, self._search(Z, y, sd, deltas), y.mean()

    @staticmethod
    def _predict_with(lib, mu, sd, model, y_mean, df):
        prims = primaries(df)
        Fsel = np.column_stack([lib[i].evaluate(prims) for i in model["idx"]])
        return y_mean + ((Fsel - mu[model["idx"]]) / sd[model["idx"]]) @ model["beta_std"]

    # ---------- public API ----------
    def fit(self, df, y, groups=None, n_folds=5, n_terms=None):
        """n_terms=None picks the number of terms by grouped inner CV (needs groups);
        an explicit n_terms skips the CV (used by bootstrap refits of an already-chosen d)."""
        y = np.asarray(y, float)
        if n_terms is not None:
            self.cv_rmse_, self.n_terms_ = {}, n_terms
            self.lib_, self.mu_, self.sd_, self.models_, self.y_mean_ = self._fit_all_d(df, y)
            self.model_ = self.models_[n_terms]
            return self
        # inner grouped CV over d, on the training set only
        cv_rmse = {d: [] for d in range(1, self.max_terms + 1)}
        for tr, va in GroupKFold(n_folds).split(df, y, groups):
            lib, mu, sd, best, ym = self._fit_all_d(df.iloc[tr], y[tr])
            for d, m in best.items():
                pred = self._predict_with(lib, mu, sd, m, ym, df.iloc[va])
                cv_rmse[d].append(np.sqrt(np.mean((pred - y[va]) ** 2)))
        self.cv_rmse_ = {d: (np.mean(v), np.std(v) / np.sqrt(len(v))) for d, v in cv_rmse.items()}
        d_best = min(self.cv_rmse_, key=lambda d: self.cv_rmse_[d][0])
        thresh = self.cv_rmse_[d_best][0] + self.cv_rmse_[d_best][1]
        self.n_terms_ = min(d for d in self.cv_rmse_ if self.cv_rmse_[d][0] <= thresh)

        self.lib_, self.mu_, self.sd_, self.models_, self.y_mean_ = self._fit_all_d(df, y)
        self.model_ = self.models_[self.n_terms_]
        return self

    def predict(self, df):
        return self._predict_with(self.lib_, self.mu_, self.sd_, self.model_, self.y_mean_, df)

    def terms(self, model=None):
        return [self.lib_[i].name for i in (model or self.model_)["idx"]]

    def raw_coefficients(self, model=None):
        m = model or self.model_
        coef = m["beta_std"] / self.sd_[m["idx"]]
        intercept = self.y_mean_ - coef @ self.mu_[m["idx"]]
        return intercept, coef

    def equation(self, model=None, target="y"):
        c0, coef = self.raw_coefficients(model)
        s = f"{target} = {c0:.4g}"
        for c, name in zip(coef, self.terms(model)):
            s += f" {'+' if c >= 0 else '-'} {abs(c):.4g}*{name}"
        return s

    def constraints_satisfied(self, model=None):
        return (model or self.model_)["constraint_rank"] >= 0
