from typing import Dict, List, Tuple, Any, Union, Optional
import time
import numpy as np
from math import comb
import dimod
import itertools
from dimod import SimulatedAnnealingSampler, ExactSolver
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
import warnings

def _qubo_to_diagonal_HP(qubo: Dict[Tuple[int,int], float]) -> Tuple[sp.csr_matrix, int]:
    """
    Build the diagonal problem Hamiltonian H_P as a sparse diagonal matrix (2^n x 2^n).
    Returns (H_P_sparse, n_qubits).
    """
    if not qubo:
        raise ValueError("Empty QUBO provided.")
    max_idx = 0
    for (i,j) in qubo.keys():
        max_idx = max(max_idx, i, j)
    n = max_idx + 1
    Q = np.zeros((n, n), dtype=float)
    for (i,j), v in qubo.items():
        if i == j:
            Q[i,i] += v
        else:
            if i < j:
                Q[i,j] += v
            else:
                Q[j,i] += v

    dim = 1 << n
    if dim > 1<<20:
        warnings.warn(f"H_P dimension {dim} is large; this will use ~{dim*8/1024**3:.2f} GiB if made dense.")

    idxs = np.arange(dim, dtype=np.int64)
    shifts = np.arange(n-1, -1, -1, dtype=np.int64)
    X = ((idxs[:, None] >> shifts[None, :]) & 1).astype(np.int8)

    diag = np.diag(Q)
    linear = np.sum(X * diag[None, :], axis=1)

    quad = np.zeros(dim, dtype=float)
    iu = np.triu_indices(n, k=1)
    if iu[0].size:
        for a,b in zip(iu[0], iu[1]):
            qval = Q[a,b]
            if qval != 0.0:
                quad += qval * (X[:,a] * X[:,b])

    energies = (linear + quad).astype(float)
    H_P = sp.diags(energies, offsets=0, format='csr')
    return H_P, n


def _build_transverse_field_HB(n: int, gamma: float = 1.0) -> sp.csr_matrix:
    """
    Build  H_B = -gamma * sum_i sigma_x^i.
    """
    sx = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    id2 = sp.eye(2, format='csr', dtype=float)
    dim = 1 << n
    HB = sp.csr_matrix((dim, dim), dtype=float)
    for i in range(n):
        op = None
        for q in range(n):
            mat = sx if q == i else id2
            op = mat if op is None else sp.kron(op, mat, format='csr')
        HB += -gamma * op
    return HB


def compute_homotopy_spectrum(
    qubo: Dict[Tuple[int,int], float],
    normalize: bool = False,
    num_s: int = 201,
    k_lowest: int = 6,
    gamma: float = 1.0,
    tol: float = 1e-8,
    maxiter: int = None,
    return_eigenvectors: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the k_lowest eigenvalues of H(s) = (1-s)H_B + s H_P using sparse eigensolver.
    Returns (s_vals, eigvals_array, n_qubits). eigvals_array shape = (len(s_vals), k_lowest).
    If return_eigenvectors=True, also returns eigenvectors (not recommended for large dim).
    """
    H_P, n = _qubo_to_diagonal_HP(qubo)
    dim = H_P.shape[0]
    H_B = _build_transverse_field_HB(n, gamma=gamma)

    if normalize:
        diag_HP = H_P.diagonal()
        max_abs = float(np.max(np.abs(diag_HP))) if diag_HP.size else 1.0
        if max_abs == 0:
            max_abs = 1.0
        H_P = H_P / max_abs
        H_B = H_B 

    s_vals = np.linspace(0.0, 1.0, num_s)
    k = min(k_lowest, dim-1) if dim > 1 else 1
    eigs_out = np.zeros((len(s_vals), k), dtype=float)
    eigvecs_out = None
    if return_eigenvectors:
        eigvecs_out = np.zeros((len(s_vals), dim, k), dtype=float)  # big; optional

    for idx, s in enumerate(s_vals):
        Hs = (1.0 - s) * H_B + s * H_P 
        compute_k = k
        if dim <= 2 or compute_k >= dim:
            w = sla.eigvalsh(Hs.todense() if sp.isspmatrix(Hs) else Hs)
            vals = np.sort(w)[:compute_k]
            eigs_out[idx, :len(vals)] = vals
            if return_eigenvectors:
                pass
            continue

        try:
            vals, vecs = spla.eigsh(Hs, k=compute_k, which='SA', tol=tol, maxiter=maxiter, return_eigenvectors=True)
            order = np.argsort(vals)
            vals = vals[order]
            eigs_out[idx, :] = vals
            if return_eigenvectors:
                eigvecs_out[idx, :, :] = vecs[:, order]
        except Exception as e:
            warnings.warn(f"eigsh failed at s={s:.4f} with error {e}; falling back to dense diagonalization for this s.")
            w = sla.eigvalsh(Hs.todense() if sp.isspmatrix(Hs) else Hs)
            vals = np.sort(w)[:compute_k]
            eigs_out[idx, :len(vals)] = vals
            if return_eigenvectors:
                pass

    if return_eigenvectors:
        return s_vals, eigs_out, n, eigvecs_out
    return s_vals, eigs_out, n

def plot_homotopy_eigenvalues(
    qubo: Dict[Tuple[int, int], float],
    normalize: bool = False,
    num_s: int = 201,
    k_lowest: int = 6,
    figsize: Tuple[int, int] = (10, 6),
    zoom_halfwidth: float = 0.025,
    gamma: float = 1.0,
    n_refine: int = 301
) -> Tuple[Any, Any]:
    """
    Plot the lowest k eigenvalues λ_j(s) of H(s) = (1-s) H_B + s H_P for s∈[0,1].
    Automatically locate the minimum ground/first-excited gap and include a zoom around the minimum.
    """

    s_vals, eigs_out, n = compute_homotopy_spectrum(qubo, normalize=normalize, num_s=num_s, k_lowest=k_lowest, gamma=gamma)
    if eigs_out.shape[1] < 2:
        raise RuntimeError("Need at least two eigenvalues to compute a gap.")
    gaps = eigs_out[:, 1] - eigs_out[:, 0]
    min_idx = int(np.argmin(gaps))
    s_min = float(s_vals[min_idx])
    gap_min = float(gaps[min_idx])

    fig, ax = plt.subplots(figsize=figsize)
    for j in range(eigs_out.shape[1]):
        ax.plot(s_vals, eigs_out[:, j], label=f'λ_{j}')
    ax.plot(s_vals, eigs_out[:, 0], color='C0', lw=2)
    ax.set_xlabel('s (annealing parameter)')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Homotopy spectrum (n={n}, normalize={normalize}) — min Δ≈{gap_min:.3e} at s={s_min:.4f}')
    ax.grid(alpha=0.25)
    ax.axvline(s_min, color='k', ls='--', alpha=0.6)
    ax.annotate(f's*={s_min:.4f}\nΔ={gap_min:.3e}', xy=(s_min, (eigs_out[min_idx, 0] + eigs_out[min_idx, 1]) / 2),
                xytext=(min(1.0, s_min + 0.05), eigs_out[min_idx, 0]),
                arrowprops=dict(arrowstyle='->', lw=1.2), bbox=dict(fc='white', alpha=0.8))

    left = max(0.0, s_min - zoom_halfwidth)
    right = min(1.0, s_min + zoom_halfwidth)
    if right - left < 1e-12:
        left = max(0.0, s_min - max(1e-6, zoom_halfwidth))
        right = min(1.0, s_min + max(1e-6, zoom_halfwidth))
    s_ref = np.linspace(left, right, max(3, int(n_refine)))
    HP, _ = _qubo_to_diagonal_HP(qubo)
    HB = _build_transverse_field_HB(n, gamma=gamma)
    if sp.issparse(HP):
        HP = HP.toarray()
    if sp.issparse(HB):
        HB = HB.toarray()
    HP = np.asarray(HP, dtype=float)
    HB = np.asarray(HB, dtype=float)

    k = eigs_out.shape[1]
    eigs_ref = np.zeros((len(s_ref), k), dtype=float)
    for ii, s in enumerate(s_ref):
        Hs_ref = (1.0 - float(s)) * HB + float(s) * HP
        vals = sla.eigvalsh(Hs_ref)
        eigs_ref[ii, :min(k, vals.size)] = np.sort(vals)[:k]

    axins = inset_axes(ax, width="42%", height="42%", bbox_to_anchor=(0.56, 0.06, 0.40, 0.40),
                       bbox_transform=ax.transAxes, loc='lower left', borderpad=0)
    for j in range(eigs_ref.shape[1]):
        axins.plot(s_ref, eigs_ref[:, j], lw=1.2)
    axins.set_xlim(left, right)
    ymin = float(np.min(eigs_ref[:, :2]))
    ymax = float(np.max(eigs_ref[:, :2]))
    if ymax <= ymin:
        ymin -= 1e-12
        ymax += 1e-12
    span = ymax - ymin
    tighten = 0.12
    low = ymin + tighten * span
    high = ymax - tighten * span
    if low >= high:
        low = ymin - 0.01 * span
        high = ymax + 0.01 * span
    axins.set_ylim(low, high)
    if (span / max(abs(ymin), 1.0)) < 1e-3:
        axins.set_yscale('log')
    axins.tick_params(axis='both', which='major', labelsize=8)
    axins.set_title('Zoom near min gap', fontsize=9)
    axins.spines['top'].set_linewidth(0.8)
    axins.spines['right'].set_linewidth(0.8)
    axins.spines['bottom'].set_linewidth(0.8)
    axins.spines['left'].set_linewidth(0.8)
    ax.legend(loc='upper left')
    fig.subplots_adjust(bottom=0.11, right=0.975, top=0.90)
    return fig, ax



def plot_gap_vs_s(
    qubo: Dict[Tuple[int, int], float],
    normalize: bool = False,
    num_s: int = 401,
    figsize: Tuple[int, int] = (8, 5),
    zoom_halfwidth: float = 0.02,
    gamma: float = 1.0
) -> Tuple[Any, Any]:
    """
    Plot the instantaneous gap(s) = lanbda_1(s) - lambda_0(s) as a function of s and
    add a zoomed inset around the minimizing s.
    """
    s_vals, eigs_out, n = compute_homotopy_spectrum(qubo, normalize=normalize, num_s=num_s, k_lowest=6, gamma=gamma)
    gaps = eigs_out[:, 1] - eigs_out[:, 0]
    min_idx = int(np.argmin(gaps))
    s_min = s_vals[min_idx]
    gap_min = gaps[min_idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s_vals, gaps, lw=2)
    ax.set_xlabel('s')
    ax.set_ylabel('Gap Δ(s)')
    ax.set_title(f'Gap vs s (n={n}, normalize={normalize}) — min Δ={gap_min:.3e} at s={s_min:.4f}')
    ax.grid(alpha=0.25)

    left = max(0.0, s_min - zoom_halfwidth)
    right = min(1.0, s_min + zoom_halfwidth)
    mask = (s_vals >= left) & (s_vals <= right)
    axins = inset_axes(ax, width="45%", height="35%", loc='upper right', borderpad=2)
    axins.plot(s_vals, gaps, lw=1.0)
    axins.set_xlim(left, right)
    ymin = np.min(gaps[mask]) * 0.5 if np.any(mask) else np.min(gaps) * 0.5
    ymax = np.max(gaps[mask]) * 1.5 if np.any(mask) else np.max(gaps) * 1.5
    axins.set_ylim(ymin, ymax)
    axins.set_yscale('log')
    axins.set_title('Zoom on min gap (log scale)')
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    fig.subplots_adjust(right=0.95, top=0.92) 
    return fig, ax


def extract_qubo_for_indices(Q_full: Dict[Tuple[int,int], float], indices: List[int]) -> Dict[Tuple[int,int], float]:
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
    Q_new = {}
    for (i,j), val in Q_full.items():
        if i in indices and j in indices:
            ni, nj = index_map[i], index_map[j]
            Q_new[(ni, nj)] = val
    return Q_new

def sweep_gap_vs_size(
    full_qubo: Dict[Tuple[int,int], float],
    full_drug_ids: List[str],
    sizes: List[int],
    sample_per_size: int = 5,
    allowed_sizes: List[int] = None
    ) -> List[Dict[str, Any]]:
    """
    For each n in sizes, choose sample_per_size random subsets of drug indices of size n,
    build a reduced QUBO for that subset (extract corresponding linear/quadratic terms),
    and run estimate_gap. Returns list of records with timing and gap info.
    """
    rng = np.random.RandomState(123)
    results = []
    n_full = len(full_drug_ids)

    for n in sizes:
        if n > n_full:
            continue
        for s in range(sample_per_size):
            chosen = sorted(rng.choice(range(n_full), size=n, replace=False).tolist())
            sub_qubo = extract_qubo_for_indices(full_qubo, chosen)
            sub_ids = [full_drug_ids[i] for i in chosen]
            rec = {'size': n, 'sample': s, 'subset_indices': chosen, 'subset_ids': sub_ids}
            t0 = time.time()
            try:
                gap_info = estimate_gap(sub_qubo, sub_ids, allowed_sizes=allowed_sizes)
                rec.update(gap_info)
            except Exception as e:
                rec.update({'error': str(e)})
            t1 = time.time()
            rec['wall_time_total'] = t1 - t0
            results.append(rec)
    return results


def estimate_gap(
    qubo: Dict[Tuple[int,int], float],
    drug_ids: List[str],
    allowed_sizes: List[int] = None
) -> Dict[str, Any]:
    """
    If allowed_sizes is provided, enumerate only allowed combos (fast) and get exact gap.
    """
    n = len(drug_ids)
    if allowed_sizes:
        # enumerate only combos of allowed sizes
        total_candidates = sum(comb(n, k) for k in allowed_sizes if 0 <= k <= n)
        if total_candidates >= 2000000:  
            print("[Warning] Large number of candidates for allowed_sizes enumeration:", total_candidates)
        t0 = time.time()
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        candidates = []
        for k in allowed_sizes:
            if 0 <= k <= n:
                candidates.extend(itertools.combinations(range(n), k))
        energies = []
        for combi in candidates:
            sample = {i: (1 if i in combi else 0) for i in range(n)}
            energies.append(float(bqm.energy(sample)))
        energies = np.array(energies)
        order = np.argsort(energies)
        uniq_vals, idx = np.unique(energies[order], return_index=True)
        E_gs = float(uniq_vals[0]) if uniq_vals.size > 0 else None
        E_es = float(uniq_vals[1]) if uniq_vals.size > 1 else None
        t1 = time.time()
        return {
            'method': 'enumeration_allowed_sizes',
            'E_gs': E_gs,
            'E_es': E_es,
            'gap': (E_es - E_gs) if (E_gs is not None and E_es is not None) else None,
            'time': t1 - t0,
            'total_candidates': int(total_candidates)
        }
        
    else:
        try:
            res = exact_gap(qubo, drug_ids)
            res['method'] = 'exact_enumeration'
            return res
        except Exception as e:
            print("[Warning] exact enumeration failed, falling back to sampling:", e)

def exact_gap(qubo: Dict[Tuple[int,int], float], drug_ids: List[str]) -> Dict[str, Any]:
    """
    Exact enumeration (ExactSolver). Returns dict with keys:
      'E_gs', 'E_es', 'gap', 'gs_states' (list), 'es_states' (list), 'num_states' (2**n), 'time'
    Only feasible for small n (n <= ~20).
    """
    t0 = time.time()
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    sampleset = ExactSolver().sample(bqm)  # enumerates all states
    recs = list(sampleset.record)
    energies = np.array([float(r.energy) for r in recs])

    order = np.argsort(energies)
    sorted_energies = energies[order]

    uniq_vals, indices = np.unique(sorted_energies, return_index=True)
    if uniq_vals.size == 0:
        raise RuntimeError("No states found")
    E_gs = float(uniq_vals[0])
    if uniq_vals.size == 1:
        E_es = float(uniq_vals[0])
    else:
        E_es = float(uniq_vals[1])

    gs_states = []
    es_states = []
    var_order = list(sampleset.variables)
    for r in recs:
        if abs(float(r.energy) - E_gs) < 1e-12:
            bits = tuple(int(b) for b in r.sample)
            sel = tuple(drug_ids[i] for i, bit in enumerate(bits) if bit == 1)
            gs_states.append(sel)
        elif abs(float(r.energy) - E_es) < 1e-12:
            bits = tuple(int(b) for b in r.sample)
            sel = tuple(drug_ids[i] for i, bit in enumerate(bits) if bit == 1)
            es_states.append(sel)
    t1 = time.time()
    return {
        'E_gs': E_gs,
        'E_es': E_es,
        'gap': E_es - E_gs,
        'gs_states': gs_states,
        'es_states': es_states,
        'num_states': len(energies),
        'time': t1 - t0
    }

def _ensure_df(df_res: Union[pd.DataFrame, list]) -> pd.DataFrame:
    if isinstance(df_res, pd.DataFrame):
        return df_res.copy()
    return pd.DataFrame(df_res)

def plot_gap_scaling(
    df_res: Union[pd.DataFrame, list],
    size_col: str = 'size',
    gap_col: str = 'gap',
    groupby_agg: str = 'mean',   
    figsize: Tuple[int,int] = (8,5),
    top_n_points: Optional[int] = None,
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot scaling of observed gap (E_es - E_gs) vs subset size.

    - Shows scatter of individual samples, and mean +/- std errorbar per size.
    - Ignores NaN/None/negative gaps (prints a warning if many filtered).
    """
    df = _ensure_df(df_res)
    if size_col not in df.columns or gap_col not in df.columns:
        raise ValueError(f"Data must contain columns '{size_col}' and '{gap_col}'")

    df = df.copy()
    df[gap_col] = pd.to_numeric(df[gap_col], errors='coerce')
    df[size_col] = pd.to_numeric(df[size_col], errors='coerce')

    before = len(df)
    df = df[df[gap_col].notna()]
    df = df[df[size_col].notna()]

    # remove non positive gaps (probably untrustworthy)
    df = df[df[gap_col] > 0]
    filtered = before - len(df)
    if filtered > 0:
        print(f"[plot_gap_scaling] Filtered out {filtered} rows with missing/non positive gaps.")

    if df.empty:
        raise RuntimeError("No valid gap data to plot after filtering.")

    stats = df.groupby(size_col)[gap_col].agg(['mean','std','count']).reset_index().sort_values(size_col)
    sizes = stats[size_col].values
    means = stats['mean'].values
    stds = stats['std'].values
    counts = stats['count'].values

    rng = np.random.RandomState(0)
    jitter = (rng.rand(len(df)) - 0.5) * 0.2

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[size_col] + jitter, df[gap_col], alpha=0.5, s=40, label='samples')
    ax.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=4, color='C1', lw=2, label='mean $\\pm$ std')

    ax.set_xlabel('Subset size $(n)$', fontsize=12)
    ax.set_ylabel('Observed gap $(E_{es} - E_{gs})$', fontsize=12)
    ax.set_title(title or 'Gap scaling vs subset size', fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    return fig, ax

def plot_walltime_scaling(
    df_res: Union[pd.DataFrame, list],
    size_col: Optional[str] = None,
    wall_time_col: Optional[str] = None,
    use_log_y: bool = True,
    figsize: Tuple[int,int] = (8,5),
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot measured wall-time vs subset size.
    """

    if not isinstance(df_res, pd.DataFrame):
        df = pd.DataFrame(df_res)
    else:
        df = df_res.copy()

    size_candidates = ['size', 'n', 'subset_size', 'subset', 'k', 'num_nodes']
    walltime_candidates = [
        'wall_time_total', 'total_wall_time', 'time_s (estimate)', 'time_s',
        'time', 'time_seconds', 'total_time', 'time_total', 'time_s (est)', 'time_sec'
    ]

    def _first_match(colname, candidates):
        if colname and colname in df.columns:
            return colname
        for c in candidates:
            if c in df.columns:
                return c
        return None

    size_col_found = _first_match(size_col, size_candidates)
    wall_col_found = _first_match(wall_time_col, walltime_candidates)

    if size_col_found is None or wall_col_found is None:
        msg_parts = []
        if size_col_found is None:
            msg_parts.append(f"Could not find a 'size' column. Tried: {size_candidates}")
        if wall_col_found is None:
            msg_parts.append(f"Could not find a wall-time column. Tried: {walltime_candidates}")
        msg = " ; ".join(msg_parts)
        avail = list(df.columns)
        raise ValueError(f"{msg}. Available columns: {avail}")

    df = df.rename(columns={size_col_found: 'size', wall_col_found: 'wall_time_total'})

    df['wall_time_total'] = pd.to_numeric(df['wall_time_total'], errors='coerce')
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df = df[df['wall_time_total'].notna() & df['size'].notna()]

    if df.empty:
        raise RuntimeError("No valid wall-time rows after filtering NaNs.")

    stats = df.groupby('size')['wall_time_total'].agg(['mean','std','count']).reset_index().sort_values('size')
    sizes = stats['size'].values
    means = stats['mean'].values
    stds = stats['std'].values

    rng = np.random.RandomState(1)
    jitter = (rng.rand(len(df)) - 0.5) * 0.2

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df['size'] + jitter, df['wall_time_total'], alpha=0.5, s=40, label='samples')
    ax.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=4, color='C2', lw=2, label='mean $\\pm$ std')

    ax.set_xlabel('Subset size (n)', fontsize=12)
    ax.set_ylabel('Wall time (s)', fontsize=12)
    ax.set_title(title or 'Wall-time vs subset size', fontsize=14)
    ax.grid(True, alpha=0.25)
    if use_log_y:
        ax.set_yscale('log')
        ax.set_ylabel('Wall time $(s)$ (log scale)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_adiabatic_time_scaling(
    df_res: Union[pd.DataFrame, list],
    size_col: str = 'size',
    gap_col: str = 'gap',
    prefactor: float = 1.0,
    eps_gap: float = 1e-12,
    figsize: Tuple[int,int] = (8,5),
    title: str | None = None
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    df = pd.DataFrame(df_res) if not isinstance(df_res, pd.DataFrame) else df_res.copy()

    if size_col not in df.columns or gap_col not in df.columns:
        raise ValueError(f"Data must contain '{size_col}' and '{gap_col}'; available: {list(df.columns)}")
    
    df[size_col] = pd.to_numeric(df[size_col], errors='coerce')
    df[gap_col] = pd.to_numeric(df[gap_col], errors='coerce')
    df = df.dropna(subset=[size_col, gap_col])
    df = df[df[gap_col] > 0]

    if df.empty:
        raise RuntimeError("No valid positive gaps to compute adiabatic times.")
    
    df['_T'] = float(prefactor) / (df[gap_col].clip(lower=eps_gap) ** 2)
    grouped = df.groupby(size_col)['_T']
    sizes = np.array(sorted(grouped.groups.keys()))

    if sizes.size < 2:
        raise RuntimeError("Need at least two distinct sizes to fit.")
    
    Tgeom = np.array([np.exp(np.mean(np.log(grouped.get_group(s).clip(lower=eps_gap)))) for s in sizes])
    log_std = np.array([np.std(np.log(grouped.get_group(s).clip(lower=eps_gap))) for s in sizes])
    lower_band = Tgeom * np.exp(-log_std)
    upper_band = Tgeom * np.exp(+log_std)

    # Modified model: no prefactor C
    def model(n_val, alpha, B):
        return (2.0 ** (alpha * n_val)) + B
    
    # Initial guess for alpha, B
    log2T = np.log2(Tgeom)
    A = np.vstack([sizes, np.ones_like(sizes)]).T
    sol, *_ = np.linalg.lstsq(A, log2T, rcond=None)
    alpha0, log2C0 = float(sol[0]), float(sol[1])   # C is ignored now
    B0 = max(0.0, 0.1 * np.min(Tgeom))
    p0 = [alpha0, B0]
    lower = [-10.0, 0.0]
    upper = [10.0, np.max(Tgeom)]

    popt, pcov = curve_fit(model, sizes, Tgeom, p0=p0, bounds=(lower, upper), maxfev=200000)

    alpha_fit, B_fit = float(popt[0]), float(popt[1])
    T_pred = model(sizes, alpha_fit, B_fit)
    ss_res = np.sum((Tgeom - T_pred) ** 2)
    ss_tot = np.sum((Tgeom - np.mean(Tgeom)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fit_result = {'method': 'scipy_curve_fit_geom_2alphaB', 'alpha': alpha_fit, 'B': B_fit, 'cov': pcov, 'r2': r2}

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.RandomState(2)
    jitter = (rng.rand(len(df)) - 0.5) * 0.2
    ax.scatter(df[size_col] + jitter, df['_T'], alpha=0.35, s=30, label='samples')
    ax.errorbar(sizes, Tgeom, yerr=[Tgeom - lower_band, upper_band - Tgeom], fmt='o-', capsize=4, color='C3', lw=2, label='geom-mean ± log-std')
    
    n_plot = np.linspace(np.min(sizes), np.max(sizes), 300)
    T_fit = model(n_plot, alpha_fit, B_fit)

    ax.plot(n_plot, T_fit, 'k-', lw=2, label=f'Fit:  $2^{{\\alpha n}} + B$')

    ax.set_yscale('log', base=2)
    ax.set_xlabel('Subset size (n)', fontsize=12)
    ax.set_ylabel(f'Estimated adiabatic time T ~ {prefactor}/$\\Delta^2$ (log2 scale)', fontsize=12)
    ax.set_title(title or 'Adiabatic time scaling: fit $2^{\\alpha n} + B$', fontsize=14)
    ax.grid(True, which='both', alpha=0.2)
    txt = f"$\\alpha$={alpha_fit:.4g}\nB={B_fit:.3g}\n$R^2$={r2:.3f}"
    ax.annotate(txt, xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left',
                fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.legend()
    plt.tight_layout()
    return fig, ax, fit_result
