import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Use LaTeX-style mathtext and serif fonts for prettier rendering (no external LaTeX required)
# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "serif",
#     "mathtext.fontset": "stix",                # STIX math fonts (good for publications)
#     "font.serif": ["Times New Roman", "STIXGeneral"],
#     "font.size": 12,
#     "axes.titlesize": 14,
#     "axes.labelsize": 12,
#     "legend.fontsize": 10,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "figure.dpi": 150
# })
plt.rcParams.update({
    "text.usetex": True,                       # require TeX, dvipng/ghostscript installed
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],   # LaTeX default (Computer Modern)
    "mathtext.fontset": "cm",                  # math rendering to Computer Modern
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150
})

def load_and_aggregate_errors(csv_file, estimator=None):
    """
    Load estimation_error.csv and aggregate errors by time step.
    
    Args:
        csv_file: path to estimation_error.csv
        estimator: filter by estimator name (e.g. 'EKF'). If None, process all.
    
    Returns:
        t_unique: sorted unique time points
        mean_err: mean error at each time point
        std_err: std error at each time point
        n_runs: number of simulation instances
    """
    df = pd.read_csv(csv_file)
    
    # Filter by estimator if specified
    if estimator:
        df = df[df['estimator'] == estimator]
    
    # Group by time and calculate mean/std across all runs
    grouped = df.groupby('time')['estimation_error_norm'].agg(['mean', 'std', 'count'])
    
    t_unique = grouped.index.values
    mean_err = grouped['mean'].values
    std_err = grouped['std'].values
    n_runs = grouped['count'].values[0]
    
    return t_unique, mean_err, std_err, n_runs

if __name__ == "__main__":
    
    # ===================================================================================================== #
    csv_path = os.path.join(os.path.dirname(__file__), "2025-12-04_scenario1_estimation_error.csv")
    # ===================================================================================================== #

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Estimators to plot
    # estimators = ['LMHE2', 'EKF', 'LMHE1', 'LMHE3'] # sort highest performing last for visibility
    estimators = ['EKF', 'LMHE1', 'LMHE3'] # sort highest performing last for visibility
    
    # ===================================================================================================== #
    colors = {
        'KF': 'r',
        'EKF': 'tab:red',
        'LMHE1': 'tab:blue', 
        'LMHE2': 'tab:orange',
        'LMHE3': 'tab:green',
        'NMHE': 'tab:brown'
    }
    # markers = {
    #     'KF': '.',
    #     'EKF': 'o',
    #     'LMHE1': '^', 
    #     'LMHE2': 's',
    #     'LMHE3': 'x',
    #     'NMHE': 'v'
    # }
    markers = {
        'KF': None,
        'EKF': None,
        'LMHE1': None, 
        'LMHE2': None,
        'LMHE3': None,
        'NMHE': None
    }
    plot_labels = {
        'KF': 'KF',
        'EKF': 'EKF',
        'LMHE1': 'MHE (OSQP)', 
        'LMHE2': 'MHE (PCM)',
        'LMHE3': 'MHE ($\mathcal{L}_1$-AO)',
        'NMHE': 'NMHE'
    }
    plt.figure(figsize=(9,5))
    n_stddev = 1.0
    for est in estimators:
        try:
            t, mean_err, std_err, n_runs = load_and_aggregate_errors(csv_path, estimator=est)
            plt.plot(
                t, mean_err, color=colors.get(est, 'k'), lw=2.0,
                label=plot_labels.get(est, ''),
                marker=markers.get(est, '.'), markersize=5, markevery=2
            )
            plt.fill_between(t, mean_err - n_stddev*std_err, mean_err + n_stddev*std_err, 
                           color=colors.get(est, 'k'), alpha=0.25)
        except Exception as e:
            print(f"Skipping {est}: {e}")
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(r'$\|x-\hat{x}\|$', fontsize=12)
    # plt.title(rf'Mean ± Stddev estimation error over {n_runs} runs', fontsize=14)

    # reorder legend by custom order
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [
        labels.index(plot_labels.get('EKF')),
        labels.index(plot_labels.get('LMHE1')),
        # labels.index(plot_labels.get('LMHE2')),
        labels.index(plot_labels.get('LMHE3'))
    ]
    plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=10)

    plt.grid(True, which='both', color='k', ls='--', lw=0.5, alpha=0.4)
    # plt.axis('square')
    # plt.xlim((t[0]-.005, .2))
    # plt.ylim((0., 10.5))

    plt.xlim((t[0]-.05, t[-1]))
    plt.yscale('log')

    # -------------------- inset: zoom first 0.2 s --------------------
    ax = plt.gca()
    # Use t from the previous loop (fallback if not defined)
    try:
        t0 = float(t[0])
    except Exception:
        t0 = 0.0
    zoom_width = 0.5
    zoom_t0 = t0
    zoom_t1 = zoom_t0 + zoom_width

    # Create inset axes (x, y, width, height in axes fraction coords)
    axins = ax.inset_axes([0.3, 0.52, 0.4, 0.4])
    for est in estimators:
        try:
            t_est, mean_err, std_err, _ = load_and_aggregate_errors(csv_path, estimator=est)
        except Exception:
            continue
        # select indices in zoom window
        mask = (t_est >= zoom_t0) & (t_est <= zoom_t1)
        if not np.any(mask):
            continue
        c = colors.get(est, 'k')
        axins.plot(t_est[mask], mean_err[mask], color=c, lw=1.6, label=plot_labels.get(est, est))
        axins.fill_between(t_est[mask], mean_err[mask] - std_err[mask], mean_err[mask] + std_err[mask],
                           color=c, alpha=0.25)
    axins.set_xlim(zoom_t0, zoom_t1)
    # y-limits with small padding
    yvals_masked = []
    for est in estimators:
        try:
            t_est, mean_err, std_err, _ = load_and_aggregate_errors(csv_path, estimator=est)
        except Exception:
            continue
        mask = (t_est >= zoom_t0) & (t_est <= zoom_t1)
        if np.any(mask):
            yvals_masked.append(mean_err[mask])
    if yvals_masked:
        yall = np.concatenate(yvals_masked)
        ymin, ymax = yall.min(), yall.max()
        pad = max(1e-6, 0.05*(ymax - ymin) if (ymax - ymin) > 0 else 0.1*ymax)
        axins.set_ylim(max(0.0, ymin - pad), ymax + pad)
    axins.grid(True, linestyle='--', linewidth=0.4)
    axins.tick_params(axis='both', which='major', labelsize=8)
    # indicate zoom rectangle on main axes (best-effort)
    try:
        ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.6)
    except Exception:
        # fallback: draw simple rectangle
        rect = plt.Rectangle((zoom_t0, ax.get_ylim()[0]), zoom_width, ax.get_ylim()[1]-ax.get_ylim()[0],
                             linewidth=0.8, edgecolor='black', facecolor='none', linestyle='--', alpha=0.4)
        ax.add_patch(rect)
    axins.set_yscale('log')
    # ------------------ end inset --------------------

    # ===================================================================================================== #
    """
    linestyles = {
        'EKF': 'dotted',
        'LMHE1': 'solid', 
        'LMHE2': 'dashed',
        'LMHE3': 'dashdot'
    }
    plt.figure(figsize=(5,4))
    for est in estimators:
        t, mean_err, std_err, n_runs = load_and_aggregate_errors(csv_path, estimator=est)
        plt.plot(
            t, mean_err, color='k', lw=1.2, ls=linestyles.get(est, None),
            label=plot_labels.get(est, '')
        )
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(r'$\|x-\hat{x}\|$', fontsize=12)
    # plt.title(rf'Mean ± Stddev estimation error over {n_runs} runs', fontsize=14)

    # reorder legend by custom order
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [
        labels.index(plot_labels.get('EKF')),
        labels.index(plot_labels.get('LMHE1')),
        # labels.index(plot_labels.get('LMHE2')),
        labels.index(plot_labels.get('LMHE3'))
    ]
    plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=10)

    plt.grid(True, which='both', color='k', ls='--', lw=0.5, alpha=0.4)
    # plt.xlim((t[0]-.1, t[-1]))
    # plt.yscale('log')

    plt.xlim((t[0]-.005, .2))
    plt.ylim((0., 10.5))
    """
    # ===================================================================================================== #
    plt.show()
