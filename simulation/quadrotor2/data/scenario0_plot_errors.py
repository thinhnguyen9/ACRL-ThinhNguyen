#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150
})

DEFAULT_LABEL_MAP = {
    "EKF": "EKF",
    "LMHE1": "MHE (OSQP)",
    "LMHE3": "MHE ($\mathcal{L}_1$-AO)"
}

DEFAULT_COLORS = {
    "EKF": "tab:red",
    "LMHE1": "tab:blue", 
    "LMHE3": "tab:green"
}

DEFAULT_MARKERS = {
    "EKF": "o",
    "MHE (OSQP N=10)": "^", 
    "MHE (OSQP N=100)": "s",
    "MHE (Newton+L1AO N=10)": "x",
    "MHE (Newton+L1AO N=100)": "v"
}

DEFAULT_LINES = {
    "EKF": "solid",
    "LMHE1": "solid", 
    "LMHE3": "solid"
}

DEFAULT_LINEWEIGHT = {
    "EKF": 1.5,
    "LMHE1": 1.5, 
    "LMHE3": 1.5
}

def plot_csv(
    csv_path,
    run_id_filter=None,
    estimators_filter=None,
    ylog=False,
    usemarker=False,
    markevery=20,
    markersize=6,
    figsize=(10,6)
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    expected = {'run_id','estimator','time','estimation_error_norm'}
    if not expected.issubset(set(df.columns)):
        raise RuntimeError(f"CSV missing required columns. Found: {df.columns.tolist()}")

    # Convert types
    df['time'] = df['time'].astype(float)
    df['estimation_error_norm'] = df['estimation_error_norm'].astype(float)

    # Apply filters if requested
    if run_id_filter is not None:
        df = df[df['run_id'].astype(str).isin(run_id_filter)]
        if df.empty:
            raise RuntimeError(f"No rows found for run_id='{run_id_filter}'")
    if estimators_filter is not None:
        df = df[df['estimator'].isin(estimators_filter)]
        if df.empty:
            raise RuntimeError(f"No rows found for estimators={estimators_filter}")
        
    # Preserve estimator order from filter (or use sorted if no filter)
    if estimators_filter is not None:
        estimators = [e for e in estimators_filter if e in df['estimator'].unique()]
    else:
        estimators = sorted(df['estimator'].unique())

    label_map = DEFAULT_LABEL_MAP.copy()

    plt.figure(figsize=figsize)
    ax = plt.gca()

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    for i, est in enumerate(estimators):
        display = label_map.get(est, est)
        sub = df[df['estimator'] == est]

        # Get style settings from defaults, fallback to rc colors/markers/lines
        color = DEFAULT_COLORS.get(est, (colors[i % len(colors)] if colors else None))
        marker = DEFAULT_MARKERS.get(est, None)
        linestyle = DEFAULT_LINES.get(est, '-')
        lineweight = DEFAULT_LINEWEIGHT.get(est, 1.0)

        # Plot each run (light lines). If run_id_filter was used there will typically be 1 run.
        run_ids = sorted(sub['run_id'].unique())
        for rid in run_ids:
            seq = sub[sub['run_id'] == rid].sort_values('time')
            # thin line per run
            # alpha = 1.0 if run_id_filter is not None else 0.35
            # lw = lineweight if run_id_filter is not None else 0.5
            label = display if rid == run_ids[0] else None

            ax.plot(
                seq['time'].values,
                seq['estimation_error_norm'].values,
                alpha=1.0,
                linewidth=lineweight,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker if usemarker else None,
                markevery=markevery,
                markersize=markersize
            )
    
    # =================================================================================
    # ---------- inset: zoom 1-second window ----------
    # Default: show first 1 second of the plot; change zoom_t0/window to adjust.
    zoom_t0 = 21.0          # start time for zoom (seconds)
    zoom_width = 2.0         # width of zoom window (seconds)
    zoom_t1 = zoom_t0 + zoom_width

    # Create inset axes in upper-right of main axes (box: x, y, width, height in axes coords)
    axins = ax.inset_axes([0.30, 0.05, 0.36, 0.36])

    # Plot each estimator's traces in the inset (same styles as main plot)
    for est in estimators:
        sub = df[df['estimator'] == est]
        color = DEFAULT_COLORS.get(est, None)
        marker = DEFAULT_MARKERS.get(est, None)
        linestyle = DEFAULT_LINES.get(est, '-')
        run_ids = sorted(sub['run_id'].unique())
        for rid in run_ids:
            seq = sub[sub['run_id'] == rid].sort_values('time')
            seq_zoom = seq[(seq['time'] >= zoom_t0) & (seq['time'] <= zoom_t1)]
            if seq_zoom.empty:
                continue
            axins.plot(
                seq_zoom['time'].values,
                seq_zoom['estimation_error_norm'].values,
                color=color,
                linestyle=linestyle,
                marker=marker if usemarker else None,
                markevery=markevery,
                markersize=markersize,
                linewidth=DEFAULT_LINEWEIGHT.get(est, 1.0),
                alpha=1.0
            )

    axins.set_xlim(zoom_t0, zoom_t1)
    # Auto-scale y-limits for inset with small padding
    yvals = df[(df['time'] >= zoom_t0) & (df['time'] <= zoom_t1)]['estimation_error_norm']
    if not yvals.empty:
        ymin, ymax = yvals.min(), yvals.max()
        pad = max(1e-6, 0.05*(ymax - ymin) if (ymax - ymin) > 0 else 0.1*ymax)
        axins.set_ylim(max(0.0, ymin - pad), ymax + pad)
    axins.grid(True, linestyle='--', linewidth=0.4)
    axins.set_xticks([zoom_t0, zoom_t1])
    axins.set_xticklabels([f"{zoom_t0:.2f}", f"{zoom_t1:.2f}"], fontsize=8)
    axins.tick_params(axis='y', labelsize=8)
    # axins.set_yscale('log')


    # Draw an indication (box) on the main axes that corresponds to the inset region
    try:
        ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.6)
    except Exception:
        # Fallback: draw a rectangle manually if `indicate_inset_zoom` is unavailable
        rect = plt.Rectangle((zoom_t0, ax.get_ylim()[0]), zoom_width, ax.get_ylim()[1] - ax.get_ylim()[0],
                             linewidth=0.8, edgecolor='black', facecolor='none', linestyle='--', alpha=0.4)
        ax.add_patch(rect)
    # =================================================================================

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\|x - \hat{x}\|$')
    ax.grid(which='both', linestyle='--', linewidth=0.4)
    ax.set_xlim((df['time'].min(), df['time'].max()))
    ax.set_ylim((0., df['estimation_error_norm'].max()*1.01))
    if ylog:
        ax.set_yscale('log')
        # avoid zero or negative values when log scale selected
        ymin = df['estimation_error_norm'][df['estimation_error_norm'] > 0].min()
        if pd.notna(ymin):
            ax.set_ylim(bottom=max(ymin*0.1, 1e-12))
    # Create a clean legend (avoid duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        leg = ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
        try:
            leg.set_draggable(True)
        except Exception:
            pass

    plt.tight_layout()
    plt.show()

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "2025-12-06_scenario0_estimation_error.csv")

    plot_csv(
        csv_path=csv_path,
        # run_id_filter=["33e0cf8b (Q1)", "677a5eaa (Q1)"],
        # estimators_filter=[
        #     "EKF", 
        #     "MHE (OSQP N=10)", 
        #     "MHE (Newton+L1AO N=10)", 
        #     "MHE (OSQP N=100)", 
        #     "MHE (Newton+L1AO N=100)"
        # ],
        # estimators_filter=["EKF", "LMHE1", "LMHE3"],
        ylog=True,
        # usemarker=True,
        markevery=100,
        markersize=3,
        figsize=(9,5)
    )

if __name__ == "__main__":
    main()