import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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

if __name__ == "__main__":

    # ===================================================================================================== #
    csv_path = os.path.join(os.path.dirname(__file__), "2025-12-04_scenario1_estimation_error.csv")
    # ===================================================================================================== #

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Load raw estimation error data
    df = pd.read_csv(csv_path)
    
    # Calculate RMSE per estimator per run_id
    # RMSE = sqrt(mean(error_norm^2))
    rmse_per_run = df.groupby(['run_id', 'loop', 'estimator']).apply(
        lambda x: np.sqrt(np.mean(x['estimation_error_norm'] ** 2))
    ).reset_index(name='RMSE')
    
    # Extract unique estimators and sort for consistent order
    estimators = sorted(rmse_per_run['estimator'].unique())
    estimator_names = {
        'KF': 'KF',
        'EKF': 'EKF',
        'LMHE1': 'MHE (OSQP)', 
        'LMHE2': 'MHE (PCM)',
        'LMHE3': 'MHE ($\mathcal{L}_1$AO)',
        'NMHE': 'NMHE'
    }
    
    # Prepare data for box plot
    data_for_box = [rmse_per_run[rmse_per_run['estimator'] == est]['RMSE'].values for est in estimators]
    custom_labels = [estimator_names.get(est, est) for est in estimators]
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot(data_for_box, labels=custom_labels, patch_artist=True)
    
    # Customize box colors
    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']
    for patch, color in zip(bp['boxes'], colors[:len(estimators)]):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
    
    # Make median line black and thicker
    for median in bp['medians']:
        median.set(color='black', linewidth=1)
    
    # ax.set_xlabel('Estimator', fontsize=12)
    ax.set_ylabel(r'$\|x-\hat{x}\|_{RMS}$', fontsize=12)
    # ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))  # more gridlines
    ax.yaxis.set_major_locator(MultipleLocator(0.05)) # Set major ticks every 0.5 units
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Set 5 minor ticks between major ticks
    ax.grid(which='major', axis='y', color='black', ls='-', lw=0.5, alpha=0.7)
    ax.grid(which='minor', axis='y', color='black', ls='--', lw=0.25, alpha=0.4)
    ax.tick_params(axis='x', labelrotation=30)
    
    
    plt.tight_layout()
    plt.show()