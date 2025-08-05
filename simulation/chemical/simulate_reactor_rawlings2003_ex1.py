import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import pathlib
from math import sin, cos
import cvxpy as cp
import copy
import time
import csv
from datetime import datetime

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from src.estimators import KF, MHE
from models.chemical import Reactor1
from src.simulator import Simulator

def main(enabled_estimators, T=5.0, ts=0.01, loops=1, mhe_horizon=10, save_csv=False, enable_plot=False):
    
    # ----------------------- Reactor -----------------------
    reactor = Reactor1()

    # ----------------------- Simulation -----------------------
    v_means = np.zeros(reactor.Ny)
    v_stds = np.array([.1])
    w_means = np.zeros(reactor.Nx)
    w_stds = np.array([1e-3, 1e-3])
    Q = np.diag(w_stds**2)
    R = np.diag(v_stds**2)
    # Q = np.diag([1e-4, 1e-2])
    # R = np.diag([1e-2])
    sim = Simulator(
        mode = 'reactor',
        sys = reactor,
        w_means = w_means,
        w_stds = w_stds,
        v_means = v_means,
        v_stds = v_stds,
        T = T,
        ts = ts
    )

    # ----------------------- Run estimation -----------------------
    P0 = np.eye(reactor.Nx) * (6**2)
    X0 = np.array([0.1, 4.5])
    for loop in range(loops):
        print("============ Simulation instance " + str(loop+1) + " of " + str(loops) + " ============")
        # Initialize estimators - must be done every loop
        if 'KF' in enabled_estimators:
            SKF = KF(
                model = reactor,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                P0 = P0,
                type = "standard",
                xs = np.zeros(reactor.Nx),
                us = None
            )
        if 'EKF' in enabled_estimators:
            EKF = KF(
                model = reactor,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                P0 = P0,
                type = "extended"
            )
        if 'LMHE1' in enabled_estimators:
            LMHE_once = MHE(
                model = reactor,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                N = mhe_horizon,
                X0 = X0,
                P0 = P0,
                mhe_type = "linearized_once",
                mhe_update = "filtering",
                xs = np.array([0.3, 2.4]),
                us = None
            )
        if 'LMHE2' in enabled_estimators:
            LMHE_every = MHE(
                model = reactor,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                N = mhe_horizon,
                X0 = X0,
                P0 = P0,
                mhe_type = "linearized_every",
                mhe_update = "filtering",
                xs = np.array([0.3, 2.4]),
                us = None
            )
        if 'NMHE' in enabled_estimators:
            NMHE = MHE(
                model = reactor,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                N = mhe_horizon,
                X0 = X0,
                P0 = P0,
                mhe_type = "nonlinear",
                mhe_update = "filtering",
                xs = np.array([0.3, 2.4]),
                us = None
            )

        tvec, xvec, uvec, yvec = sim.simulate_free_response(
                x0 = np.array([3., 1.]),
                # zero_disturbance = True,
                # zero_noise = True
        )
        N = len(tvec)
        if 'KF' in enabled_estimators:
            xhat_kf, kf_time = sim.run_estimation(SKF, X0)
            rmse_kf = np.sqrt(np.mean((xvec - xhat_kf)**2))
        if 'EKF' in enabled_estimators:
            xhat_ekf, ekf_time = sim.run_estimation(EKF, X0)
            rmse_ekf = np.sqrt(np.mean((xvec - xhat_ekf)**2))
        if 'LMHE1' in enabled_estimators:
            xhat_lmhe1, lmhe1_time = sim.run_estimation(LMHE_once, X0)
            rmse_lmhe1 = np.sqrt(np.mean((xvec - xhat_lmhe1)**2))
        if 'LMHE2' in enabled_estimators:
            xhat_lmhe2, lmhe2_time = sim.run_estimation(LMHE_every, X0)
            rmse_lmhe2 = np.sqrt(np.mean((xvec - xhat_lmhe2)**2))
        if 'NMHE' in enabled_estimators:
            xhat_nmhe, nmhe_time = sim.run_estimation(NMHE, X0)
            rmse_nmhe = np.sqrt(np.mean((xvec - xhat_nmhe)**2)) 

        print("\t   RMSE\t\tAvg. step time (ms)")
        if 'KF' in enabled_estimators:      print(f"KF\t: {rmse_kf:.4f}\t\t{kf_time*1000./N:.4f}")
        if 'EKF' in enabled_estimators:     print(f"EKF\t: {rmse_ekf:.4f}\t\t{ekf_time*1000./N:.4f}")
        if 'LMHE1' in enabled_estimators:   print(f"LMHE1\t: {rmse_lmhe1:.4f}\t\t{lmhe1_time*1000./N:.4f}")
        if 'LMHE2' in enabled_estimators:   print(f"LMHE2\t: {rmse_lmhe2:.4f}\t\t{lmhe2_time*1000./N:.4f}")
        if 'NMHE' in enabled_estimators:    print(f"NMHE\t: {rmse_nmhe:.4f}\t\t{nmhe_time*1000./N:.4f}")

        # Save results of this instance
        if save_csv:
            data = []
            date_str = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            # date,estimator,RMSE,max_err,computation_time_per_step,T,ts
            if 'KF' in enabled_estimators:      data.append([date_str, 'KF',    rmse_kf,    np.max(np.abs(xvec-xhat_kf)),    kf_time*1000./N,    T, ts])
            if 'EKF' in enabled_estimators:     data.append([date_str, 'EKF',   rmse_ekf,   np.max(np.abs(xvec-xhat_ekf)),   ekf_time*1000./N,   T, ts])
            if 'LMHE1' in enabled_estimators:   data.append([date_str, 'LMHE1', rmse_lmhe1, np.max(np.abs(xvec-xhat_lmhe1)), lmhe1_time*1000./N, T, ts])
            if 'LMHE2' in enabled_estimators:   data.append([date_str, 'LMHE2', rmse_lmhe2, np.max(np.abs(xvec-xhat_lmhe2)), lmhe2_time*1000./N, T, ts])
            if 'NMHE' in enabled_estimators:    data.append([date_str, 'NMHE',  rmse_nmhe,  np.max(np.abs(xvec-xhat_nmhe)),  nmhe_time*1000./N,  T, ts])
            with open('quadrotor-simulator/sim_instances.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)
    print("======================================================")

    # ----------------------- Plot results -----------------------
    if enable_plot:
        plt.figure(1)
        plt.suptitle('Estimators comparison (measured states)')
        if 'KF' in enabled_estimators:      rmse_kf     = np.sqrt(np.mean((xvec - xhat_kf)**2, axis=0))
        if 'EKF' in enabled_estimators:     rmse_ekf    = np.sqrt(np.mean((xvec - xhat_ekf)**2, axis=0))
        if 'LMHE1' in enabled_estimators:   rmse_lmhe1  = np.sqrt(np.mean((xvec - xhat_lmhe1)**2, axis=0))
        if 'LMHE2' in enabled_estimators:   rmse_lmhe2  = np.sqrt(np.mean((xvec - xhat_lmhe2)**2, axis=0))
        if 'NMHE' in enabled_estimators:    rmse_nmhe   = np.sqrt(np.mean((xvec - xhat_nmhe)**2, axis=0))

        def plot_state(idx, ylabel=None, invert_y=False, rad2deg=False, title_prefix=''):
            plt.plot(tvec, xvec[:,idx]*(180/np.pi if rad2deg else 1), 'k--', lw=2., label=title_prefix+'_true')
            # if idx < reactor.Ny:
            #     plt.plot(tvec, yvec[:,idx]*(180/np.pi if rad2deg else 1), 'k:', lw=0.5, label=title_prefix+'_meas')
            if 'KF' in enabled_estimators:
                plt.plot(tvec, xhat_kf[:,idx]*(180/np.pi if rad2deg else 1), 'y-', lw=1., label='KF')
            if 'EKF' in enabled_estimators:
                plt.plot(tvec, xhat_ekf[:,idx]*(180/np.pi if rad2deg else 1), 'r-', lw=1., label='EKF')
            if 'LMHE1' in enabled_estimators:
                plt.plot(tvec, xhat_lmhe1[:,idx]*(180/np.pi if rad2deg else 1), 'm-', lw=1., label='LMHE1')
            if 'LMHE2' in enabled_estimators:
                plt.plot(tvec, xhat_lmhe2[:,idx]*(180/np.pi if rad2deg else 1), 'c-', lw=1., label='LMHE2')
            if 'NMHE' in enabled_estimators:
                plt.plot(tvec, xhat_nmhe[:,idx]*(180/np.pi if rad2deg else 1), 'b-', lw=1., label='NMHE')
            plt.grid()
            leg = plt.legend()
            leg.set_draggable(True)
            if ylabel: plt.ylabel(ylabel)
            if invert_y: plt.gca().invert_yaxis()
            # Compose RMSE string only for enabled estimators
            rmse_str = []
            if 'KF' in enabled_estimators:      rmse_str.append(f"KF={rmse_kf[idx]*(180/np.pi if rad2deg else 1):.4f}")
            if 'EKF' in enabled_estimators:     rmse_str.append(f"EKF={rmse_ekf[idx]*(180/np.pi if rad2deg else 1):.4f}")
            if 'LMHE1' in enabled_estimators:   rmse_str.append(f"LMHE1={rmse_lmhe1[idx]*(180/np.pi if rad2deg else 1):.4f}")
            if 'LMHE2' in enabled_estimators:   rmse_str.append(f"LMHE2={rmse_lmhe2[idx]*(180/np.pi if rad2deg else 1):.4f}")
            if 'NMHE' in enabled_estimators:    rmse_str.append(f"NMHE={rmse_nmhe[idx]*(180/np.pi if rad2deg else 1):.4f}")
            plt.title(f"{title_prefix} RMSE: {', '.join(rmse_str)}", fontsize=10)

        plt.subplot(221)
        plot_state(0, ylabel='p_A', title_prefix='pA')
        # plt.ylim((-2, 4))
        plt.subplot(223)
        plot_state(1, ylabel='p_B', title_prefix='pB')
        # plt.ylim((0, 5))

        plt.show()


if __name__ == "__main__":
    main(
        enabled_estimators=['EKF', 'LMHE2', 'NMHE'],
        T=10.,
        ts=.1,
        loops=1,
        mhe_horizon=10,
        save_csv=False,
        enable_plot=True
    )
            # Estimators to simulate: 'KF', 'EKF',
            #                         'LMHE1' (linearized once),'LMHE2' (linearized every step),
            #                         'NMHE' (using nonlinear dynamics)
