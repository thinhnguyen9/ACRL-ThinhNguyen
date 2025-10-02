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
from models.quadrotors import Quadrotor1
from src.simulator import Simulator

def main(
        enabled_estimators,
        v_means,    # Measurement noise (gaussian)
        v_stds,     # max error ~ 3 std.dev.
        w_means,    # Process noise (gaussian)
        w_stds,     # max error ~ 3 std.dev.
        X0,         # initial estmate (unused in this simulation)
        P0,         # initial covariance
        Q=None,     # weighting matrix, override if needed
        R=None,     # weighting matrix, override if needed
        T=1.0,
        ts=0.01,
        loops=1,
        mhe_horizon=10,
        mhe_update="filtering",
        prior_method="ekf",
        save_csv=False,
        enable_plot=False
    ):
    
    # ----------------------- Quadrotor -----------------------
    drone = Quadrotor1(
        l   = .2,
        m   = .8,
        Jx  = .05,
        Jy  = .05,
        Jz  = .07,
        kf  = [0.0, 1.6, 4.0],
        Kdt = [1.99049588e-04, 3.50416237e-02, -1.40556979e-03, 2.61568995e-04, -1.86983501e-05],
        tc  = .1,
        Kdx = 1.,
        Kdy = 1.,
        Kdz = 1.
    )
    drone_est = copy.deepcopy(drone)
    # drone_est = Quadrotor1(     # used for estimation
    #     l   = .2,
    #     m   = .8 * 1.25,
    #     Jx  = .05 * 1.25,
    #     Jy  = .05 * 1.25,
    #     Jz  = .07 * 1.25,
    #     kf  = [0.0, 1.6, 4.0],
    #     Kdt = [1.99049588e-04, 3.50416237e-02, -1.40556979e-03, 2.61568995e-04, -1.86983501e-05],
    #     tc  = .1,
    #     Kdx = 1.,
    #     Kdy = 1.,
    #     Kdz = 1.
    # )
    xhover_est = np.zeros(drone_est.Nx)
    xhover_est[12:16] = drone_est.f_h
    uhover_est = np.array([drone_est.u_h]*drone_est.Nu)

    # ----------------------- Simulation -----------------------
    if Q is None:   Q = np.diag(w_stds**2)
    if R is None:   R = np.diag(v_stds**2)

    sim = Simulator(
        mode = 'quadrotor',
        sys = drone,
        w_means = w_means,
        w_stds = w_stds,
        v_means = v_means,
        v_stds = v_stds,
        T = T,
        ts = ts,
        # noise_distribution = 'uniform'
    )

    # ----------------------- Run estimation -----------------------
    for loop in range(loops):
        print("============ Simulation instance " + str(loop+1) + " of " + str(loops) + " ============")
        # Initialize estimators - must be done every loop
        if 'KF' in enabled_estimators:
            SKF = KF(
                model = drone_est,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                P0 = P0,
                type = "standard",
                xs = xhover_est,
                us = uhover_est
            )
        if 'EKF' in enabled_estimators:
            EKF = KF(
                model = drone_est,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                P0 = P0,
                type = "extended"
            )
        if 'LMHE1' in enabled_estimators:
            LMHE_once = MHE(
                model = drone_est,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                N = mhe_horizon,
                X0 = xhover_est,
                P0 = P0,
                mhe_type = "linearized_once",
                mhe_update = mhe_update,
                prior_method = prior_method,
                xs = xhover_est,
                us = uhover_est
            )
        if 'LMHE2' in enabled_estimators:
            LMHE_every = MHE(
                model = drone_est,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                N = mhe_horizon,
                X0 = xhover_est,
                P0 = P0,
                mhe_type = "linearized_every",
                mhe_update = mhe_update,
                prior_method = prior_method,
                xs = xhover_est,
                us = uhover_est
            )
        if 'NMHE' in enabled_estimators:
            NMHE = MHE(
                model = drone_est,
                ts = sim.get_time_step(),
                Q = Q,
                R = R,
                N = mhe_horizon,
                X0 = xhover_est,
                P0 = P0,
                mhe_type = "nonlinear",
                mhe_update = mhe_update,
                prior_method = prior_method,
                xs = xhover_est,
                us = uhover_est
            )

        tvec, xvec, uvec, yvec = sim.simulate_point_to_point_LQR(
                # x0 = copy.deepcopy(xhover_est),
                # x0 = np.array([ 0., 0., 0., 0., -1., 0.,
                #                 .35, -.35, 0.,
                #                 0., 0., 0.,
                #                 drone.f_h*.8, drone.f_h*.9, drone.f_h*1.2, drone.f_h*1.1 ]),
                zero_disturbance = False,
                zero_noise = False
        )
        N = len(tvec)
        if 'KF' in enabled_estimators:
            xhat_kf, kf_time = sim.run_estimation(SKF, xhover_est)
            rmse_kf = np.sqrt(np.mean((xvec - xhat_kf)**2))
        if 'EKF' in enabled_estimators:
            xhat_ekf, ekf_time = sim.run_estimation(EKF, xhover_est)
            rmse_ekf = np.sqrt(np.mean((xvec - xhat_ekf)**2))
        if 'LMHE1' in enabled_estimators:
            xhat_lmhe1, lmhe1_time = sim.run_estimation(LMHE_once, xhover_est)
            rmse_lmhe1 = np.sqrt(np.mean((xvec - xhat_lmhe1)**2))
        if 'LMHE2' in enabled_estimators:
            xhat_lmhe2, lmhe2_time = sim.run_estimation(LMHE_every, xhover_est)
            rmse_lmhe2 = np.sqrt(np.mean((xvec - xhat_lmhe2)**2))
        if 'NMHE' in enabled_estimators:
            xhat_nmhe, nmhe_time = sim.run_estimation(NMHE, xhover_est)
            rmse_nmhe = np.sqrt(np.mean((xvec - xhat_nmhe)**2))

        print("\t   RMSE\t\tAvg. step time (ms)")
        if 'KF' in enabled_estimators:      print(f"KF\t: {rmse_kf:.4f}\t\t{kf_time*1000./N:.4f}")
        if 'EKF' in enabled_estimators:     print(f"EKF\t: {rmse_ekf:.4f}\t\t{ekf_time*1000./N:.4f}")
        if 'LMHE1' in enabled_estimators:   print(f"LMHE1\t: {rmse_lmhe1:.4f}\t\t{lmhe1_time*1000./N:.4f}")
        if 'LMHE2' in enabled_estimators:   print(f"LMHE2\t: {rmse_lmhe2:.4f}\t\t{lmhe2_time*1000./N:.4f}")
        if 'NMHE' in enabled_estimators:    print(f"NMHE\t: {rmse_nmhe:.4f}\t\t{nmhe_time*1000./N:.4f}")
        if 'EKF' in enabled_estimators and 'LMHE2' in enabled_estimators:
            print("----------------------------")
            print(f"LMHE2-EKF RMSE: {np.sqrt(np.mean((xhat_lmhe2 - xhat_ekf)**2)):.4f}")

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
            with open('simulation/quadrotor/sim_instances.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)
    print("======================================================")
    # with open('simulation/quadrotor/sim_data.npy', 'wb') as f:
    #     np.save(f, tvec)
    #     np.save(f, xvec)
    #     np.save(f, xhat)
    #     np.save(f, yvec)
    #     np.save(f, uvec)
    # print("Simulation data saved to 'sim_data.npy'")

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
            plt.plot(tvec, xvec[:,idx]*(180/np.pi if rad2deg else 1), 'k-', lw=3., label=title_prefix+'_true')
            if idx < drone.Ny:
                plt.plot(tvec, yvec[:,idx]*(180/np.pi if rad2deg else 1), 'k:', lw=0.5, label=title_prefix+'_meas')
            if 'KF' in enabled_estimators:
                plt.plot(tvec, xhat_kf[:,idx]*(180/np.pi if rad2deg else 1), 'r-', lw=1., label='KF')
            if 'EKF' in enabled_estimators:
                plt.plot(tvec, xhat_ekf[:,idx]*(180/np.pi if rad2deg else 1), 'b-', lw=1., label='EKF')
            if 'LMHE1' in enabled_estimators:
                plt.plot(tvec, xhat_lmhe1[:,idx]*(180/np.pi if rad2deg else 1), 'm-', lw=1., label='LMHE1')
            if 'LMHE2' in enabled_estimators:
                plt.plot(tvec, xhat_lmhe2[:,idx]*(180/np.pi if rad2deg else 1), 'c-', lw=1., label='LMHE2')
            if 'NMHE' in enabled_estimators:
                plt.plot(tvec, xhat_nmhe[:,idx]*(180/np.pi if rad2deg else 1), 'y-', lw=1., label='NMHE')
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

        plt.subplot(331)
        plot_state(0, ylabel='Position - world (m)', title_prefix='x')
        plt.subplot(332)
        plot_state(2, title_prefix='y')
        plt.subplot(333)
        plot_state(4, invert_y=True, title_prefix='z')
        plt.subplot(334)
        plot_state(1, ylabel='Velocity - world (m/s)', title_prefix='xd')
        plt.subplot(335)
        plot_state(3, title_prefix='yd')
        plt.subplot(336)
        plot_state(5, invert_y=True, title_prefix='zd')
        plt.subplot(337)
        plot_state(6, ylabel='Angles - world (deg)', rad2deg=True, title_prefix='roll')
        plt.xlabel('Time (s)')
        plt.subplot(338)
        plot_state(7, rad2deg=True, title_prefix='pitch')
        plt.xlabel('Time (s)')
        plt.subplot(339)
        plot_state(8, rad2deg=True, title_prefix='yaw')
        plt.xlabel('Time (s)')

        plt.figure(2)
        plt.suptitle('Estimators comparison (unmeasured states)')
        plt.subplot(331)
        plot_state(9, ylabel='Angular rates - body (rad/s)', title_prefix='p')
        plt.subplot(332)
        plot_state(10, title_prefix='q')
        plt.subplot(333)
        plot_state(11, title_prefix='r')
        plt.subplot(334)
        plot_state(12, ylabel='Motor forces (N)', title_prefix='f1')
        plt.subplot(335)
        plot_state(13, title_prefix='f2')
        plt.xlabel('Time (s)')
        plt.subplot(336)
        plot_state(14, title_prefix='f3')
        plt.xlabel('Time (s)')
        plt.subplot(337)
        plot_state(15, ylabel='Motor forces (N)', title_prefix='f4')
        plt.xlabel('Time (s)')
        plt.subplot(338)
        plt.axis('off')
        plt.subplot(339)
        plt.axis('off')
        plt.xlabel('Time (s)')
        # plt.tight_layout()

        def plot_error(idx, ylabel=None, title_prefix=''):
            if 'KF' in enabled_estimators:
                plt.plot(tvec, xvec[:,idx]-xhat_kf[:,idx], 'r-', lw=1., label='KF')
            if 'EKF' in enabled_estimators:
                plt.plot(tvec, xvec[:,idx]-xhat_ekf[:,idx], 'b-', lw=1., label='EKF')
            if 'LMHE1' in enabled_estimators:
                plt.plot(tvec, xvec[:,idx]-xhat_lmhe1[:,idx], 'm-', lw=1., label='LMHE1')
            if 'LMHE2' in enabled_estimators:
                plt.plot(tvec, xvec[:,idx]-xhat_lmhe2[:,idx], 'c-', lw=1., label='LMHE2')
            if 'NMHE' in enabled_estimators:
                plt.plot(tvec, xvec[:,idx]-xhat_nmhe[:,idx], 'y-', lw=1., label='NMHE')
            if 'EKF' in enabled_estimators and 'LMHE2' in enabled_estimators:
                plt.plot(tvec, xhat_lmhe2[:,idx]-xhat_ekf[:,idx], 'k--', lw=1., label='LMHE2-EKF')
            plt.grid()
            leg = plt.legend()
            leg.set_draggable(True)
            if ylabel: plt.ylabel(ylabel)
            plt.title(title_prefix, fontsize=10)

        plt.figure(3)
        plt.suptitle('Estimation error (by state)')
        for idx in range(drone.Nx):
            plt.subplot(4, 4, idx+1)
            plot_error(idx)

        if 'EKF' in enabled_estimators and 'LMHE2' in enabled_estimators:
            plt.figure(4)
            plt.suptitle('Estimation error')
            err_ekf   = np.linalg.norm(xvec - xhat_ekf, axis=1)
            err_lmhe2 = np.linalg.norm(xvec - xhat_lmhe2, axis=1)

            plt.subplot(211)
            plt.plot(tvec, err_ekf, 'r-', lw=1., label='EKF')
            plt.plot(tvec, err_lmhe2, 'b-', lw=1., label='LMHE2')
            plt.grid()
            plt.ylabel(r'$\|x - \hat{x}\|$', fontsize=10)
            plt.legend()

            plt.subplot(212)
            plt.plot(tvec, err_lmhe2-err_ekf, 'k--', lw=1., label='LMHE2-EKF')
            plt.grid()
            plt.xlabel('Time (s)')
            plt.ylabel('LMHE2 - EKF')

        plt.show()


if __name__ == "__main__":
    """
    Estimators to simulate: 'KF', 'EKF',
                            'LMHE1' (linearized once),'LMHE2' (linearized every step),
                            'NMHE' (using nonlinear dynamics)
    """
    main(
        enabled_estimators=['EKF', 'LMHE2'],
        v_means=np.zeros(9),
        # v_means = np.array([.1, 0., .1, 0., .1, 0., -.1, -.1, -.1]),
        v_stds=np.array([.02, .5, .02, .5, .02, .5, .03, .03, .03])/3,
        w_means=np.zeros(16),
        # w_means = np.array([.1, -.5, .1, -.5, .1, -.5,
        #                     0., 0., 0., 0., 0., 0.,
        #                     0., 0., 0., 0.]),
        w_stds=np.array([.1, 2., .1, 2., .1, 2.,
                         .1, .1, .1, .5, .5, .5,
                         10., 10., 10., 10. ])/3,
        X0=None,
        P0=np.eye(16) * 1e0,
        # Q=np.diag([ .1, 2., .1, 2., .1, 2.,
        #             .1, .1, .1, .5, .5, .5,
        #             10., 10., 10., 10. ])**.5,
        # R=np.diag([.02, .5, .02, .5, .02, .5, .03, .03, .03])**.5,
        T=1.,
        # ts=0.001,
        # loops=5,
        # mhe_horizon  = 30,
        mhe_update   = "filtering",     # "filtering", "smoothing"
        prior_method = "ekf",           # "zero", "uniform", "ekf"
        # save_csv=True,
        enable_plot=True
    )
