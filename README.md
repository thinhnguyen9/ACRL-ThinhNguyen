# Moving Horizon Estimation with 𝓛₁ Adaptive Optimizer

- Paper: "Moving-Horizon State Estimation for Unmanned Aerial Vehicles: An L1 Adaptive Optimization Approach", AIAA SciTech 2026.
- Simulation examples (for quadrotors):
    - Quadrotor v1:
        - 16-dimensional, 9 directly measurable states, modeled nonlinear actuator dynamics.
        - Simulate: `py simulation\quadrotor\simulate_quadrotor.py`
        - Animate:  `py simulation\quadrotor\animate_quadrotor.py`
    - Quadrotor v2 (recommended):
        - 12-dimensional, 6 directly measureable states, direct thrust and torque inputs without constraints.
        - Simulate: `py simulation\quadrotor2\simulate_quadrotor.py`
        - Animate:  `py simulation\quadrotor2\animate_quadrotor.py`


## Project structure
- `models/` — dynamical systems used in simulation:
    - `basic.py` — base class `DynamicalSystem`.
    - `quadrotors.py` — `Quadrotor1`, `Quadrotor2`.
    - `chemical.py` — a chemical reactor as an example for constrained MHE.
- `src/` — all important implementations for MHE, PCIP, L1AO, etc.:
    - `kf.py` — Kalman filters (standard and extended) implementation.
    - `mhe.py` — Moving Horizon Estimator implementation:
        - Available solvers: CVXPY, PCIP, L1AO.
        - PCIP and L1AO tuning must be performed in this file (lines 68-81).
    - `pcip.py` — PCIP implementation for unconstrained Quadratic Programs.
    - `l1ao.py` — L1AO implementation for unconstrained Quadratic Programs.
    - `controllers.py` — basic LQR control for simulation.
    - `simulator.py` — simulation loop / runner utilities.
    - `utils.py` — functions to formulate MHE as Quadratic Programs:
        - Method 1 (inactive): dynamics constraints as equality constraints with Lagrange duality.
        - Method 2 (active): dynamics constraints incorporated into the objective function.
- `simulation/` — run simulation here:
    - `quadrotor2/`
        - `simulate_quadrotor.py`: call main() with desired simulation case (descriptions at the bottom).
        - `animate_quadrotor.py`: run to animate the simulation data saved to `sim_data.npy`.

