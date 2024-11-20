"""
Dedalus script simulating the 1D Heat equation.
This script demonstrates solving a 1D initial value problem and produces
a space-time plot of the solution. It should take just a few seconds to
run (serial only).

We use a Fourier basis to solve the IVP:
    dt(u) = mu * dx(dx(u))

To run and plot:
    $ python3 kdv_burgers.py
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import h5py

logger = logging.getLogger(__name__)


# Parameters
Lx = 6
Nx = 1024
mu = 1e-2
dealias = 3 / 2
stop_sim_time = 16
timestepper = d3.SBDF2
timestep = 1e-3
dtype = np.float64

n = 20
np.random.seed = 42
li = [1, 2, 3]


times, solns = [], []

for tic_idx in range(500):
    print(tic_idx)
    # Bases
    xcoord = d3.Coordinate("x")
    dist = d3.Distributor(xcoord, dtype=dtype)
    xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

    # Fields
    u = dist.Field(name="u", bases=xbasis)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Problem
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) - mu*dx(dx(u)) = 0")

    # Initial conditions
    x = dist.local_grid(xbasis)
    print(x)

    ic = np.zeros(len(x))
    for i in range(10):
        A = np.random.random() - 0.5
        idx = np.random.randint(3)
        phi = np.random.random() * 2 * np.pi
        test = A * np.sin(2 * np.pi * li[idx] * x / Lx + phi)
        ic += A * np.sin(2 * np.pi * li[idx] * x / Lx + phi)

    u["g"] = ic

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Main loop
    u.change_scales(1)
    u_list = [np.copy(u["g"])]
    t_list = [solver.sim_time]
    while solver.proceed:
        solver.step(timestep)
        # if solver.iteration % 100 == 0:
        #     logger.info(
        #         "Iteration=%i, Time=%e, dt=%e"
        #         % (solver.iteration, solver.sim_time, timestep)
        #     )
        if solver.iteration % 25 == 0:
            u.change_scales(1)
            u_list.append(np.copy(u["g"]))
            t_list.append(solver.sim_time)

    solns.append(np.array(u_list))

f = h5py.File("dataset/heat_new.hdf5", "w")
f.create_dataset("time", data=t_list)
f.create_dataset("x", data=x)
f.create_dataset("soln", data=solns)


# # Plot
# plt.figure(figsize=(6, 4))
# plt.pcolormesh(
#     x.ravel(),
#     np.array(t_list),
#     np.array(u_list),
#     cmap="RdBu_r",
#     shading="gouraud",
#     rasterized=True,
#     clim=(-0.8, 0.8),
# )
# plt.xlim(0, Lx)
# plt.ylim(0, stop_sim_time)
# plt.xlabel("x")
# plt.ylabel("t")
# plt.title(f"Heat equation")
# plt.tight_layout()
# plt.savefig("test.pdf")
