#!/usr/bin/env python3i
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from MPC.QP_solver import QPController
import spatialgeometry as sg
from MPC.LMPC_solver import LinearMPCController
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import square

# Init env
env = swift.Swift()
panda = rtb.models.DH.Panda()
panda.q = panda.qr
panda.taulim = [87, 87, 87, 87, 12, 12, 12]
panda.qdlim = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]) * 100

pandaShow = rtb.models.Panda()
pandaShow.q = panda.q
env.launch(realtime=True)
env.add(pandaShow)
dt = 0.05

# Init desired position
T_ini = panda.fkine(panda.q)
T_des = panda.fkine(panda.q)
target = sg.Sphere(radius=0.02, pose=T_des, color=[0,1,0])
env.add(target)
env.set_camera_pose([1.0, 1.0, 0.7], [0, 0, 0.4])

# Init LMPC solver for path planning, gamma the gain of the controller as it ensures slows commands u, the lower the faster
lmpc_solver = LinearMPCController(horizon=25, dt=dt, gamma = 0.00001,
                                    u_min=np.array([-2.0, -2.0, -2.0, -10.0, -10.0, -10.0]),
                                    u_max=np.array([ 2.0, 2.0, 2.0, 10.0, 10.0, 10.0]))

# Init QP solver for IK with safety
qp_solver = QPController(panda, dt=dt)

thetas = []
thetas_target = []
t_x, t_y, t_z = [], [], []
t_target_x, t_target_y, t_target_z = [], [], []
times = []

# Loop
while env.sim_time < 15:
    #Compute desired velocity from simple prop controller
    T_current = panda.fkine(panda.q)
    T_des.t[0] = T_ini.t[0] + 0.2 * np.sin(env.sim_time * 0.5)
    T_des.t[0] = T_ini.t[0] + square(env.sim_time * 0.5) * 0.2
    xdot = panda.jacobe(panda.q) @ panda.qd
    Uopt, Xopt, poses = lmpc_solver.solve(T_current, T_des, xdot)
    target.T = T_des

    #Solve QP
    tau_reg = 5 * (panda.qr - panda.q) - 0.5 * panda.qd # PD to rest position, careful tunning
    qp_solver.update_robot_state(panda)
    qp_solver.solve(Uopt[0:6], w_tau_reg=1.0, tau_reg=tau_reg, f_ext=None)
    tau = qp_solver.solution
    if tau is None:
        tau = panda.rne(panda.q, panda.qd, panda.qdd)
        print("QP failed, using gravity compensation")
    
    #Store data
    R_rel = T_ini.R @ T_current.R
    theta = R.from_matrix(R_rel).as_rotvec()[0]
    t = T_current.t - T_ini.t
    thetas.append(theta)
    t_x.append(t[0])
    t_y.append(t[1])
    t_z.append(t[2])
    times.append(env.sim_time)
    
    R_rel_target = T_ini.R @ T_des.R
    theta_target = R.from_matrix(R_rel_target).as_rotvec()[0]
    t_target = T_des.t - T_ini.t
    t_target_x.append(t_target[0])
    t_target_y.append(t_target[1])
    t_target_z.append(t_target[2])
    thetas_target.append(theta_target) 
    
    #Simulate
    panda.qdd = panda.accel(panda.q, panda.qd, tau)
    panda.qd += panda.qdd * dt
    panda.q += panda.qd * dt
    pandaShow.q = panda.q
    env.step(dt)
    
env.close()
# Set up dynamic plot
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel("Time")
ax.grid(True)

# Append to lists
ax.scatter(times, t_x, color='blue', label='X (m)', s=8)
ax.scatter(times, t_y, color='green', label='Y (m)', s=8)
ax.scatter(times, t_z, color='yellow', label='Z (m)', s=8)
ax.scatter(times, thetas, color='red', label='Rotation (rad)', s=8)

ax.plot(times, t_target_x, 'b--', label = 'X target (m)', markersize=4)
ax.plot(times, t_target_y, 'g--', markersize=4)
ax.plot(times, t_target_z, 'y--', markersize=4)
ax.plot(times, thetas_target, 'r--', markersize=4)
ax.legend()

plt.show()