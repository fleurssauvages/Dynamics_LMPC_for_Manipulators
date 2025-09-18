#!/usr/bin/env python3
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from MPC.QP_solver import QPController
import spatialgeometry as sg
from MPC.LMPC_solver import LinearMPCController
import time

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
dt = 0.05 # Ideally, should be less than 0.01, but QP solving is too slow on my computer, 0.05 is realtime but unstable when close to singularities

# Init desired position
T_ini = panda.fkine(panda.q)
T_des = panda.fkine(panda.q)
target = sg.Sphere(radius=0.02, pose=T_des, color=[0,1,0])
env.add(target)
env.set_camera_pose([1.0, 1.0, 0.7], [0, 0, 0.4])

# Init LMPC solver for path planning, gamma the gain of the controller as it ensures slows commands u, the lower the faster
lmpc_solver = LinearMPCController(horizon=25, dt=dt, gamma = 0.08,
                                    u_min=np.array([-2.0, -2.0, -2.0, -10.0, -10.0, -10.0]),
                                    u_max=np.array([ 2.0, 2.0, 2.0, 10.0, 10.0, 10.0]))

# Init QP solver for IK with safety
qp_solver = QPController(panda, dt=dt)

# Add sliders to control desired position
x, y, z = 0.0, 0.0, 0.0
def set_x(x_set):
    global T_des, target, x, y, z
    x, T_des = float(x_set), T_ini * sm.SE3.Trans(float(x_set), y, z)
    target.T = T_des
env.add(swift.Slider(lambda x: set_x(x),min=-0.4,max=0.4,step=0.01,desc="x",))
def set_y(y_set):
    global T_des, target, x, y, z
    y, T_des = float(y_set), T_ini * sm.SE3.Trans(x, float(y_set), z)
    target.T = T_des
env.add(swift.Slider(lambda x: set_y(x),min=-0.4,max=0.4,step=0.01,desc="y",))
def set_z(z_set):
    global T_des, target, x, y, z
    z, T_des = float(z_set), T_ini * sm.SE3.Trans(x, y, float(z_set))
    target.T = T_des
env.add(swift.Slider(lambda x: set_z(-x),min=-0.5,max=0.5,step=0.01,desc="z",))

# Loop
init_tau = panda.rne(panda.q, panda.qd, panda.qdd)
qp_solver.solution = init_tau
xdot = np.zeros(6)
while True:
    #Compute desired velocity from simple prop controller
    T_current = panda.fkine(panda.q)
    xdot = panda.jacobe(panda.q) @ panda.qd
    Uopt, Xopt, poses = lmpc_solver.solve(T_current, T_des, xdot)
    
    #Solve QP
    tau_reg = 5 * (panda.qr - panda.q) - 0.5 * panda.qd # PD to rest position, careful tunning
    qp_solver.update_robot_state(panda)
    qp_solver.solve(Uopt[0:6], w_tau_reg=1.0, tau_reg=tau_reg, f_ext=None)
    tau = qp_solver.solution
    if tau is None:
        tau = panda.rne(panda.q, panda.qd, panda.qdd)
        print("QP failed, using gravity compensation")
    
    #Simulate
    panda.qdd = panda.accel(panda.q, panda.qd, tau)
    panda.qd += panda.qdd * dt
    panda.q += panda.qd * dt
    pandaShow.q = panda.q
    env.step(dt)