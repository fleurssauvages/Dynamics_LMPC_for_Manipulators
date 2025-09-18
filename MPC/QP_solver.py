import numpy as np
from qpsolvers import solve_qp
import scipy.sparse as sp

class QPController:
    def __init__(self, robot, dt=0.05):
        self.n_dof = robot.n
        self.robot = robot
        self.joint_positions = robot.q
        self.joint_velocities = robot.qd
        self.joints_limits = robot.qlim + np.array([0.5, -0.5]) @ np.ones((2, robot.n))
        self.joints_velocities_limits = robot.qdlim
        self.joints_tau_limits = robot.taulim
        self.H = np.eye(robot.n)  # Hessian
        self.g = np.zeros(robot.n)  # Gradient
        self.A = np.zeros((0, robot.n))  # Inequality constraints
        self.b = np.zeros(0)  # Inequality constraint bounds
        self.eqA = np.zeros((0, robot.n))  # Equality constraints
        self.eqb = np.zeros(0) # Equality constraint bounds
        self.lb = -np.ones(robot.n) * np.inf  # Lower bounds
        self.ub = np.ones(robot.n) * np.inf   # Upper bounds
        self.dt = dt # Time step of the controller loop / simulation
        self.solution = None

    def solve(self, xdotdot, w_tau_reg=0.01, tau_reg=None, f_ext=None, W = np.diag([1.0, 1.0, 2.0, 0.1, 0.1, 0.1])):
        """
        Solve the quadratic programming problem using previous solution as initial value
        Minimize the cost function ||J qdot - xdot||^2 + alpha ||N qdot||^2 - beta * manipulability_gradient * qdot
        where N is the nullspace projector of J
        xdot the desired end-effector velocity (6D vector)
        alpha the weight on the secondary task (minimize joint velocities)
        beta the weight on maximizing manipulability
        The weight matrix W can be used to prioritize translation over rotation or vice-versa,
        if translations are prioritize, set higher values on the first 3 diagonal elements
        """
        self.update_IK_problem(self.joint_positions, self.joint_velocities, xdotdot, w_tau_reg=w_tau_reg, tau_reg=tau_reg, f_ext=f_ext, W = W)
        self.update_joints_limits(self.joint_positions, self.joint_velocities, f_ext=None)
        x = solve_qp(sp.csc_matrix(self.H), self.g, G=sp.csc_matrix(self.A), h=self.b, A=sp.csc_matrix(self.eqA), b=self.eqb, lb=self.lb, ub=self.ub, solver="osqp", initvals=self.solution)
        self.solution = x
        self.reset_constraints()
        pass
    
    def update_robot_state(self, robot):
        self.robot = robot
        self.joint_positions = robot.q
        self.joint_velocities = robot.qd
        pass
            
    def update_IK_problem(self, q, qdot, xdotdot, w_tau_reg=0.01, tau_reg=None, f_ext=None, W = np.diag([1.0, 1.0, 2.0, 0.1, 0.1, 0.1])):
        """
        Update the IK problem parameters based on desired end-effector velocity (6D vector) and current joint positions
        xdotdot: np.array of shape (6,)
        joint_pos and velocities: np.array of shape (n_dof,)
        The cost-function solved is min∥J*invM(τ-C*qd-G)+Jd*qd-xdd∥^2 + w_reg ||τ - τ_reg||^2
        Nullspace solved with secondary task of minimizing joint velocities and keeping elbow at 0 position (gain alpha)
        And maximizing manipulability (gain beta)
        The weight matrix W can be used to prioritize translation over rotation or vice-versa
        """
        J = self.robot.jacobe(q)
        Jdot = self.robot.jacob0_dot(q, qdot, J0=J)
        M =  self.robot.inertia(q)              # n x n
        C = self.robot.coriolis(q, qdot) @ qdot # (n,)
        G = self.robot.gravload(q)  # (n,)
        if f_ext is None:
            f_ext = np.zeros(6)

        Minv = np.linalg.inv(M + 1e-8 * np.eye(self.n_dof))  
        A = J @ Minv 
        b = A @ (-C -G + J.T @ f_ext) + Jdot @ qdot
        # QP: variable τ (n,)
        # cost = ||A τ + b - a_des||^2 
        H_task = 2 * A.T @ W @ A           # n x n
        g_task = 2 * A.T @ W @ (b - xdotdot) # n

        # Regularization around tau_reg (if None use zeros), the cost also minimizes: w_reg ||τ - τ_reg||^2 projected on the null space
        if tau_reg is None:
            tau_reg = np.zeros(self.n_dof)
        J_pinv = np.linalg.pinv(J)               # n x 6
        N = np.eye(self.n_dof) - J_pinv @ J  
        
        tau_reg_null = N @ tau_reg   # n,
        H_reg = 2 * w_tau_reg * N.T @ N
        g_reg = -2 * w_tau_reg * tau_reg_null

        self.H = H_task + H_reg
        self.g = g_task + g_reg
        
        self.lb = -abs(np.array(self.robot.taulim))
        self.ub = abs(np.array(self.robot.taulim))
        pass
        
    def add_constraint(self, A, b):
        """
        Add position constraints to the QP problem
        A: np.array of shape (m, n_dof)
        b: np.array of shape (m,)
        """
        self.A = np.vstack((self.A, A))
        self.b = np.hstack((self.b, b))
        pass
    
    def reset_constraints(self):
        """
        Reset all position constraints
        """
        self.A = np.zeros((0, self.n_dof))
        self.b = np.zeros(0)
        pass
    
    def update_joints_limits(self, q, qdot, f_ext=None):
        
        """
        Add joint limits constraints to the QP problem
        limits: np.array of shape (2, n_dof) with min and max limits for each joint
        The joint limits are converted to velocity limits based on current position and dt
        """
        J = self.robot.jacobe(self.joint_positions)
        M =  self.robot.inertia(q)              # n x n
        bias = self.robot.coriolis(q, qdot) @ qdot + self.robot.gravload(q)  # bias is coriolis + gravity (n,)
        
        qmin, qmax = self.joints_limits[0, 0:self.n_dof], self.joints_limits[1, 0:self.n_dof]
        qdot_min, qdot_max = -self.joints_velocities_limits, self.joints_velocities_limits
        
        if f_ext is None:
            f_ext = np.zeros(6)
        Minv = np.linalg.inv(M)
        bias_term = Minv @ (-J.T @ f_ext - bias)

        A_q = 0.5 * (self.dt**2) * Minv      # maps τ -> contribution to q_plus
        b_q = q + qdot * self.dt + 0.5 * (self.dt**2) * bias_term  # offset

        A_qdot = self.dt * Minv              # maps τ -> contribution to qdot_plus
        b_qdot = qdot + self.dt * bias_term

        self.add_constraint(A_q, qmax - b_q) # Position upper
        self.add_constraint(-A_q, b_q - qmin) # Position lower
        self.add_constraint(A_qdot, qdot_max - b_qdot) # Velocity upper
        self.add_constraint(-A_qdot, b_qdot - qdot_min) # Velocity lower
        pass