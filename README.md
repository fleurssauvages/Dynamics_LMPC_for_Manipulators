# 🤖 Acceleration based LMPC for Robotic Manipulators

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-numpy%2C%20roboticstoolbox--python-green)](https://pypi.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project demonstrates **Quadratic Programming (QP)** and **Linear Model Predictive Control (LMPC)** for robotic manipulators in dynamics.  
- **LMPC** is used for online path planning in dynamic workspaces.
- **QP** solves Inverse Kinematics (IK) with constraints such as joint limits.

For easier understanding, a velocity based implementation can be found at: https://github.com/fleurssauvages/LMPC_for_Manipulators.
This repository is acceleration based, with the state of the LMPC being augmented to both position and velocity, outputing a desired acceleration, and the QP calculating a torque for a given desired acceleration.

The LMPC problem formulation is based on:  
> Alberto, Nicolas Torres, et al.  
> ["Predictive control of collaborative robots in dynamic
contexts."](https://hal.science/hal-03790059/document) (2023)

---

## ⚡ Installation

The robotics toolbox experiences some bugs when using URDF and Dynamics. The DH robot is used for dynamics calculation, and the URDF for visualization.
Clone the repository and install the required dependencies:

```bash
pip install roboticstoolbox-python numpy
```

---

## 🚀 Run the Simulations

- **LMPC + QP for Path Planning**  
  ```bash
  python simulation_robot_LMPC.py
  ```
## 📂 Project Structure

```
├── simulation_robot_LMPC.py        # LMPC + QP for path planning
├── README.md                       # Project documentation
└── LICENSE                         # License file
```

## 📜 License
This project is licensed under the [MIT License](LICENSE).  

---

## ⭐ Acknowledgments
- Inspired by the work of Alberto, Nicolas Torres, et al. (2023).  
- Built with [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python).  
