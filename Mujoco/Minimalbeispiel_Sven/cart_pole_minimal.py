""" Sim of the Mujoco Cart Pole model. """
import time
import mujoco
import mujoco.viewer as viewer
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


# Specify model to be loaded
model_xml_file = os.path.join(os.getcwd(), 'Mujoco', 'Minimalbeispiel', 'cartpole_minimal.xml')

# Load xml file from given path to obtain the mjModel file
model = mujoco.MjModel.from_xml_path(model_xml_file)

# Read the model's state via mjData
data = mujoco.MjData(model)

print("Number of generalized coordinates:", model.nq)
print('Total number of DoFs in the model:', model.nv)

# Launch the mujoco viewer in passive mode -> not blocking further user code
viewer = viewer.launch_passive(model=model, data=data)

# Set initial position of the cart pole
x0 = -0.1  # [m]
phi0 = 3  # [rad]
q_d = np.array([x0, phi0])

spec = 2    # spec 2 sets desired positions
mujoco.mj_setState(model, data, q_d, spec)

# Set initial velocities of the cart pole
xp0 = 0.1  # [m]
phip0 = -0.1  # [rad]
qp_d = np.array([xp0, phip0])

spec = 4    # spec 4 sets desired velocities
mujoco.mj_setState(model, data, qp_d, spec)

# Synchronize viewer
viewer.sync()

# ACHTUNG: ABTASTZEIT WIRD IM XML FILE FESTGELEGT
t_max = 10   # Maximum simulation time
t_max = 5   # Maximum simulation time
dt = 0.002  # [s] time step in Sim

# Controller parameters
Kp = 100
Kd = 1

step = 0

max_steps = int(t_max/dt)   # Maximum number of time steps for this simulation

timevals = np.zeros(shape=len(np.arange(start=0, stop=t_max+dt, step=dt)))

# Logging vectors
x_vec = np.zeros(shape=(4, len(np.arange(start=0, stop=t_max+dt, step=dt))))
ctrl_vec = np.zeros(shape=len(np.arange(start=0, stop=t_max+dt, step=dt)))

# Desired slider position
xd = 0.2

# Get the current joint angles and joint velocities
q = data.qpos
qp = data.qvel

# Save initial position
x_vec[:, step] = np.reshape(np.concatenate((q, qp)), newshape=(4, ))

# ----------------------------------------------------------------------------------------------------------------------
#                                                   SIMULATION LOOP
# ----------------------------------------------------------------------------------------------------------------------
t_sim_start = time.time()
while step < max_steps:

    # Get the current joint angles and joint velocities
    q = data.qpos
    qp = data.qvel
    # ---------------------------------------- Compute the actuation -----------------------------------------------

    # Dynamic desired position
    xd = 0.3 * np.sin(data.time)

    # Position error
    e = xd - q[0]

    # Send force command to slider
    data.ctrl = Kp * e

    # Take step forward in environment and synchronize viewer
    mujoco.mj_step(model, data)
    viewer.sync()

    time.sleep(0.001)

    step += 1

    # ------------------------------------------------- Data logging ---------------------------------------------------
    timevals[step] = data.time
    x_vec[:, step] = np.reshape(np.concatenate((q, qp)), newshape=(4, ))
    ctrl_vec[step] = data.ctrl.copy()

t_sim_end = time.time()
print("wall-clock time: ", t_sim_end - t_sim_start)

# Corrupt state vectors with measurement noise
std_m = 0.01    # noise scale
x_vec_m = np.zeros(shape=(4, len(np.arange(start=0, stop=t_max+dt, step=dt))))
for k in range(len(timevals)):
    x_vec_m[:, k] = x_vec[:, k] + np.random.normal(loc=0., scale=std_m, size=4)

# # Save to mat file
# mat_dict = {'x_m': x_vec_m, 'u': ctrl_vec, 't': timevals, 'sigma_m': std_m}
# savemat(file_name="cart_pole_mujoco_data.mat", mdict=mat_dict)

# ----------------------------------------------------------------------------------------------------------------------
#                                                   PLOT SIMULATION RESULTS
# ----------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, layout="constrained")
title_labels = ['Position / m', 'Angle / rad', 'Speed / m/s', 'Angular Speed / rad/s']
for i, ax in enumerate(fig.axes):
    ax.grid()
    ax.set_title(title_labels[i])
    ax.set_xlabel('Time / s')
    ax.plot(timevals, x_vec[i, :])
    ax.plot(timevals, x_vec_m[i, :])
    ax.legend(("Ist", "Messung"))

plt.figure(2)
plt.title("Control Input")
plt.plot(timevals, ctrl_vec)
plt.grid()
plt.ylabel("Force / N")
plt.xlabel("Time / s")

plt.show()
