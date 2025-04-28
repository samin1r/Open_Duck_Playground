import numpy as np
import time
import mujoco
import mujoco.viewer

from playground.sigmaban2024.footstepnet_wrapper import FootstepnetWrapper, Trajectory

FW = FootstepnetWrapper(
    "/home/antoine/Téléchargements/footsteps-planning-any-v0_actor.onnx"
)
TR = Trajectory()

phase = np.array([0.0, 0.0])
prev_phase = np.array([0.0, 0.0])
# nb_steps_in_period = 36
nb_steps_in_period = 2
# nb_steps_in_period = 100
i = 0.0

sim_dt = 0.002
decimation = 10

scene = """
<mujoco model="scene">

	<visual>
		<headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
		<rgba haze="0.15 0.25 0.35 1"/>
		<global azimuth="160" elevation="-20"/>
	</visual>

	<asset>
		<texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
		<texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0"
			width="300" height="300"/>
		<material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
	</asset>

	<worldbody>
		<body name="floor">
			<geom name="floor" size="0 0 0.01" type="plane" material="groundplane" contype="1" conaffinity="0"
				priority="1" friction="0.6" condim="3"/>
		</body>
	</worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(scene)

model.opt.timestep = 0.002
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

freq = 50.0  # Hz

with mujoco.viewer.launch_passive(
    model,
    data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:
    while True:
        s = time.time()
        viewer.user_scn.ngeom = 0  # Clear previous custom geometries

        i += 1.0
        i = i % nb_steps_in_period
        prev_phase = phase.copy()
        phase = np.array(
            [
                np.cos(i / nb_steps_in_period * 2 * np.pi),
                np.sin(i / nb_steps_in_period * 2 * np.pi),
            ]
        )

        switch = np.sign(phase[0]) != np.sign(prev_phase[0])

        if switch:
            FW.step()

        FW.render(viewer.user_scn)

        if FW.reached_target():
            FW.reset_random()
            TR.sample_trajectory(
                FW.feet.foot[FW.feet.support_foot],
                FW.feet.support_foot,
                FW.target,
                FW.target_support_foot,
            )

        TR.render(viewer.user_scn)

        took = time.time() - s
        time.sleep(max(0, 1 / freq - took))

        viewer.sync()
