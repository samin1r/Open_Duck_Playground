from playground.common.onnx_infer import OnnxInfer
import numpy as np
import time
import mujoco
import mujoco.viewer

from playground.common.utils import render_footstep

footstepnet_actor = OnnxInfer(
    "/home/antoine/Téléchargements/footsteps-planning-any-v0_actor.onnx",
    awd=True,
    input_name="onnx::Flatten_0",
)


class Feet:
    def __init__(self):
        self.foot = {}
        self.foot["left"] = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.foot["right"] = np.array([0.0, -0.15, 0.0])  # x, y, theta
        self.support_foot = "left"
        self.left_foot_color = [0, 0, 0, 0.5]
        self.right_foot_color = [1, 0, 0, 0.5]

    def set_support_foot(self, foot):
        if foot not in ["left", "right"]:
            raise ValueError("Foot must be 'left' or 'right'")
        self.support_foot = foot

    def move_foot(self, displacement, support_foot=True):
        x, y, theta = displacement
        # theta = np.deg2rad(theta)
        displacement = np.array([x, y, theta])
        if support_foot:
            self.foot[self.support_foot] += displacement
        else:
            self.foot[
                "left" if self.support_foot == "right" else "right"
            ] += displacement

    def get_T_world_support(self):
        T_world_support = np.eye(3)
        T_world_support[0:2, 2] = self.foot[self.support_foot][0:2]
        T_world_support[0:2, 0:2] = np.array(
            [
                [
                    np.cos(self.foot[self.support_foot][2]),
                    -np.sin(self.foot[self.support_foot][2]),
                ],
                [
                    np.sin(self.foot[self.support_foot][2]),
                    np.cos(self.foot[self.support_foot][2]),
                ],
            ]
        )
        return T_world_support

    def get_T_world_neutral(self):
        offset_sign = 1 if self.support_foot == "right" else -1
        T_support_neutral = np.eye(3)
        T_support_neutral[0:2, 2] = [0, offset_sign * 0.15]

        return self.get_T_world_support() @ T_support_neutral

    def draw(self, scene):
        render_footstep(
            scene,
            pos=self.foot["left"],
            color=self.left_foot_color,
        )
        render_footstep(
            scene,
            pos=self.foot["right"],
            color=self.right_foot_color,
        )


feet = Feet()
phase = np.array([0.0, 0.0])
prev_phase = np.array([0.0, 0.0])
# nb_steps_in_period = 36
nb_steps_in_period = 10
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

# model = mujoco.MjModel.from_xml_path(
#     "/home/antoine/MISC/Open_Duck_Playground/playground/sigmaban2024/xmls/empty_scene.xml"
# )
model = mujoco.MjModel.from_xml_string(scene)

model.opt.timestep = 0.002
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

freq = 50.0  # Hz

target = np.array([1.0, -0.2, 0.3])
# target = np.random.uniform([-2, -2, -np.pi], [2, 2, np.pi])
target_support_foot = "left"

action_low = [-0.08, -0.04, -20]
action_high = [0.08, 0.04, 20]


def get_obs():
    T_world_support = feet.get_T_world_support()
    T_support_world = np.linalg.inv(T_world_support)

    T_world_target = np.eye(3)
    T_world_target[0:2, 2] = target[0:2]
    T_world_target[0:2, 0:2] = np.array(
        [
            [np.cos(target[2]), -np.sin(target[2])],
            [np.sin(target[2]), np.cos(target[2])],
        ]
    )
    T_support_target = T_support_world @ T_world_target
    support_target = np.array(
        [
            T_support_target[0, 2],  # x
            T_support_target[1, 2],  # y
            T_support_target[0, 0],  # cos(theta)
            T_support_target[1, 0],  # sin(theta)
        ],
        dtype=np.float32,
    )

    # if self.options["has_obstacle"]:
    #     self.support_obstacle = tr.apply(T_support_world, self.options["obstacle_position"])

    is_target_foot = 1 if (feet.support_foot == target_support_foot) else 0

    # Handling symmetry
    if feet.support_foot == "left":
        # Invert the target foot position and orientation for the other foot
        support_target[1] = -support_target[1]
        support_target[3] = -support_target[3]

        # Invert the obstacle position for the other foot if there is one
        # if self.options["has_obstacle"]:
        #     self.support_obstacle[1] = -self.support_obstacle[1]

    state = np.concatenate(
        [
            support_target,
            [is_target_foot],
            [0, 0, 0],
        ]
    )
    state = np.array(state, dtype=np.float32)
    return state


def ellipsoid_clip(step: np.ndarray) -> np.ndarray:
    """
    Applying a rescale of the order in an "ellipsoid" manner. This transforms the target step to
    a point in a space where it should lie on a sphere, ensure its norm is not high than 1 and takes
    it back to the original scale.
    """
    factor = np.array(
        [
            (0.08 if step[0] >= 0 else 0.03),
            0.04,
            np.deg2rad(20),
        ],
        dtype=np.float32,
    )
    clipped_step = step / factor

    # In this space, the step norm should be <= 1
    norm = np.linalg.norm(clipped_step)
    if norm > 1:
        clipped_step /= norm

    return clipped_step * factor


def take_step(action):
    dx, dy, dtheta = action
    T_neutral_target = np.eye(3)
    T_neutral_target[0:2, 2] = [dx, dy]
    T_neutral_target[0:2, 0:2] = np.array(
        [
            [np.cos(dtheta), -np.sin(dtheta)],
            [np.sin(dtheta), np.cos(dtheta)],
        ]
    )
    T_world_support = feet.get_T_world_neutral() @ T_neutral_target
    # print(T_world_support)
    # print(feet.foot[feet.support_foot])
    feet.foot[feet.support_foot][:2] = T_world_support[0:2, 2]
    feet.foot[feet.support_foot][2] = np.arctan2(
        T_world_support[1, 0], T_world_support[0, 0]
    )
    feet.support_foot = (
        "left" if feet.support_foot == "right" else "right"
    )  # Switch support foot

    # T_neutral_target = tr.frame(dx, dy, dtheta)
    # self.T_world_support = self.T_world_neutral() @ T_neutral_target
    # self.support_foot = other_foot(self.support_foot)


with mujoco.viewer.launch_passive(
    model,
    data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:
    while True:
        s = time.time()

        i += 1.0
        i = i % nb_steps_in_period
        prev_phase = phase.copy()
        phase = np.array(
            [
                np.cos(i / nb_steps_in_period * 2 * np.pi),
                np.sin(i / nb_steps_in_period * 2 * np.pi),
            ]
        )
        # prev_support_foot = feet.support_foot
        # if phase[0] > 0.0:
        #     feet.set_support_foot("left")
        # else:
        #     feet.set_support_foot("right")
        # switched = prev_support_foot != feet.support_foot

        switch = np.sign(phase[0]) != np.sign(prev_phase[0])

        if switch:
            obs = get_obs()
            action = footstepnet_actor.infer(obs)
            if feet.support_foot == "left":
                action[1] = -action[1]
                action[2] = -action[2]

            action = np.clip(action, action_low, action_high)
            action = ellipsoid_clip(action)
            # take_step(action)
            feet.move_foot(action, support_foot=True)
            feet.set_support_foot("left" if feet.support_foot == "right" else "right")

        viewer.user_scn.ngeom = 0  # Clear previous custom geometries

        feet.draw(viewer.user_scn)

        render_footstep(
            viewer.user_scn,
            pos=target,
            color=[0, 1, 0, 0.5],
        )

        took = time.time() - s
        time.sleep(max(0, 1 / freq - took))

        viewer.sync()
