from playground.common.onnx_infer import OnnxInfer
from playground.common.utils import render_footstep, render_obstacle
import numpy as np


def close(pos1, pos2, tol=0.05):
    distance = np.linalg.norm(pos1[:2] - pos2[:2])
    if distance > tol:
        return False

    angle_diff = np.arctan2(np.sin(pos1[2] - pos2[2]), np.cos(pos1[2] - pos2[2]))
    if abs(angle_diff) > tol:
        return False
    return True


class Feet:
    def __init__(self, foot_size=[0.14, 0.08], feet_spacing=0.15):
        self.foot_size = foot_size
        self.feet_spacing = feet_spacing
        self.foot = {}
        self.foot["left"] = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.foot["right"] = np.array([0.0, -self.feet_spacing, 0.0])  # x, y, theta
        self.support_foot = "left"
        self.left_foot_color = [0, 0, 0, 0.5]
        self.right_foot_color = [1, 0, 0, 0.5]

    def copy(self):
        feet_copy = Feet(self.foot_size, self.feet_spacing)
        feet_copy.foot["left"] = self.foot["left"].copy()
        feet_copy.foot["right"] = self.foot["right"].copy()
        feet_copy.support_foot = self.support_foot
        return feet_copy

    def set_support_foot(self, foot):
        if foot not in ["left", "right"]:
            raise ValueError("Foot must be 'left' or 'right'")
        self.support_foot = foot

    def switch_support_foot(self):
        if self.support_foot == "left":
            self.support_foot = "right"
        else:
            self.support_foot = "left"

    def move_swing_foot(self, displacement):
        x, y, theta = displacement
        displacement = np.array([x, y, theta])

        T_neutral_target = np.eye(3)
        T_neutral_target[0:2, 2] = [x, y]
        T_neutral_target[0:2, 0:2] = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        T_world_neutral = self.get_T_world_neutral()

        T_world_other = T_world_neutral @ T_neutral_target

        self.foot[self.get_other_foot()][:2] = T_world_other[0:2, 2]
        self.foot[self.get_other_foot()][2] = np.arctan2(
            T_world_other[1, 0], T_world_other[0, 0]
        )

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

    def get_T_world_other_foot(self):
        T_world_other_foot = np.eye(3)
        T_world_other_foot[0:2, 2] = self.foot[
            "left" if self.support_foot == "right" else "right"
        ][0:2]
        T_world_other_foot[0:2, 0:2] = np.array(
            [
                [
                    np.cos(
                        self.foot["left" if self.support_foot == "right" else "right"][
                            2
                        ]
                    ),
                    -np.sin(
                        self.foot["left" if self.support_foot == "right" else "right"][
                            2
                        ]
                    ),
                ],
                [
                    np.sin(
                        self.foot["left" if self.support_foot == "right" else "right"][
                            2
                        ]
                    ),
                    np.cos(
                        self.foot["left" if self.support_foot == "right" else "right"][
                            2
                        ]
                    ),
                ],
            ]
        )
        return T_world_other_foot

    def get_other_foot(self):
        return "left" if self.support_foot == "right" else "right"

    def get_T_world_neutral(self):
        offset_sign = 1 if self.support_foot == "right" else -1
        T_support_neutral = np.eye(3)
        T_support_neutral[0:2, 2] = [0, offset_sign * self.feet_spacing]

        return self.get_T_world_support() @ T_support_neutral

    def draw(self, scene, alpha=1.0):

        left_color = self.left_foot_color.copy()
        left_color[3] = alpha
        right_color = self.right_foot_color.copy()
        right_color[3] = alpha
        render_footstep(
            scene,
            pos=self.foot["left"],
            foot_size=self.foot_size,
            color=left_color,
        )
        render_footstep(
            scene,
            pos=self.foot["right"],
            foot_size=self.foot_size,
            color=right_color,
        )


class FootstepnetWrapper:
    def __init__(self, model_path):
        self.policy = OnnxInfer(
            model_path,
            awd=True,
            input_name="onnx::Flatten_0",
        )
        self.target = [0, 0, 0]
        self.target_support_foot = "left"
        self.feet = Feet()
        self.action_low = [-0.08, -0.04, np.deg2rad(-20)]
        self.action_high = [0.08, 0.04, np.deg2rad(20)]
        self.saved_footsteps = []
        self.max_saved_footsteps = 10

    def step(self):
        self.saved_footsteps.append(self.feet.copy())
        self.saved_footsteps = self.saved_footsteps[-self.max_saved_footsteps :]

        obs = self.get_obs()
        action = self.policy.infer(obs)
        if self.feet.support_foot == "left":
            action[1] = -action[1]
            action[2] = -action[2]

        action = np.clip(action, self.action_low, self.action_high)
        action = self.ellipsoid_clip(action)
        self.feet.move_swing_foot(action)
        self.feet.switch_support_foot()

    def render(self, scene, render_steps_history=True):
        self.feet.draw(scene)

        for i, feet in enumerate(self.saved_footsteps):
            alpha = i / len(self.saved_footsteps)
            feet.draw(scene, alpha=alpha)

        render_footstep(
            scene,
            pos=self.target,
            foot_size=self.feet.foot_size,
            color=[0, 1, 0, 0.5],
        )

    def reached_target(self):
        support_foot = self.feet.foot[self.feet.support_foot]
        return close(support_foot, self.target, tol=0.05)

    def reset_random(self):
        self.target = np.random.uniform([-2, -2, -np.pi], [2, 2, np.pi])
        self.obstacle = np.random.uniform([-2, -2, 0], [2, 2, 0.25])
        self.target_support_foot = (
            "left" if self.feet.support_foot == "right" else "right"
        )

    def get_obs(self):
        T_world_support = self.feet.get_T_world_support()
        T_support_world = np.linalg.inv(T_world_support)

        T_world_target = np.eye(3)
        T_world_target[0:2, 2] = self.target[0:2]
        T_world_target[0:2, 0:2] = np.array(
            [
                [np.cos(self.target[2]), -np.sin(self.target[2])],
                [np.sin(self.target[2]), np.cos(self.target[2])],
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

        is_target_foot = (
            1 if (self.feet.support_foot == self.target_support_foot) else 0
        )

        # Handling symmetry
        if self.feet.support_foot == "left":
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

    def ellipsoid_clip(self, step: np.ndarray) -> np.ndarray:
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
