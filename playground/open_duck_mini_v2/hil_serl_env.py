import numpy as np
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase
import mujoco
import mujoco.viewer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.rewards_numpy import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    cost_stand_still,
    reward_alive,
)
from playground.open_duck_mini_v2.custom_rewards_numpy import reward_imitation

# TODO torch ?


class Env(MJInferBase):
    def __init__(self, model_path: str, reference_data: str, visualize=False):
        super().__init__(model_path)

        self.visualize = visualize
        if self.visualize:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                # key_callback=self.key_callback,
            )

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.nb_steps_in_period = 27
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.max_motor_velocity = 5.24  # rad/s
        self.tracking_sigma = 0.01

        self.PRM = PolyReferenceMotion(reference_data)

        self.motor_targets = self.default_actuator
        self.prev_motor_targets = self.default_actuator

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        # x vel, y vel, z vel, neck_pitch, head_pitch, head_yaw, head_roll
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.lin_vel_x = [-0.15, 0.15]
        self.lin_vel_y = [-0.2, 0.2]
        self.ang_vel_yaw = [-1.0, 1.0]
        self.neck_pitch_range = [-0.34, 1.1]
        self.head_pitch_range = [-0.78, 0.78]
        self.head_yaw_range = [-1.5, 1.5]
        self.head_roll_range = [-0.5, 0.5]
        self.head_range_factor = 1.0

    def step(self, action):
        self._tick_imitation_stuff()

        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()

        self.motor_targets = self.default_actuator + action * self.action_scale

        self.motor_targets = np.clip(
            self.motor_targets,
            self.prev_motor_targets
            - self.max_motor_velocity * (self.sim_dt * self.decimation),
            self.prev_motor_targets
            + self.max_motor_velocity * (self.sim_dt * self.decimation),
        )

        self.prev_motor_targets = self.motor_targets.copy()

        # apply action
        self.data.ctrl[:] = self.motor_targets.copy()

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        if self.visualize:
            self.viewer.sync()

        obs = self._get_obs()
        reward = self._get_reward()
        truncated = self._truncated()
        done = self._done()
        info = {}

        return (
            obs,
            reward,
            truncated,
            done,
            info,
        )

    def reset(self):

        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.ctrl[:] = self.default_actuator
        self.data.qvel[:] = np.zeros(len(self.data.qvel))

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])

        self.motor_targets = self.default_actuator
        self.prev_motor_targets = self.default_actuator

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.commands = self._sample_command()

        return self._get_obs()

    def _get_obs(self):

        joint_angles = self.get_actuator_joints_qpos(self.data.qpos)
        joint_vel = self.get_actuator_joints_qvel(self.data.qvel)

        obs = np.hstack(
            [
                self.get_gyro(self.data),  # 3
                self.get_accelerometer(self.data),  # 3
                self.commands,  # 3
                joint_angles - self.default_actuator,  # 14
                joint_vel * self.dof_vel_scale,  # 14
                self.last_action,  # 14
                self.last_last_action,  # 14
                self.last_last_last_action,  # 14
                self.motor_targets,  # 14
                self.get_feet_contacts(self.data),  # 2
                self.imitation_phase,
            ]
        )

        ret = {
            "observation.state": obs,
        }
        return ret

    def _get_reward(self):
        lin_vel = reward_tracking_lin_vel(
            self.commands, self.get_linvel(self.data), self.tracking_sigma
        )
        ang_vel = reward_tracking_ang_vel(
            self.commands, self.get_gyro(self.data), self.tracking_sigma
        )
        torques = cost_torques(self.data.actuator_force)
        action_rate = cost_action_rate(
            self.last_action, self.last_last_action
        )  # TODO not exactly the same
        alive = reward_alive()
        imitation = reward_imitation(
            self.get_floating_base_qpos(self.data.qpos),
            self.get_floating_base_qvel(self.data.qvel),
            self.get_actuator_joints_qpos(self.data.qpos),
            self.get_actuator_joints_qvel(self.data.qvel),
            self.get_feet_contacts(self.data),
            self.PRM.get_reference_motion(*self.commands[:3], self.imitation_i),
            self.commands,
            True,
        )
        stand_still = cost_stand_still(
            self.commands,
            self.get_actuator_joints_qpos(self.data.qpos),
            self.get_actuator_joints_qvel(self.data.qvel),
            self.default_actuator,
            ignore_head=False,
        )

        reward = (
            lin_vel * 2.5
            + ang_vel * 6.0
            + torques * -1.0e-3
            + action_rate * -0.5
            + alive * 20.0
            + imitation * 1.0
            + stand_still * -0.2
        )
        return reward

    def _truncated(self):
        fall_termination = self.get_gravity(self.data)[-1] < 0.0
        return (
            fall_termination
            | np.isnan(self.data.qpos).any()
            | np.isnan(self.data.qvel).any()
        )

    def _done(self):
        return False

    def _tick_imitation_stuff(self):
        self.imitation_i += 1
        self.imitation_i = self.imitation_i % self.nb_steps_in_period
        self.imitation_phase = np.array(
            [
                np.cos(self.imitation_i / self.nb_steps_in_period * 2 * np.pi),
                np.sin(self.imitation_i / self.nb_steps_in_period * 2 * np.pi),
            ]
        )

    def _sample_command(self):

        x_vel = np.random.uniform(self.lin_vel_x[0], self.lin_vel_x[1])
        y_vel = np.random.uniform(self.lin_vel_y[0], self.lin_vel_y[1])
        ang_vel = np.random.uniform(self.ang_vel_yaw[0], self.ang_vel_yaw[1])
        neck_pitch = np.random.uniform(
            self.neck_pitch_range[0], self.neck_pitch_range[1]
        )
        head_pitch = np.random.uniform(
            self.head_pitch_range[0], self.head_pitch_range[1]
        )
        head_yaw = np.random.uniform(self.head_yaw_range[0], self.head_yaw_range[1])
        head_roll = np.random.uniform(self.head_roll_range[0], self.head_roll_range[1])

        return [x_vel, y_vel, ang_vel, neck_pitch, head_pitch, head_yaw, head_roll]


if __name__ == "__main__":
    env = Env(
        "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml",
        "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
        visualize=True,
    )
    obs = env.reset()
    for _ in range(10000):
        action = np.random.randn(14)
        obs, reward, truncated, done, info = env.step(action)
        print(reward)
        if done:
            break
