import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion

from playground.sigmaban2024.mujoco_infer_base import MJInferBase
from playground.sigmaban2024.footstepnet_wrapper_numpy import Trajectory
from playground.common.utils import render_plane

USE_MOTOR_SPEED_LIMITS = False


class MjInfer(MJInferBase):

    def __init__(self, model_path: str, reference_data: str, onnx_model_path: str):
        super().__init__(model_path)

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 1.0

        self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.footstepnet_actor = OnnxInfer(
            "/home/antoine/Téléchargements/footsteps-planning-any-v0_actor.onnx",
            awd=True,
            input_name="onnx::Flatten_0",
        )

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

        self.TR = Trajectory(
            model_path="playground/sigmaban2024/data/footsteps-planning-any-v0_actor.onnx"
        )

        self.TR.sample_trajectory(
            [0.0, 0.15 / 2, 0.0],
            "left",
            np.random.uniform([-2, -2, -np.pi], [2, 2, np.pi]),
            "left",
        )
        self.trajectory_i = 0
        # self.commands[:3] = self.TR.velocities[self.trajectory_i]

    def get_feet_contacts(self, data):
        left_foot_cleat_back_left = self.check_contact(
            data, "left_foot_cleat_back_left", "floor"
        )
        left_foot_cleat_back_right = self.check_contact(
            data, "left_foot_cleat_back_right", "floor"
        )
        left_foot_cleat_front_left = self.check_contact(
            data, "left_foot_cleat_front_left", "floor"
        )
        left_foot_cleat_front_right = self.check_contact(
            data, "left_foot_cleat_front_right", "floor"
        )
        right_foot_cleat_back_left = self.check_contact(
            data, "right_foot_cleat_back_left", "floor"
        )
        right_foot_cleat_back_right = self.check_contact(
            data, "right_foot_cleat_back_right", "floor"
        )
        right_foot_cleat_front_left = self.check_contact(
            data, "right_foot_cleat_front_left", "floor"
        )
        right_foot_cleat_front_right = self.check_contact(
            data, "right_foot_cleat_front_right", "floor"
        )
        left_contact = (
            left_foot_cleat_back_left
            or left_foot_cleat_back_right
            or left_foot_cleat_front_left
            or left_foot_cleat_front_right
        )
        right_contact = (
            right_foot_cleat_back_left
            or right_foot_cleat_back_right
            or right_foot_cleat_front_left
            or right_foot_cleat_front_right
        )

        # left_contact = self.check_contact(data, "left_foot___list_t0v6opd9rekumc_default", "floor")
        # right_contact = self.check_contact(data, "right_foot_", "floor")
        return left_contact, right_contact

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        linvel = self.get_linvel(data)

        obs = np.concatenate(
            [
                # linvel,
                gyro,
                accelerometer,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                self.imitation_phase,
            ]
        )

        return obs

    def get_projected_left_foot(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ps_2")
        pos = self.data.xpos[body_id]  # np.array([x, y, z])

        offset = [0.14 / 2, -0.08 / 2, 0.0]
        mat = self.data.xmat[body_id].reshape(3, 3)  # rotation matrix

        # project pos on the ground
        pos[2] = 0.001

        # cancel all rotation except z
        theta = np.arctan2(mat[1, 0], mat[0, 0])

        # Build a pure yaw rotation matrix
        mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        # apply the offset to the position resulting from the rotation
        # the offset is in the local frame of the left foot
        offset_world = mat @ offset
        pos += offset_world

        return pos, theta, mat

    def project_left_foot(self, scene, pos, mat):

        render_plane(
            scene,
            pos,
            mat,
            [0.14, 0.08],
            [1, 0, 0, 0.5],
        )

    def key_callback(self, keycode):
        return 0
        print(f"key: {keycode}")
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0

        if keycode == 265:  # arrow up
            lin_vel_x = self.COMMANDS_RANGE_X[1]
        if keycode == 264:  # arrow down
            lin_vel_x = self.COMMANDS_RANGE_X[0]
        if keycode == 263:  # arrow left
            lin_vel_y = self.COMMANDS_RANGE_Y[1]
        if keycode == 262:  # arrow right
            lin_vel_y = self.COMMANDS_RANGE_Y[0]
        if keycode == 81:  # a
            ang_vel = self.COMMANDS_RANGE_THETA[1]
        if keycode == 69:  # e
            ang_vel = self.COMMANDS_RANGE_THETA[0]
        if keycode == 80:  # p
            self.phase_frequency_factor += 0.1
        if keycode == 59:  # m
            self.phase_frequency_factor -= 0.1

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel
        print(self.commands)

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:
                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        viewer.user_scn.ngeom = 0  # Clear previous custom geometries

                        left_foot_pos, left_foot_theta, left_foot_mat = (
                            self.get_projected_left_foot()
                        )

                        self.imitation_i += 1.0 * self.phase_frequency_factor
                        self.imitation_i = (
                            self.imitation_i % self.PRM.nb_steps_in_period
                        )

                        self.prev_imitation_phase = self.imitation_phase.copy()
                        self.imitation_phase = np.array(
                            [
                                np.cos(
                                    self.imitation_i
                                    / self.PRM.nb_steps_in_period
                                    * 2
                                    * np.pi
                                ),
                                np.sin(
                                    self.imitation_i
                                    / self.PRM.nb_steps_in_period
                                    * 2
                                    * np.pi
                                ),
                            ]
                        )
                        switch = np.sign(self.imitation_phase[0]) != np.sign(
                            self.prev_imitation_phase[0]
                        )
                        if switch:
                            if self.trajectory_i >= len(self.TR.world_velocities) - 1:
                                pos = [
                                    left_foot_pos[0],
                                    left_foot_pos[1],
                                    left_foot_theta,
                                ]
                                self.TR.sample_trajectory(
                                    pos,
                                    "left",
                                    np.random.uniform([-2, -2, -np.pi], [2, 2, np.pi]),
                                    "left",
                                )
                                self.trajectory_i = 0
                            else:
                                self.trajectory_i += 1

                            # # [lin_vel_x, lin_vel_y, ang_vel]
                            world_velocities = self.TR.world_velocities[
                                self.trajectory_i
                            ]

                            theta = left_foot_theta
                            # Extract linear and angular parts
                            linear_vel_world = world_velocities[:2]  # [vx, vy]
                            angular_vel_world = world_velocities[2]  # omega (yaw rate)

                            # Rotation matrix to rotate from world to local frame (2D)
                            rot_inv_2d = np.array(
                                [
                                    [np.cos(theta), np.sin(theta)],
                                    [-np.sin(theta), np.cos(theta)],
                                ]
                            )

                            # Rotate linear velocity
                            linear_vel_local = rot_inv_2d @ linear_vel_world

                            # Angular velocity stays the same
                            angular_vel_local = angular_vel_world

                            # Combine
                            local_velocities = np.hstack(
                                (linear_vel_local, angular_vel_local)
                            )

                            self.commands[:3] = local_velocities

                        obs = self.get_obs(
                            self.data,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )

                            self.prev_motor_targets = self.motor_targets.copy()

                        # head_targets = self.commands[3:]
                        # self.motor_targets[5:9] = head_targets
                        self.data.ctrl = self.motor_targets.copy()

                        self.project_left_foot(
                            viewer.user_scn, left_foot_pos, left_foot_mat
                        )

                        self.TR.render(
                            viewer.user_scn,
                        )

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/sigmaban2024/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/sigmaban2024/xmls/scene_flat_terrain.xml",
    )

    args = parser.parse_args()

    mjinfer = MjInfer(args.model_path, args.reference_data, args.onnx_model_path)
    mjinfer.run()
