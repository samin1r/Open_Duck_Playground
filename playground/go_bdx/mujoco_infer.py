import mujoco
import pickle
import numpy as np
import glfw

import mujoco.viewer
import time
import argparse

from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
parser.add_argument("-k", action="store_true", default=False, help="Use keyboard control")
parser.add_argument("-rough", action="store_true", default=False, help="Use rough terrain")
parser.add_argument("-joystick", action="store_true", default=False, help="Use joystick control")
parser.add_argument(
    "--reference_data",
    type=str,
    default="playground/go_bdx/data/polynomial_coefficients.pkl",
)
args = parser.parse_args()

NUM_DOFS = 10

joystick2 = None
if args.k or args.joystick:
    import pygame
    pygame.init()
    if args.k:
        screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Press arrow keys to move robot")
    if args.joystick:
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print("Joystick initialized:", joystick.get_name())
            joystick2 = None
            if pygame.joystick.get_count() > 1:
                joystick2 = pygame.joystick.Joystick(1)
                joystick2.init()
                print("Joystick 2 (drag) initialized:", joystick2.get_name())
            else:
                print("Only one joystick detected; interactive drag via second joystick will be disabled.")
        else:
            print("No joystick found!")
            exit(1)

# Params
linearVelocityScale = 1.0
angularVelocityScale = 1.0
dof_pos_scale = 1.0
dof_vel_scale = 0.05
action_scale = 1.0

PRM = PolyReferenceMotion(args.reference_data)

terrain_path="playground/go_bdx/xmls/scene_mjx_flat_terrain.xml"
if args.rough:
    terrain_path="playground/go_bdx/xmls/scene_mjx_rough_terrain.xml"
model = mujoco.MjModel.from_xml_path(terrain_path)
model.opt.timestep = 0.002
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

home_frame = model.keyframe("home")
home_qpos = np.array(home_frame.qpos)
home_ctrl = np.array(home_frame.ctrl)

init_pos = np.array(home_frame.qpos[7:])  # robot joints only, using jax.numpy style

policy = OnnxInfer(args.onnx_model_path, awd=True)

COMMANDS_RANGE_X = [-0.1, 0.15]
COMMANDS_RANGE_Y = [-0.2, 0.2]
COMMANDS_RANGE_THETA = [-0.5, 0.5]  # [-1.0, 1.0]

last_action = np.zeros(NUM_DOFS)
last_last_action = np.zeros(NUM_DOFS)
last_last_last_action = np.zeros(NUM_DOFS)
commands = [0.0, 0.0, 0.0]
decimation = 10

data.qpos[:] = home_qpos
data.ctrl[:] = home_ctrl

gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
gyro_dimensions = 3
linvel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel")
linvel_dimensions = 3


imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")

imitation_i = 0


def get_sensor(model, data, name, dimensions):
    i = model.sensor_name2id(name)
    return data.sensordata[i : i + dimensions]


def get_gyro(data):
    return data.sensordata[gyro_id : gyro_id + gyro_dimensions]


def get_linvel(data):
    return data.sensordata[linvel_id : linvel_id + linvel_dimensions]


def get_gravity(data):
    return data.site_xmat[imu_site_id].reshape((3, 3)).T @ np.array([0, 0, -1])


def check_contact(data, model, body1_name, body2_name):
    body1_id = data.body(body1_name).id
    body2_id = data.body(body2_name).id

    for i in range(data.ncon):
        try:
            contact = data.contact[i]
        except Exception as e:
            return False

        if (
            model.geom_bodyid[contact.geom1] == body1_id
            and model.geom_bodyid[contact.geom2] == body2_id
        ) or (
            model.geom_bodyid[contact.geom1] == body2_id
            and model.geom_bodyid[contact.geom2] == body1_id
        ):
            return True

    return False


def get_feet_contacts():
    left_contact = check_contact(data, model, "left_foot", "floor")
    right_contact = check_contact(data, model, "right_foot", "floor")
    return left_contact, right_contact


def get_obs(
    data, last_action, command  # , qvel_history, qpos_error_history, gravity_history
):
    gravity = get_gravity(data)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]

    contacts = get_feet_contacts()

    ref = PRM.get_reference_motion(*command, imitation_i)

    obs = np.concatenate(
        [
            gravity,
            command,
            joint_angles - init_pos,
            joint_vel * dof_vel_scale,
            last_action,
            last_last_action,
            last_last_last_action,
            contacts,
            ref,
        ]
    )
    return obs


def key_callback(keycode):
    pass


def handle_keyboard():
    global commands
    keys = pygame.key.get_pressed()
    lin_vel_x = 0
    lin_vel_y = 0
    ang_vel = 0
    if keys[pygame.K_z]:
        lin_vel_x = COMMANDS_RANGE_X[1]
    if keys[pygame.K_s]:
        lin_vel_x = COMMANDS_RANGE_X[0]
    if keys[pygame.K_q]:
        lin_vel_y = COMMANDS_RANGE_Y[1]
    if keys[pygame.K_d]:
        lin_vel_y = COMMANDS_RANGE_Y[0]
    if keys[pygame.K_a]:
        ang_vel = COMMANDS_RANGE_THETA[1]
    if keys[pygame.K_e]:
        ang_vel = COMMANDS_RANGE_THETA[0]

    commands[0] = lin_vel_x
    commands[1] = lin_vel_y
    commands[2] = ang_vel

    pygame.event.pump()  # process event queue


def get_body_xmat(body_id):
    # If data.xmat is 1D, use flat slicing; if 2D, index directly.
    if data.xmat.ndim == 1:
        return data.xmat[body_id*9:(body_id+1)*9].reshape(3, 3)
    else:
        return data.xmat[body_id].reshape(3, 3)

def duck_reset():
    mujoco.mj_resetData(model, data)
    data.qpos[:] = home_qpos
    data.ctrl[:] = home_ctrl
    imitation_i = 0
    last_action.fill(0)
    last_last_action.fill(0)
    last_last_last_action.fill(0)


def handle_joystick():
    global commands, root_id
    pygame.event.pump()

    if joystick2 is not None:
        drag_x = joystick2.get_axis(0)
        drag_y = joystick2.get_axis(1)
        deadzone = 0.1
        if abs(drag_x) < deadzone:
            drag_x = 0.0
        if abs(drag_y) < deadzone:
            drag_y = 0.0

        drag_force_scale = 10.0  # newtons
        global_drag_force = np.array([drag_x * drag_force_scale,
                                      drag_y * drag_force_scale,
                                      0.0])
        xmat = get_body_xmat(root_id)
        local_drag_force = np.linalg.inv(xmat) @ global_drag_force
        data.xfrc_applied[root_id, :3] += local_drag_force
        # print("Applied drag force (local):", local_drag_force)

    if joystick.get_button(0):
        duck_reset()

    joy_y = joystick.get_axis(1)
    joy_x = joystick.get_axis(0)

    if joy_y < 0:
        lin_vel_x = (-joy_y) * COMMANDS_RANGE_X[1]
    else:
        lin_vel_x = -joy_y * abs(COMMANDS_RANGE_X[0])
    lin_vel_y = -joy_x * COMMANDS_RANGE_Y[1]

    commands[0] = lin_vel_x
    commands[1] = lin_vel_y


saved_obs = []
root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "root")
floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "floor")
try:
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
    ) as viewer:
        counter = 0
        while True:

            step_start = time.time()

            mujoco.mj_step(model, data)

            counter += 1

            if counter % decimation == 0:
                imitation_i += 1
                imitation_i = imitation_i % PRM.nb_steps_in_period
                obs = get_obs(
                    data,
                    last_action,
                    commands,
                )
                saved_obs.append(obs)
                action = policy.infer(obs)

                last_last_last_action = last_last_action.copy()
                last_last_action = last_action.copy()
                last_action = action.copy()
                # action = np.zeros(10)
                action = init_pos + action * action_scale
                data.ctrl = action.copy()

            viewer.sync()

            if args.joystick:
                handle_joystick()
            elif args.k:
                handle_keyboard()

            root_z = data.xpos[root_id][2]
            floor_z = data.xpos[floor_id][2]
            if root_z < floor_z:
                print("Duck fell over. Resetting")
                duck_reset()
                continue

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
except KeyboardInterrupt:
    pickle.dump(saved_obs, open("mujoco_saved_obs.pkl", "wb"))
