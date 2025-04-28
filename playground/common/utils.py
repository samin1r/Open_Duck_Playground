import jax.numpy as jp
import jax
import mujoco
import numpy as np


class LowPassActionFilter:
    def __init__(self, control_freq, cutoff_frequency=30.0):
        self.last_action = 0
        self.current_action = 0
        self.control_freq = float(control_freq)
        self.cutoff_frequency = float(cutoff_frequency)
        self.alpha = self.compute_alpha()

    def compute_alpha(self):
        return (1.0 / self.cutoff_frequency) / (
            1.0 / self.control_freq + 1.0 / self.cutoff_frequency
        )

    def push(self, action: jax.Array) -> None:
        self.current_action = jp.array(action)

    def get_filtered_action(self) -> jax.Array:
        self.last_action = (
            self.alpha * self.last_action + (1 - self.alpha) * self.current_action
        )
        return self.last_action


def render_plane(scene, center, rot_mat, size, rgba):
    if scene.ngeom >= len(scene.geoms):
        return

    sx, sy = size

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_PLANE,
        size=np.array(
            [sx / 2, sy / 2, 0.0], dtype=np.float64
        ),  # Planes have no thickness
        pos=np.array(center, dtype=np.float64),
        mat=rot_mat.flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )

    scene.ngeom += 1


def render_footstep(scene, pos, foot_size, color=[1, 0, 0, 0.5]):
    """
    pos : [x, y, theta] (m, m, rad)
    """
    x, y = pos[:2]
    center = [x, y, 0.001]
    theta = pos[2]
    size = foot_size

    rot_mat = np.eye(3)
    rot_mat[0, 0] = np.cos(theta)
    rot_mat[0, 1] = -np.sin(theta)
    rot_mat[1, 0] = np.sin(theta)
    rot_mat[1, 1] = np.cos(theta)

    tip_size = [size[0] / 6, size[1]]
    tip_color = [0, 0, 0, color[3]]

    # rotate and translate the tip
    tip_rot_mat = np.eye(3)
    tip_rot_mat[0, 0] = np.cos(theta)
    tip_rot_mat[0, 1] = -np.sin(theta)
    tip_rot_mat[1, 0] = np.sin(theta)
    tip_rot_mat[1, 1] = np.cos(theta)
    tip_center = [
        x + size[0] / 2 * np.cos(theta),
        y + size[0] / 2 * np.sin(theta),
        0.001,
    ]
    tip_center = np.array(tip_center, dtype=np.float64)
    tip_rot_mat = np.array(tip_rot_mat, dtype=np.float64)

    render_plane(scene, center=center, size=size, rot_mat=rot_mat, rgba=color)
    render_plane(
        scene,
        center=tip_center,
        size=tip_size,
        rot_mat=tip_rot_mat,
        rgba=tip_color,
    )


def render_obstacle(scene, obstacle, color=[1, 0, 0, 0.5]):
    """
    obstacle : [x, y, radius] (m, m, m)
    """
    # render as a cylinder
    x, y = obstacle[:2]
    radius = obstacle[2]

    center = [x, y, 0.001]
    size = [radius, radius]

    rot_mat = np.eye(3)

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([radius, 0.5, 0.0], dtype=np.float64),
        pos=np.array(center, dtype=np.float64),
        mat=rot_mat.flatten(),  # orientation matrix (3x3 flattened)
        rgba=np.array(color, dtype=np.float32),
    )
    scene.ngeom += 1  # don't forget to increment after adding a geom
