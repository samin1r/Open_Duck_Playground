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


def render_footstep(scene, pos, color=[1, 0, 0, 0.5]):
    """
    pos : [x, y, theta] (m, m, rad)
    """
    x, y = pos[:2]
    center = [x, y, 0.001]
    theta = pos[2]
    size = [0.14, 0.08]

    rot_mat = np.eye(3)
    rot_mat[0, 0] = np.cos(theta)
    rot_mat[0, 1] = -np.sin(theta)
    rot_mat[1, 0] = np.sin(theta)
    rot_mat[1, 1] = np.cos(theta)

    render_plane(scene, center=center, size=size, rot_mat=rot_mat, rgba=color)
