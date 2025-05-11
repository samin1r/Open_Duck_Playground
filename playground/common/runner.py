"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from ml_collections import config_dict
from abc import ABC
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from brax.training.agents.sac import networks as sac_networks, train as sac
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax

from playground.common.export_onnx import export_onnx
from mujoco_playground.config import dm_control_suite_params
from flax import linen


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        self.algo = args.algo

        if self.algo == "ppo":
            self.algo_params = locomotion_params.brax_ppo_config(
                "BerkeleyHumanoidJoystickFlatTerrain"
            )  # TODO
        elif self.algo == "sac":
            self.algo_params = dm_control_suite_params.brax_sac_config("HumanoidWalk")

            # Matching the params of brax ppo from locomotion_params.py
            self.algo_params.discounting = 0.97
            self.algo_params.learning_rate = 3e-4
            self.algo_params.batch_size = 256
            self.algo_params.num_evals = 15
            self.algo_params.num_resets_per_eval = 1
            self.algo_params.network_factory = config_dict.create(
                q_network_layer_norm=True,
                hidden_layer_sizes=(512, 256, 128),
                activation=linen.swish,
                # value_hidden_layer_sizes=(512, 256, 128),
                # policy_obs_key="state",
                # value_obs_key="privileged_state",
            )

        else:
            raise ValueError(f"Unknown algorithm {self.algo}")

        self.algo_training_params = dict(self.algo_params)

        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

    def progress_callback(self, num_steps: int, metrics: dict) -> None:

        for metric_name, metric_value in metrics.items():
            # Convert to float, but watch out for 0-dim JAX arrays
            self.writer.add_scalar(metric_name, metric_value, num_steps)

        print("-----------")
        print(
            f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}'
        )
        print("-----------")

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
        mean = params[0].mean["state"]
        std = params[0].std["state"]
        policy_params = params[1].policy["params"]
        export_onnx(
            mean,
            std,
            policy_params,
            self.action_size,
            self.algo_params,
            self.obs_size,  # may not work
            output_path=onnx_export_path,
        )

    def train(self) -> None:

        # if self.algo == "ppo":
        #     self.algo_params = locomotion_params.brax_ppo_config(
        #         "BerkeleyHumanoidJoystickFlatTerrain"
        #     )  # TODO
        # elif self.algo == "sac":
        #     self.algo_params = dm_control_suite_params.brax_sac_config("HumanoidWalk")
        # else:
        #     raise ValueError(f"Unknown algorithm {self.algo}")

        # self.algo_training_params = dict(self.algo_params)

        if "network_factory" in self.algo_params:
            if self.algo == "ppo":
                network_factory = functools.partial(
                    ppo_networks.make_ppo_networks, **self.algo_params.network_factory
                )
            else:
                network_factory = functools.partial(
                    sac_networks.make_sac_networks, **self.algo_params.network_factory
                )

            del self.algo_training_params["network_factory"]

        else:
            if self.algo == "ppo":
                network_factory = ppo_networks.make_ppo_networks
            else:
                network_factory = sac_networks.make_sac_networks

        self.algo_training_params["num_timesteps"] = self.num_timesteps
        print(f"{self.algo} params: {self.algo_training_params}")

        algo = ppo if self.algo == "ppo" else sac

        train_fn = functools.partial(
            algo.train,
            **self.algo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            # policy_params_fn=self.policy_params_fn,
            # restore_checkpoint_path=self.restore_checkpoint_path,
            checkpoint_logdir="./checkpoints/",
        )  # TODO SAC

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
