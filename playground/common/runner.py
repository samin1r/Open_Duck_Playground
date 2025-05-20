"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax

from playground.common.export_onnx import export_onnx

# Check if CUDA is available, otherwise use CPU
if not jax.config.jax_enable_x64:
    try:
        # Try to initialize CUDA
        jax.devices('gpu')
        print("Using GPU for training")
    except:
        print("No GPU found, using CPU for training (this will be slow)")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        print(f"[BASE_RUNNER INIT] num_envs={args.num_envs}, batch_size={args.batch_size}")
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
        
        # Add a flag to control ONNX export during training
        self.export_onnx_during_training = getattr(args, 'export_onnx_during_training', False)
        
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
        # Log to TensorBoard every 1K steps (instead of 10K)
        if num_steps % 1000 == 0 or num_steps == 0:
            for metric_name, metric_value in metrics.items():
                # Convert to float, but watch out for 0-dim JAX arrays
                self.writer.add_scalar(metric_name, metric_value, num_steps)
            
            # Print to console every 1K steps
            print("-----------")
            print(
                f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}'
            )
            print("-----------")
        
        # For more frequent console updates without TensorBoard logging
        elif num_steps % 100 == 0:  # Print basic info every 100 steps
            print(f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]}')

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        
        # Only export ONNX if explicitly enabled
        if self.export_onnx_during_training:
            onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
            try:
                print("Exporting ONNX model to CPU...")
                export_onnx(
                    params,
                    self.action_size,
                    self.ppo_params,
                    self.obs_size,
                    output_path=onnx_export_path,
                    use_cpu=True
                )
            except Exception as e:
                print(f"ONNX export failed: {e}")
                print("Continuing training without ONNX export")
        
        # Save the last parameters for final export
        self.last_params = params
        self.last_step = current_step

    def train(self) -> None:
        print(f"[TRAIN START] num_envs={self.args.num_envs}, batch_size={self.args.batch_size}")
        
        print(f"[BEFORE PPO INIT] num_envs={self.args.num_envs}, batch_size={self.args.batch_size}")
        
        self.ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")

        # Override with your custom values
        self.ppo_params["batch_size"] = self.args.batch_size
        self.ppo_params["num_envs"] = self.args.num_envs
        self.ppo_params["num_timesteps"] = self.args.num_timesteps
        # Now create the training params
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20
        
        print(f"[PPO PARAMS] {self.ppo_params}")

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        
        print(f"[AFTER PPO INIT] Agent created with batch_size={self.args.batch_size}, num_envs={self.args.num_envs}")
        
        # Export ONNX at the end of training
        try:
            print("\n=== Exporting final ONNX model to CPU ===")
            d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            onnx_export_path = f"{self.output_dir}/final_{d}.onnx"
            
            # Force CPU usage for final export
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide GPU from TensorFlow
            
            export_onnx(
                params,
                self.action_size,
                self.ppo_params,
                self.obs_size,
                output_path=onnx_export_path,
                use_cpu=True
            )
            print(f"Final ONNX model exported to: {onnx_export_path}")
        except Exception as e:
            print(f"Final ONNX export failed: {e}")
