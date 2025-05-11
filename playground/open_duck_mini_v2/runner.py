"""Runs training and evaluation loop for Open Duck Mini V2."""

import argparse

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import joystick, standing

from playground.common.export_onnx import export_onnx
from orbax import checkpoint as ocp


class OpenDuckMiniV2Runner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task)
        self.eval_env = self.env_file[1](task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        if args.algo == "ppo":
            self.obs_size = int(
                self.env.observation_size["state"][0]
            )  # 0: state 1: privileged_state
        elif args.algo == "sac":
            self.obs_size = int(
                self.env.observation_size
            )  # 0: state 1: privileged_state
        else:
            raise ValueError(f"Unknown algo {args.algo}")
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")

        # TODO specific for sac for now
        if args.export_onnx is not None:
            print()
            print()
            print()
            print(f"EXPORTING {args.export_onnx} ONNX")

            pth_path = args.export_onnx

            from brax.io.model import load_params

            params = load_params(pth_path)
            mean = params[0].mean
            std = params[0].std
            policy_params = params[1]["params"]
            export_onnx(
                mean,
                std,
                policy_params,
                self.action_size,
                self.algo_params,
                self.obs_size,
                ppo=True if self.algo == "ppo" else False,
            )
            exit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    # parser.add_argument("--num_timesteps", type=int, default=300000000)
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        help="Algorithm to use for training (ppo or sac)",
    )
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    parser.add_argument(
        "--export_onnx",
        type=str,
        default=None,
        required=False,
        help="Export .pth to ONNX format",
    )
    args = parser.parse_args()

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
