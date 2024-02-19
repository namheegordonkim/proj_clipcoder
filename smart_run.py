import os
from argparse import ArgumentParser
from copy import deepcopy
from shutil import copytree

import numpy as np

"""
Usage: python smart_run.py [OPTIONS] script_path
Must be run from the project root.
"""

# SBATCH_OPTIONS = "--time=4:0:0 --mem=16gb"
# SBATCH_OPTIONS = "--time=4:0:0 --mem=16gb --gres=gpu:1 --constraint=\"pascal|volta\""
SBATCH_OPTIONS = '--time=4:0:0 --mem=16gb --gres=gpu:1 --constraint="pascal|volta|a100"'
# SBATCH_OPTIONS = "--time=96:0:0 --mem=32gb --gres=gpu:1 --constraint=\"a100\""
# SBATCH_OPTIONS = "--time=96:0:0 --mem=32gb --gres=gpu:1 --constraint=\"a100\""
# SBATCH_OPTIONS = "--time=4:0:0 --mem=16gb"


def main(args, remaining_args):
    # if args.debug_yes:
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
    # UNUSED
    proj_dir = os.getcwd()
    run_dir = os.path.join(proj_dir, "runs", args.run_name)
    src_dir = os.path.join(proj_dir, "runs", args.run_name, "src")

    print(remaining_args)
    python_cmd = f"PYTHONPATH=./src python {args.script_path} --run_name {args.run_name} {' '.join(remaining_args)}"
    if args.debug_yes:
        python_cmd += " --debug_yes"

    if args.sbatch_yes:
        bash_run_command_string = (
            f"#!/bin/bash\nsource ~/.bashrc\nmodule load cuda/11.7.0.lua\n{python_cmd}"
        )
    else:
        # bash_run_command_string = f"#!/bin/bash\ncd \"$(dirname $(realpath $0))\"\n{python_cmd}"
        bash_run_command_string = f"#!/bin/bash\n{python_cmd}"

    # now that we have the commands to run, we create the run directory within `runs`
    os.makedirs(run_dir, exist_ok=True)
    # if overwrite is specified, then we should replace the source
    # otherwise we don't copy because we want to preserve old behaviour (useful if this is a re-run or sanity check)
    if not os.path.exists(src_dir) or args.overwrite_yes:
        copytree("./src", src_dir, dirs_exist_ok=True)

    if not args.debug_yes:  # To preserve breakpoints
        # the main "job script" meant to be run by any worker
        os.chdir(run_dir)

    # Modification for sweeping
    if args.sweep is not None:
        # <arg>:<log/linear>:<start>:<end>:<num>
        arg, mode, start, end, num = args.sweep.split(":")
        start = float(start)
        end = float(end)
        num = int(num)
        if mode == "log10":
            sweep_list = np.logspace(np.log10(start), np.log10(end), num)
        elif mode == "log2":
            sweep_list = np.logspace(np.log2(start), np.log2(end), num, base=2)
        elif mode == "linear":
            sweep_list = np.linspace(start, end, num)
        else:
            raise ValueError(f"Unknown mode {mode}")
        print(f"Sweeping {arg} from {start} to {end} with {num} points. Mode: {mode}")
        for sweep_val in sweep_list:
            sweep_val = float(sweep_val)
            sweep_prefix = f"{arg}={sweep_val}"
            print(f"Writing bash command with {sweep_prefix}")
            sweep_specific_bash_run_command_string = deepcopy(bash_run_command_string)
            sweep_specific_bash_run_command_string += f" --{arg} {sweep_val}"
            sweep_specific_bash_run_command_string += (
                f" --out_name {args.run_name}_{sweep_prefix}"
            )
            sweep_specific_bash_run_command_path = f"{proj_dir}/runs/{args.run_name}/{args.run_name}_{sweep_prefix}_command.sh"
            with open(sweep_specific_bash_run_command_path, "w", newline="\n") as f:
                f.write(sweep_specific_bash_run_command_string)
            # depending on the running platform (sbatch, cerebro, local, etc.), create necessary aux scripts and submit
            if args.sbatch_yes:
                # SLURM doesn't need any additional management
                print("Running sbatch")
                os.system(
                    f"sbatch {SBATCH_OPTIONS} --job-name={args.run_name}_{sweep_prefix} {sweep_specific_bash_run_command_path}"
                )
            else:
                os.system(f"bash {sweep_specific_bash_run_command_path}")

    else:
        bash_run_command_path = f"{proj_dir}/runs/{args.run_name}/command.sh"
        bash_run_command_string += f" --out_name {args.run_name}"
        with open(bash_run_command_path, "w", newline="\n") as f:
            f.write(bash_run_command_string)

        # depending on the running platform (sbatch, cerebro, local, etc.), create necessary aux scripts and submit
        if args.sbatch_yes:
            # SLURM doesn't need any additional management
            print("Running sbatch")
            os.system(
                f"sbatch {SBATCH_OPTIONS} --job-name={args.run_name} {bash_run_command_path}"
            )
        else:
            os.system(f"bash {bash_run_command_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("script_path", type=str)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--sbatch_yes", "-s", action="store_true")
    parser.add_argument("--overwrite_yes", "-o", action="store_true")
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    parser.add_argument("--sweep", type=str)  # <arg>:<log/linear>:<start>:<end>:<num>
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
