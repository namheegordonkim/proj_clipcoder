from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir
import numpy as np
import matplotlib.pyplot as plt


def main(args, remaining_args):
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12346,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )

    # For Tensorboard logging
    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    writer = SummaryWriter(log_dir=logdir)
    writer.add_text("args", str(args.__dict__))
    writer.add_text("remaining_args", str(remaining_args))

    logger = my_logging.get_logger(f"{args.out_name}", logdir)
    logger.info(f"Starting")

    fig = plt.figure()

    # Make visuals here

    plt.show()  # Comment out once satisfied

    fig.canvas.draw()
    img_np = np.array(fig.canvas.buffer_rgba())
    writer.add_image(f"AssertiveTitle", img_np, 0, dataformats="HWC")

    plt.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )  # if set, will pause the program
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
