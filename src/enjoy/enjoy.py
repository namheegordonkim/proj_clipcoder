from argparse import ArgumentParser

import torch
from PIL import Image
from clip import clip
from clip.model import CLIP
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

    clip_model: CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    input_text = "a man eating pie"
    input_tokens = clip.tokenize([input_text]).to("cuda")
    input_image = Image.open(f"{proj_dir}/data/examples/cat.jpg")

    fig = plt.figure()
    plt.imshow(np.asarray(input_image))
    plt.show()
    fig.canvas.draw()
    img_np = np.array(fig.canvas.buffer_rgba())
    writer.add_image(f"TestCatImage", img_np, 0, dataformats="HWC")
    plt.close()

    input_processed_image = preprocess(input_image)[None].to("cuda")

    input_token_embedding = clip_model.encode_text(input_tokens)
    input_image_embedding = clip_model.encode_image(input_processed_image)
    similarity = torch.cosine_similarity(input_token_embedding, input_image_embedding)
    logger.info(similarity)

    # How similar is each text to the input text?
    example_texts = [
        "a photo of a cat",
        "midnight snack",
        "a boy biting food",
        "a winter tree",
    ]
    example_text_embeddings = clip_model.encode_text(
        clip.tokenize(example_texts).to("cuda")
    )
    similarities = torch.cosine_similarity(
        input_token_embedding, example_text_embeddings
    )
    argmax_similarities = similarities.argmax().item()
    logger.info(
        f"Most similar to ``{input_text}``: ``{example_texts[argmax_similarities]}``"
    )

    # How similar is each image to the input text?
    example_images = [
        Image.open(f"{proj_dir}/data/examples/cat.jpg"),
        Image.open(f"{proj_dir}/data/examples/cat2.png"),
        Image.open(f"{proj_dir}/data/examples/child.png"),
        Image.open(f"{proj_dir}/data/examples/two-dogs-0.png"),
        Image.open(f"{proj_dir}/data/examples/two-dogs-1.png"),
    ]
    example_image_embeddings = torch.cat(
        [
            clip_model.encode_image(preprocess(img)[None].to("cuda"))
            for img in example_images
        ],
        dim=0,
    )
    similarities = torch.cosine_similarity(
        input_token_embedding, example_image_embeddings
    )
    argmax_similarities = similarities.argmax().item()
    logger.info(f"Most similar to '{input_text}': this image")
    fig = plt.figure()
    plt.title(
        f"Most similar to '{input_text}' (similarity={similarities[argmax_similarities]:.2f})"
    )
    plt.imshow(np.asarray(example_images[argmax_similarities]))
    plt.show()
    fig.canvas.draw()
    img_np = np.array(fig.canvas.buffer_rgba())
    writer.add_image(f"ManEatingPie1", img_np, 0, dataformats="HWC")
    plt.close()

    # Exercise 1: Implement Google Image Search on http://images.cocodataset.org/zips/val2017.zip
    ...

    # Exercise 2: Implement Reverse Image Search
    ...


    # Just sample code for dumping figures into tensorboard
    fig = plt.figure()

    # Make visuals here
    ...

    # plt.show()  # Comment out once satisfied
    fig.canvas.draw()
    img_np = np.array(fig.canvas.buffer_rgba())
    writer.add_image(f"AssertiveTitle", img_np, 0, dataformats="HWC")
    plt.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--out_name", type=str, default="run")
    parser.add_argument(
        "--debug_yes", "-d", action="store_true"
    )
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
