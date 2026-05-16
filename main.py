from __future__ import annotations

# Entry point for the Brain Tumor MRI classifier CLI.
# Supports training, evaluation, and single-image prediction.
import argparse
import json
import sys

from src.utils import config as cfg
from src.utils.logger import get_logger

log = get_logger("main")


# Parse CLI arguments for the main pipeline.
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Brain Tumor MRI Classification – CNN pipeline", )
    p.add_argument("--mode",
                   required=True,
                   choices=["train", "evaluate", "predict"])
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--image",
                   type=str,
                   help="Path to MRI image (required for --mode predict).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg.ensure_dirs()

    # Dispatch the selected pipeline mode.
    if args.mode == "train":
        from src.training.train import train
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    elif args.mode == "evaluate":
        from src.evaluation.evaluate import evaluate
        evaluate()

    elif args.mode == "predict":
        if not args.image:
            log.error("--image is required for predict mode")
            sys.exit(1)
        from src.evaluation.predict import predict_image
        result = predict_image(args.image)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
