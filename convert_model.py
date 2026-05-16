"""Convert a Keras .h5 model to ONNX using tf2onnx.

Usage:
    python convert_model.py --input saved_models/brain_tumor_cnn_best.h5 --output saved_models/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import tf2onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Keras .h5 model to ONNX format using tf2onnx.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input Keras .h5 model file.",
    )
    parser.add_argument(
        "--output",
        default="model.onnx",
        help="Path to the output ONNX file. Defaults to model.onnx.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version to use. Defaults to 13.",
    )
    return parser.parse_args()


def build_input_signature(model: tf.keras.Model) -> list[tf.TensorSpec]:
    inputs = model.inputs
    if not isinstance(inputs, list):
        inputs = [inputs]

    signature = []
    for input_tensor in inputs:
        tensor_shape = input_tensor.shape
        tensor_dtype = input_tensor.dtype
        tensor_name = input_tensor.name.split(":")[0]
        signature.append(
            tf.TensorSpec(shape=tensor_shape,
                          dtype=tensor_dtype,
                          name=tensor_name))
    return signature


def convert_h5_to_onnx(h5_path: Path, onnx_path: Path, opset: int) -> None:
    print(f"Loading Keras model from: {h5_path}")
    model = tf.keras.models.load_model(str(h5_path))
    input_signature = build_input_signature(model)

    print(f"Converting model to ONNX with opset {opset}...")
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(onnx_path),
    )
    print(f"Saved ONNX model to: {onnx_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert_h5_to_onnx(input_path, output_path, args.opset)


if __name__ == "__main__":
    main()
