import argparse
import time

import data_reader
import numpy as np
import onnxruntime
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 512, 512), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--calibrate_dataset",
        default="../calibration_dataset",
        help="calibration data set",
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    # parser.add_argument("--per_channel", default=False, type=bool)
    parser.add_argument("--per_channel", action="store_true")
    parser.add_argument("--nb_images", default=100, type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset
    nb_images = args.nb_images
    dr = data_reader.DataReader(calibration_dataset_path, input_model_path, nb_images)
    print("DataReader created, starting calibration...")

    # Calibrate and quantize model
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul"],
        calibrate_method=CalibrationMethod.Percentile,
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path)

    print("benchmarking int8 model...")
    benchmark(output_model_path)


if __name__ == "__main__":
    main()
