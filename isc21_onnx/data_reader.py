import os
import random

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image as PIL_Image


def _preprocess_images_nested(
    root_folder: str, height: int, width: int, size_limit=10000
):
    """
    Loads a batch of images from nested folders and preprocesses them.
    parameter root_folder: path to root folder storing nested class folders
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 10000.
    return: list of matrices characterizing multiple images
    """
    class_folders = [
        os.path.join(root_folder, class_folder)
        for class_folder in os.listdir(root_folder)
    ]
    image_files = []

    for class_folder in class_folders:
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                if image_name.endswith(".jpeg"):
                    image_files.append(os.path.join(class_folder, image_name))

    if size_limit > 0 and len(image_files) > size_limit:
        image_files = random.sample(image_files, size_limit)

    batch_data = []

    for image_filepath in image_files:
        img = PIL_Image.open(image_filepath).convert("RGB")
        img = img.resize((height, width), PIL_Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, 0)
        batch_data.append(img_np)

    batch_data = np.concatenate(np.expand_dims(batch_data, axis=0), axis=0)
    print(batch_data.shape)
    return batch_data


class DataReader(CalibrationDataReader):
    def __init__(
        self, calibration_image_folder: str, model_path: str, nb_images: int = 10000
    ):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, channels, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nchw_data_list = _preprocess_images_nested(
            calibration_image_folder, height, width, size_limit=nb_images
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nchw_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nchw_data} for nchw_data in self.nchw_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
