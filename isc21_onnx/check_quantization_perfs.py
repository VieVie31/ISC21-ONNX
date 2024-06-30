import numpy as np
import onnxruntime
from PIL import Image as PIL_Image
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_image(image_path: str, height: int, width: int):
    """
    Preprocesses a single image.
    parameter image_path: path to the image
    parameter height: image height in pixels
    parameter width: image width in pixels
    return: preprocessed image array
    """
    img = PIL_Image.open(image_path).convert("RGB")
    img = img.resize((height, width), PIL_Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, 0)
    return img_np


def calculate_mse(output1, output2):
    """
    Calculate the Mean Squared Error between two outputs.
    """
    return np.mean((output1 - output2) ** 2)


def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.
    """
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


def load_model_and_get_input_shape(model_path: str):
    """
    Load the ONNX model and get input shape.
    """
    session = onnxruntime.InferenceSession(model_path, None)
    (_, channels, height, width) = session.get_inputs()[0].shape
    input_name = session.get_inputs()[0].name
    return session, input_name, (height, width)


def run_inference(session, input_name, input_data):
    """
    Run inference on the model session.
    """
    input_feed = {input_name: input_data}
    output = session.run(None, input_feed)
    return output[0]  # Assuming single output


def main(original_model_path: str, quantized_model_path: str, sample_image_path: str):
    # Load models and get input shapes
    original_session, original_input_name, input_shape = load_model_and_get_input_shape(
        original_model_path
    )
    quantized_session, quantized_input_name, _ = load_model_and_get_input_shape(
        quantized_model_path
    )

    # Preprocess sample image
    input_data = preprocess_image(sample_image_path, *input_shape)

    # Run inference on both models
    original_output = run_inference(original_session, original_input_name, input_data)
    quantized_output = run_inference(
        quantized_session, quantized_input_name, input_data
    )

    original_output = original_output / np.linalg.norm(original_output)
    quantized_output = quantized_output / np.linalg.norm(quantized_output)

    print("Original output", original_output)
    print("Quantized output", quantized_output)

    # Calculate MSE
    mse = calculate_mse(original_output, quantized_output)
    print(f"Mean Squared Error (MSE): {mse}")

    # Calculate cosine similarity for embeddings
    cosine_sim = calculate_cosine_similarity(original_output, quantized_output)
    print(f"Cosine Similarity: {cosine_sim}")


if __name__ == "__main__":
    # Paths to the original and quantized models and a sample image for testing
    original_model_path = "isc_ft_v107.onnx"
    quantized_model_path = "isc_ft_v107_quant_120.onnx"
    sample_image_path = "test.jpg"

    main(original_model_path, quantized_model_path, sample_image_path)
