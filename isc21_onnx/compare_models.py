import numpy as np
import onnx
import onnxruntime as ort
import torch
from isc_feature_extractor import create_model

# Load original model
recommended_weight_name = "isc_ft_v107"
model, preprocessor = create_model(weight_name=recommended_weight_name, device="cpu")
model = model.eval()


# Load onnx model
onnx_model = onnx.load("isc_ft_v107.onnx")
onnx.checker.check_model(onnx_model)


# Compare with batch size == 8

random_tensor = torch.randn((8, 3, 512, 512)).float()


ort_session = ort.InferenceSession("isc_ft_v107.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: random_tensor.numpy()}
ort_outs = ort_session.run(None, ort_inputs)


with torch.no_grad():
    y = model(random_tensor)
    y = y.detach().cpu().numpy()


np.testing.assert_allclose(y, ort_outs[0], rtol=1e-03, atol=1e-05)


# Compare with batch size == 1

random_tensor = torch.randn((1, 3, 512, 512)).float()


ort_session = ort.InferenceSession("isc_ft_v107.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: random_tensor.numpy()}
ort_outs = ort_session.run(None, ort_inputs)


with torch.no_grad():
    y = model(random_tensor)
    y = y.detach().cpu().numpy()


np.testing.assert_allclose(y, ort_outs[0], rtol=1e-03, atol=1e-05)
