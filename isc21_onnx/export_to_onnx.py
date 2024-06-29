import requests
import torch
from isc_feature_extractor import create_model
from model import ISCNet_
from PIL import Image

recommended_weight_name = "isc_ft_v107"
model, preprocessor = create_model(weight_name=recommended_weight_name, device="cpu")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
x = preprocessor(image).unsqueeze(0)

y = model(x)
assert y.shape == torch.Size([1, 256])


model = model.eval()

random_tensor = torch.randn((8, 3, 512, 512)).float()


model = ISCNet_(
    model.backbone, model.fc.out_features, model.p, model.eval_p, model.l2_normalize
)
model.load_state_dict(model.state_dict())
model = model.eval()


with torch.no_grad():
    torch.onnx.export(
        model,  # model being run
        random_tensor,  # model input (or a tuple for multiple inputs)
        "isc_ft_v107.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=14,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
        training=torch.onnx.TrainingMode.EVAL,
    )

torch.save(model.state_dict(), "isc_ft_v107.pth")
