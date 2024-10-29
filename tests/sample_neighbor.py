import os
import sys

sys.path.insert(0, os.path.abspath("../"))

import onnx
import torch
import si4onnx
from tests import utils

utils.set_thread()
utils.set_seed(0)

# load dataset
input_x = torch.randn(1, 1, 128)

# load onnx model
model_path = "../tests/models/unet1d.onnx"
onnx_model = onnx.load(model_path)

# load si model
si_model = si4onnx.load(
    onnx_model,
    hypothesis=si4onnx.NeighborMeanDiff(
        threshold=0.95,
        neighborhood_range=20,
        i_idx=0,
        o_idx=0,
        post_process=[si4onnx.GaussianFilter(kernel_size=3, sigma=1.0)],
        use_norm=True,
    ),
)

# inference
with utils.timer("calculation time"):
    oc_result = si_model.inference(input_x, var=1.0, inference_mode="over_conditioning")
    pp_result = si_model.inference(input_x, var=1.0, inference_mode="parametric")

print(f"naive p-value: {pp_result.naive_p_value()}")
print(f"oc p-value: {oc_result.p_value}")
print(f"selective p-value: {pp_result.p_value}")
