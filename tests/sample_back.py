import os
import sys

sys.path.insert(0, os.path.abspath("../"))

import numpy as np
import torch
import onnx
import si4onnx
from si4onnx import BackMeanDiff, InputDiff, Abs, GaussianFilter
from tests import utils

utils.set_thread()
utils.set_seed(0)

# load onnx model
model_path = "../tests/models/unet.onnx"
onnx_model = onnx.load(model_path)

size = 64
rng = np.random.default_rng(0)
X = torch.from_numpy(rng.normal(size=(1, 1, size, size)))

# load si model
si_model = si4onnx.load(
    model=onnx_model,
    hypothesis=BackMeanDiff(
        threshold=0.95,
        post_process=[InputDiff(), Abs(), GaussianFilter(kernel_size=3, sigma=1.0)],
        use_norm=True,
    ),
    seed=0,
    # memoization=False,
)

# inference
with utils.timer("calculation time"):
    oc_result = si_model.inference(X, var=1.0, inference_mode="over_conditioning")
    pp_result = si_model.inference(X, var=1.0, inference_mode="parametric")

print(f"naive p-value: {pp_result.naive_p_value()}")
print(f"oc p-value: {oc_result.p_value}")
print(f"selective p-value: {pp_result.p_value}")
