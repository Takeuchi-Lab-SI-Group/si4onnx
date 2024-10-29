import os
import sys

sys.path.insert(0, os.path.abspath("../"))


import numpy as np
import onnx
import torch
import si4onnx
from si4onnx.hypothesis import ReferenceMeanDiff
from tests import utils

utils.set_thread()
utils.set_seed(0)


size = 16
rng = np.random.default_rng(0)
x = torch.from_numpy(rng.normal(size=(1, 1, size, size)))
x_ref = torch.from_numpy(rng.normal(size=(1, 1, size, size)))

# load onnx model
model_path = "../tests/models/unet.onnx"
onnx_model = onnx.load(model_path)

# load si model
si_model = si4onnx.load(
    onnx_model,
    hypothesis=ReferenceMeanDiff(
        threshold=0.8,
        i_idx=0,
        o_idx=0,
        use_norm=True,
    ),
)


# inference
with utils.timer("calculation time"):
    oc_result = si_model.inference(
        (x, x_ref), var=1.0, inference_mode="over_conditioning"
    )
    pp_result = si_model.inference((x, x_ref), var=1.0, inference_mode="parametric")


print("naive p-value:", pp_result.naive_p_value())
print(f"oc p-value: {oc_result.p_value}")
print(f"selective p-value: {pp_result.p_value}")
