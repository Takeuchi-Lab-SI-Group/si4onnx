import os
import sys

sys.path.insert(0, os.path.abspath("../"))

import numpy as np
import torch
import onnx

from sicore import SelectiveInferenceNorm
import si4onnx
from si4onnx.utils import thresholding
from si4onnx.operators import Abs
from tests import utils

utils.set_thread()
utils.set_seed(0)

rng = np.random.default_rng(0)
input_x = torch.from_numpy(rng.normal(size=(1, 1, 28, 28)))


class CustomSelectiveInferenceModel(si4onnx.SelectiveInferenceModel):
    def __init__(self, model, thr):
        super().__init__()
        self.si_model = si4onnx.NN(model)
        self.thr = torch.tensor(thr, dtype=torch.float64)

    def construct_hypothesis(self, X, var):
        self.shape = X.shape
        input_x = X
        output_x = self.si_model.forward(input_x)
        reconstruction_error = Abs().forward(output_x - input_x)
        error = reconstruction_error > self.thr

        selected_model = error.reshape(-1).int()
        self.selected_model = selected_model

        input_vec = input_x.reshape(-1).double()
        eta = (
            selected_model / torch.sum(selected_model)
            - (1 - selected_model) / torch.sum(1 - selected_model)
        ).double()
        self.si_calculator = SelectiveInferenceNorm(input_vec, var, eta, use_torch=True)

        assert not np.isnan(self.si_calculator.stat)

    def algorithm(self, a, b, z):
        x = a + b * z
        INF = torch.tensor(torch.inf).double()
        input_x = x.reshape(self.shape).double()
        input_a = a.reshape(self.shape)
        input_b = b.reshape(self.shape)
        l = -INF
        u = INF
        output_x, output_a, output_b, l, u = self.si_model.forward_si(
            input_x, input_a, input_b, l, u, z
        )

        error_x = input_x - output_x
        error_a = input_a - output_a
        error_b = input_b - output_b

        error_x, error_a, error_b, l, u = Abs().forward_si(
            error_x, error_a, error_b, l, u, z
        )

        selected_model, l, u = thresholding(
            self.thr, error_x, error_a, error_b, l, u, z
        )
        return selected_model, [l, u]

    def model_selector(self, selected_model):
        return torch.all(torch.eq(self.selected_model, selected_model))


# load onnx model
model_path = "../tests/models/unet.onnx"
onnx_model = onnx.load(model_path)

# make si model
si_model = CustomSelectiveInferenceModel(model=onnx_model, thr=0.1)

# inference
with utils.timer("calculation time"):
    oc_result = si_model.inference(input_x, var=1.0, inference_mode="over_conditioning")
    pp_result = si_model.inference(input_x, var=1.0, inference_mode="parametric")

print(f"naive p-value: {pp_result.naive_p_value()}")
print(f"oc p-value: {oc_result.p_value}")
print(f"selective p-value: {pp_result.p_value}")
