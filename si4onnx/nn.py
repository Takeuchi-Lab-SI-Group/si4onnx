import numpy as np
import torch
from onnx import ModelProto, numpy_helper
from .utils import to_torch_tensor
from .layers import (
    Relu,
    LeakyRelu,
    Sigmoid,
    Conv,
    Gemm,
    MaxPool,
    AveragePool,
    GlobalAveragePool,
    ConvTranspose,
    Transpose,
    Shape,
    Slice,
    Exp,
    Flatten,
    ConstantOfShape,
    EyeLike,
    Reciprocal,
    Reshape,
    Resize,
    Concat,
    Add,
    Sub,
    Split,
    BatchNormalization,
    Mul,
    MatMul,
    Div,
    ReduceSum,
    Equal,
    Greater,
    Squeeze,
    Constant,
    RandomNormalLike,
)


class NN(torch.nn.Module):
    """Deep learning model class for Selective Inference.

    Methods
    -------
    __init__(model, seed=None, memoization=True)
        Initialize the class.
    forward(input)
        Forward pass of the model.
    forward_si(input, a, b, l, u, z)
        Forward pass with computing the output and the trancated interval.
    """

    def __init__(
        self, model: ModelProto, seed: int | None = None, memoization: bool = True
    ):
        """
        Parameters
        ----------
        model : onnx.ModelProto
            ONNX model
        seed : int | None, optional
            Random seed.
            If the model contains layers that generate random numbers,
            the seed is used to generate the same random numbers.
            Default is None.
            E.g., RandomNormalLike
        memoization : bool, optional
            Whether to use memoization for the forward_si method.
            If True, the memoization is enabled. Default is True.
        """
        super(NN, self).__init__()
        self.model = model
        self.seed = seed
        self.memoization = memoization
        self.is_memoization_initialized = True
        self.output_name_set = set(output.name for output in self.model.graph.output)

        # Available layers
        self.layers = {
            "Relu": Relu,
            "LeakyRelu": LeakyRelu,
            "Sigmoid": Sigmoid,
            "Conv": Conv,
            "Gemm": Gemm,
            "MaxPool": MaxPool,
            "AveragePool": AveragePool,
            "GlobalAveragePool": GlobalAveragePool,
            "ConvTranspose": ConvTranspose,
            "Transpose": Transpose,
            "Shape": Shape,
            "Slice": Slice,
            "Exp": Exp,
            "Flatten": Flatten,
            "ConstantOfShape": ConstantOfShape,
            "EyeLike": EyeLike,
            "Reciprocal": Reciprocal,
        }
        self.multi_input_layers = {
            "Reshape": Reshape,
            "Resize": Resize,
            "Concat": Concat,
            "Add": Add,
            "Sub": Sub,
            "Split": Split,
            "BatchNormalization": BatchNormalization,
            "Mul": Mul,
            "MatMul": MatMul,
            "Div": Div,
            "ReduceSum": ReduceSum,
            "Equal": Equal,
            "Greater": Greater,
            "Squeeze": Squeeze,
        }
        self.non_input_layers = {
            "Constant": Constant,
        }
        self.random_layers = {"RandomNormalLike": RandomNormalLike}

    @staticmethod
    def _calculate_output_x(a, b, z):
        if isinstance(a, torch.Tensor) and b is not None:
            return a + b * z
        else:
            return a  # constant variable is equal to a

    def _search_start_node(self, z, output, output_si):
        """Search the start node for the forward_si method.

        Parameters
        ----------
        z : float
        output : dict
            output (x) tensor dictionary
        output_si : dict
            output_si (a, b, l, u) dictionary
        """
        output_layers_cnt = 0
        graph_size = len(self.model.graph.node)

        for node_index, node in enumerate(reversed(self.model.graph.node)):
            if node.output[0] in self.output_name_set:
                output_layers_cnt += 1

            if output_layers_cnt < len(self.model.graph.output):
                continue

            _a, _b, _l, _u = output_si[node.output[0]]
            if _b is not None and (not isinstance(_a, list)) and _l < z < _u:
                start_node_index = graph_size - node_index

                if output_layers_cnt == len(self.model.graph.output):
                    return start_node_index, output.copy(), output_si.copy()

    def forward(self, input):
        """Forward pass of the model.

        Parameters
        ----------
        input : torch.Tensor | list[torch.Tensor]

        Returns
        -------
        output : torch.Tensor | list[torch.Tensor]
        """
        self.rng = np.random.default_rng(self.seed)

        node_output = dict()
        for i, input_node in enumerate(self.model.graph.input):
            if len(self.model.graph.input) == 1:
                node_output[input_node.name] = to_torch_tensor(input)
            else:
                node_output[input_node.name] = to_torch_tensor(input[i])

        for tensor in self.model.graph.initializer:
            arr = numpy_helper.to_array(tensor)
            if tensor.data_type == 7:
                arr = torch.tensor(arr).long()
            else:
                arr = torch.tensor(arr).double()

            node_output[tensor.name] = arr

        with torch.no_grad():
            for node in self.model.graph.node:
                inputs = [
                    node_output[input_name]
                    for input_name in node.input
                    if input_name != ""
                ]
                op_type = node.op_type
                if op_type in self.layers:
                    layer = self.layers[op_type](inputs, node)
                    x = node_output[node.input[0]]
                    outputs = layer.forward(x)
                elif op_type in self.multi_input_layers:
                    layer = self.multi_input_layers[op_type](inputs, node, node_output)
                    outputs = layer.forward()
                elif op_type in self.non_input_layers:
                    layer = self.non_input_layers[op_type](inputs, node)
                    outputs = layer.forward()
                elif op_type in self.random_layers:
                    layer = self.random_layers[op_type](inputs, node)
                    outputs = layer.forward(node.input[0], self.rng)
                else:
                    raise NotImplementedError(f"Layer {op_type} is not supported.")

                if isinstance(outputs, torch.Tensor) or op_type == "Constant":
                    node_output[node.output[0]] = outputs
                else:
                    for i, output_name in enumerate(node.output):
                        node_output[output_name] = outputs[i]

        self.output_obs = node_output
        outputs = [node_output[output.name] for output in self.model.graph.output]

        self.is_memoization_initialized = True

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def forward_si(self, input, a, b, l, u, z):
        """Forward pass with computing the trancated interval of the output.

        Parameters
        ----------
        input : torch.Tensor | list[torch.Tensor]
            input tensor or tensor list in the deep learning model
        a : torch.Tensor | list[torch.Tensor]
            a tensor or tensor list
        b : torch.Tensor | list[torch.Tensor]
            b tensor or tensor list
        l : torch.Tensor | list[torch.Tensor]
            l tensor or tensor list
        u : torch.Tensor | list[torch.Tensor]
            u tensor or tensor list
        z : float

        Returns
        -------
        x : torch.Tensor | list[torch.Tensor]
            output tensor or tensor list in the deep learning model
        a : torch.Tensor | list[torch.Tensor]
            output a tensor or tensor list
        b : torch.Tensor | list[torch.Tensor]
            output b tensor or tensor list
        l : torch.Tensor | list[torch.Tensor]
            lower bound of the truncated interval
        u : torch.Tensor | list[torch.Tensor]
            upper bound of the truncated interval
        """
        node_output = dict()
        node_output_si = dict()
        INF = torch.tensor(torch.inf).double()

        if not self.is_memoization_initialized:
            if hasattr(self, "output"):
                node_output = self.output.copy()
                node_output_si = self.output_si.copy()
            elif hasattr(self, "_output"):
                node_output = self._output.copy()
                node_output_si = self._output_si.copy()
        else:
            for tensor in self.model.graph.initializer:
                arr = numpy_helper.to_array(tensor)
                if tensor.data_type == 7:
                    arr = torch.tensor(arr).long()
                else:
                    arr = torch.tensor(arr).double()

                node_output[tensor.name] = arr
                node_output_si[tensor.name] = (arr, None, -INF, INF)

        for i, input_node in enumerate(self.model.graph.input):
            if len(self.model.graph.input) == 1:
                node_output[input_node.name] = to_torch_tensor(input)
                node_output_si[input_node.name] = (
                    to_torch_tensor(a),
                    to_torch_tensor(b),
                    to_torch_tensor(l),
                    to_torch_tensor(u),
                )
            else:
                node_output[input_node.name] = to_torch_tensor(input[i])
                if a[i] is not None and b[i] is not None:
                    node_output_si[input_node.name] = (
                        to_torch_tensor(a[i]),
                        to_torch_tensor(b[i]),
                        to_torch_tensor(l[i]),
                        to_torch_tensor(u[i]),
                    )
                else:
                    node_output_si[input_node.name] = (
                        to_torch_tensor(input[i]),
                        None,
                        -INF,
                        INF,
                    )

        # Find the start node
        start_node_index = 0
        if self.memoization and not self.is_memoization_initialized:
            if z > 0 and hasattr(self, "output"):
                start_node_index, node_output, node_output_si = self._search_start_node(
                    z, self.output, self.output_si
                )
            elif hasattr(self, "_output"):
                start_node_index, node_output, node_output_si = self._search_start_node(
                    z, self._output, self._output_si
                )

        with torch.no_grad():
            for node in self.model.graph.node[start_node_index:]:
                op_type = node.op_type
                inputs = [
                    node_output[input_name]
                    for input_name in node.input
                    if input_name != ""
                ]
                if op_type in self.layers:
                    layer = self.layers[op_type](inputs, node)
                    a, b, l, u = layer.forward_si(*node_output_si[node.input[0]], z)
                elif op_type in self.multi_input_layers:
                    layer = self.multi_input_layers[op_type](inputs, node, node_output)
                    a, b, l, u = layer.forward_si(node, node_output, node_output_si)
                elif op_type in self.non_input_layers:
                    layer = self.non_input_layers[op_type](inputs, node)
                    a, b, l, u = layer.forward_si()
                elif op_type in self.random_layers:
                    layer = self.random_layers[op_type](inputs, node)
                    a, b, l, u = layer.forward_si(self.output_obs[node.output[0]], z)
                else:
                    raise NotImplementedError(f"Layer {op_type} is not supported.")

                if isinstance(a, torch.Tensor) or op_type == "Constant":
                    assert l < u
                    node_output[node.output[0]] = self._calculate_output_x(a, b, z)
                    node_output_si[node.output[0]] = (a, b, l, u)
                else:
                    for i, output_name in enumerate(node.output):
                        assert l[i] < u[i]
                        node_output[output_name] = self._calculate_output_x(
                            a[i], b[i], z
                        )
                        node_output_si[output_name] = (a[i], b[i], l[i], u[i])
                    u = u[-1]

        if u > 0:
            self.output = node_output
            self.output_si = node_output_si
        else:
            self._output = node_output
            self._output_si = node_output_si

        self.is_memoization_initialized = False

        x, output_a, output_b, l, u = zip(
            *[
                [
                    self._calculate_output_x(
                        node_output_si[output.name][0],
                        node_output_si[output.name][1],
                        z,
                    ),
                    node_output_si[output.name][0],
                    node_output_si[output.name][1],
                    node_output_si[output.name][2],
                    node_output_si[output.name][3],
                ]
                for output in self.model.graph.output
            ]
        )

        if len(x) == 1:
            return x[0], output_a[0], output_b[0], l[0], u[0]
        else:
            return x, output_a, output_b, l, u
