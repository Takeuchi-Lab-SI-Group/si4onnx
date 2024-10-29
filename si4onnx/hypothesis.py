from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sicore import SelectiveInferenceNorm
from .utils import to_torch_tensor, thresholding
from .nn import NN


def check_use_sigmoid(si_model, selected_node_name):
    """
    Parameters
    ----------
    si_model : NN
        The model for the selective inference.
    selected_node_name : str
        The name of the selected node.

    Returns
    -------
    bool
        True if the selected node is a Sigmoid.
    """
    o2i = {}
    i2o = {}
    target_node = None

    for node in si_model.model.graph.node:
        if selected_node_name in node.output:
            target_node = node.input
        for input_name in node.input:
            o2i[input_name] = {"name": node.output, "op_type": node.op_type}
        for output_name in node.output:
            i2o[output_name] = {"name": node.input, "op_type": node.op_type}

    queue = list(target_node)
    visited = set(target_node)

    while queue:
        current_node_output = queue.pop(0)

        # Check if the current node is a Sigmoid
        current_node = o2i.get(current_node_output)
        if current_node and current_node["op_type"] == "Sigmoid":
            return True

        # Add unvisited inputs of the current node to the queue
        if current_node:
            for input_name in i2o.get(current_node_output, {}).get("name", []):
                if input_name not in visited:
                    queue.append(input_name)
                    visited.add(input_name)
    return False


class Hypothesis(ABC):
    @abstractmethod
    def construct_hypothesis(self, X, si_model, *kwargs):
        pass

    @abstractmethod
    def model_selector(self, roi_vector, **kwargs):
        pass

    @abstractmethod
    def algorithm(self, a, b, z, **kwargs):
        pass


class NoHypothesisError(Exception):
    """If the hypothesis is not obtained from observartion, please raise this error"""

    def __str__(self):
        return "Hypothesis is not obtained."


class PresetHypothesis(Hypothesis):
    def __init__(
        self,
        threshold: float,
        i_idx: int = 0,
        o_idx: int = 0,
        post_process: nn.Module | list[nn.Module] = [],
        use_norm: bool = False,
        **kwargs,
    ):
        """Base class for preset hypotheses in selective inference.

        This class serves as a foundation for implementing various preset hypotheses
        testing settings. It provides common functionality for constructing regions
        of interest (ROI) from model outputs and handling both single and multiple
        input/output scenarios.

        Attributes
        ----------
        thrshold : torch.Tensor
            Threshold value for ROI determination.
        i_idx : int
            Selected input index.
        o_idx : int
            Selected output index.
        post_process : list[nn.Module]
            List of post-processing operations.
        use_norm : bool
            Flag for score map normalization.
        use_sigmoid : bool
            Flag indicating if sigmoid is used in output.
        mask : torch.Tensor or None
            Optional mask for ROI calculation.
        """
        super().__init__(**kwargs)
        self.thrshold = torch.tensor(threshold).double()
        self.i_idx = i_idx
        self.o_idx = o_idx
        self.post_process = (
            post_process if isinstance(post_process, (list)) else [post_process]
        )
        self.use_norm = use_norm
        self.use_sigmoid = False

    def _construct_roi_vector(
        self, si_model: NN, X: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct the `roi` and `roi_vector` from the input data.

        Parameters
        ----------
        si_model : NN
            The model for the selective inference.
        X : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor])
            The input data.

        Returns
        -------
        input_x : torch.Tensor
            The input tensor.
        roi : torch.Tensor
            The region of interest.
        roi_vector : torch.Tensor
            The flattened roi.
        """
        # Check if the selected output is using Sigmoid
        selected_output_name = si_model.model.graph.output[self.o_idx].name
        self.use_sigmoid = check_use_sigmoid(si_model, selected_output_name)
        if self.use_sigmoid and (not (not self.use_norm or self.post_process == [])):
            raise ValueError(
                "use_sigmoid is True, so cannot specify use_norm or post_process."
            )

        if isinstance(X, (list)):
            list_x = X
        elif isinstance(X, (tuple)):
            list_x = list(X)
        else:
            list_x = [X]

        self.num_inputs = len(list_x)
        self.shape = list_x[self.i_idx].shape
        input_x = to_torch_tensor(list_x[self.i_idx])
        self.saved_inputs = list_x

        # forward process
        output = si_model.forward(X)
        self.output = output

        # Get the output
        if not isinstance(output, (tuple, list)):
            output_x = output
        else:
            output_x = output[self.o_idx]

        # Apply post-process
        score_map = output_x
        if not self.use_sigmoid:
            for operator in self.post_process:
                if operator.__class__.__name__ == "InputDiff":
                    score_map = operator.forward(score_map, input_x)
                else:
                    score_map = operator.forward(score_map)
            if self.use_norm:
                score_max = torch.max(score_map)
                score_min = torch.min(score_map)
                score_map = (score_map - score_min) / (score_max - score_min)

        score_map = score_map.squeeze()
        self.score_map = score_map

        roi = (score_map > self.thrshold).int()

        # Apply mask
        if self.mask is not None:
            roi = roi.logical_and(self.mask).int()
        self.roi = roi

        roi_vector = roi.flatten().int()
        self.roi_vector = roi_vector

        return input_x, roi, roi_vector

    def _base_algorithm(self, si_model, a, b, z):
        x = a + b * z
        INF = torch.tensor(torch.inf).double()
        input_x = self.saved_inputs
        input_a = [None] * self.num_inputs
        input_b = [None] * self.num_inputs
        input_l = [-INF] * self.num_inputs
        input_u = [INF] * self.num_inputs

        input_x[self.i_idx] = x.reshape(self.shape).double()
        input_a[self.i_idx] = a.reshape(self.shape).double()
        input_b[self.i_idx] = b.reshape(self.shape).double()
        input_l[self.i_idx] = -INF
        input_u[self.i_idx] = INF

        output_x, output_a, output_b, l, u = si_model.forward_si(
            input_x if self.num_inputs > 1 else input_x[self.i_idx],
            input_a if self.num_inputs > 1 else input_a[self.i_idx],
            input_b if self.num_inputs > 1 else input_b[self.i_idx],
            input_l if self.num_inputs > 1 else input_l[self.i_idx],
            input_u if self.num_inputs > 1 else input_u[self.i_idx],
            z,
        )

        if not isinstance(output_x, (list, tuple)):
            output_x = [output_x]
            output_a = [output_a]
            output_b = [output_b]
        else:
            l = l[self.o_idx]
            u = u[self.o_idx]

        score_map_x = output_x[self.o_idx]
        score_map_a = output_a[self.o_idx]
        score_map_b = output_b[self.o_idx]

        if not self.use_sigmoid:
            for operator in self.post_process:
                if operator.__class__.__name__ == "InputDiff":
                    score_map_x, score_map_a, score_map_b, l, u = operator.forward_si(
                        score_map_x,
                        score_map_a,
                        score_map_b,
                        l,
                        u,
                        z,
                        input_x[self.i_idx],
                        input_a[self.i_idx],
                        input_b[self.i_idx],
                    )
                else:
                    score_map_x, score_map_a, score_map_b, l, u = operator.forward_si(
                        score_map_x,
                        score_map_a,
                        score_map_b,
                        l,
                        u,
                        z,
                    )

        roi_vector, l, u = thresholding(
            self.thrshold,
            score_map_x,
            score_map_a,
            score_map_b,
            l,
            u,
            z,
            use_sigmoid=self.use_sigmoid,
            use_norm=self.use_norm,
        )

        # Apply mask
        if self.mask is not None:
            roi_vector = roi_vector.logical_and(self.mask.reshape(-1)).int()

        return roi_vector, [l, u]


class BackMeanDiff(PresetHypothesis):
    def __init__(
        self,
        threshold: float,
        i_idx: int = 0,
        o_idx: int = 0,
        post_process: nn.Module | list[nn.Module] = [],
        use_norm: bool = False,
        **kwargs,
    ):
        """Hypothesis for the mean difference between the ROI and the background region in the input data.

        Parameters
        ----------
        threshold : float
            Threshold value for `roi`.
        i_idx : int
            The index of the input to use for the test statistic.
            This option is for models with multiple inputs.
            Defaults to 0.
        o_idx : int
            The index of the output to use for the calculation of the `roi`.
            This option is for models with multiple outputs.
            Defaults to 0.
        use_norm : bool, optional
            Whether to apply min-max normalization to the `score_map`.
        """
        super().__init__(threshold, i_idx, o_idx, post_process, use_norm, **kwargs)

    def construct_hypothesis(
        self,
        si_model: NN,
        X: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
        var: float | torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        si_model : NN
            The model for the selective inference.
        X : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
            The input data.
        var : float | torch.Tensor | np.ndarray
            The variance of the noise.
        mask : torch.Tensor | numpy.ndarray | None, optional
            The mask can be used to specify the region that may be used for the hypothesis test.
            The mask to apply the logical AND operator to the `roi` and `non_roi`.
            Defaults to None.
        """
        self.mask = mask
        input_x, roi, roi_vector = self._construct_roi_vector(si_model, X)

        non_roi = 1 - roi
        if self.mask is not None:
            non_roi = non_roi.logical_and(self.mask).int()
        self.non_roi = non_roi

        non_roi_vector = non_roi.reshape(-1).int()
        self.non_roi_vector = non_roi_vector

        eta = (
            roi_vector / torch.sum(roi_vector)
            - (non_roi_vector) / torch.sum(non_roi_vector)
        ).double()

        input_vec = input_x.reshape(-1).double()
        self.si_calculator = SelectiveInferenceNorm(input_vec, var, eta, use_torch=True)

        if np.isnan(self.si_calculator.stat):
            raise NoHypothesisError

    def algorithm(
        self, si_model: NN, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, list]:
        """
        Parameters
        ----------
        si_model : NN
            The model for the selective inference.
        a : torch.Tensor
            The input tensor a.
        b : torch.Tensor
            The input tensor b.
        z : float
            The test statistic z.

        Returns
        -------
        roi_vector : torch.Tensor
            The flattened roi.
        [l, u] : list(float)
            The over-conditioning interval that contains z.
        """
        return self._base_algorithm(si_model, a, b, z)

    def model_selector(self, roi_vector: torch.Tensor) -> bool:
        """
        Parameters
        ----------
        roi_vector : torch.Tensor
            should be a tensor of int

        Returns
        -------
        bool
            True if the input roi_vector is the same as the roi_vector in the construct_hypothesis method
        """
        return torch.all(torch.eq(self.roi_vector, roi_vector))


class NeighborMeanDiff(PresetHypothesis):
    def __init__(
        self,
        threshold: float,
        neighborhood_range: int = 1,
        i_idx: int = 0,
        o_idx: int = 0,
        post_process: nn.Module | list[nn.Module] = [],
        use_norm: bool = False,
        **kwargs,
    ):
        """Hypothesis for the mean difference between the ROI and the neighborhood region in the input data.

        Parameters
        ----------
        threshold : float
            Threshold value for `roi`.
        neighborhood_range : int
            The range of the neighborhood region (`non_roi`).
            If the `roi` is True at (i, j),
            the neighborhood region is True at (i - neighborhood_range, j - neighborhood_range)
            to (i + neighborhood_range, j + neighborhood_range).
            Defaults to 1.
        i_idx : int
            The index of the input to use for the test statistic.
            This option is for models with multiple inputs.
            Defaults to 0.
        o_idx : int
            The index of the output to use for the calculation of the `roi`.
            This option is for models with multiple outputs.
            Defaults to 0.
        use_norm : bool, optional
            Whether to apply min-max normalization to the `score_map`.
        """
        super().__init__(threshold, i_idx, o_idx, post_process, use_norm, **kwargs)
        self.neighborhood_range = neighborhood_range

    def construct_hypothesis(
        self,
        si_model: NN,
        X: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
        var: float | torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        si_model : NN
            The model for the selective inference.
        X : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
            The input data.
        var: float | torch.Tensor | np.ndarray
            The variance of the noise.
        mask : torch.Tensor | numpy.ndarray | None, optional
            The mask can be used to specify the region that may be used for the hypothesis test.
            The mask to apply the logical AND operator to the `roi` and `non_roi`.
            Defaults to None.
        """
        self.mask = mask
        input_x, roi, roi_vector = self._construct_roi_vector(si_model, X)

        # set parameters for neighborhood region
        kernel_size = 2 * self.neighborhood_range + 1
        padding = self.neighborhood_range
        ndim = self.roi.dim()

        # compute the neighborhood region
        if ndim == 1:
            x_expanded = self.roi.unsqueeze(0).unsqueeze(0).float()
            neighborhood_region = F.max_pool1d(
                x_expanded, kernel_size=kernel_size, stride=1, padding=padding
            )
        elif ndim == 2:
            x_expanded = self.roi.unsqueeze(0).unsqueeze(0).float()
            neighborhood_region = F.max_pool2d(
                x_expanded, kernel_size=kernel_size, stride=1, padding=padding
            )

        neighborhood_region = neighborhood_region.squeeze()
        neighborhood_region = neighborhood_region.logical_xor(self.roi).int()
        if self.mask is not None:
            neighborhood_region = neighborhood_region.logical_and(self.mask).int()

        self.non_roi = neighborhood_region

        neighborhood_vector = neighborhood_region.reshape(-1).int()
        self.neighborhood_vector = neighborhood_vector

        input_vec = input_x.reshape(-1).double()
        eta = (
            roi_vector / torch.sum(roi_vector)
            - (neighborhood_vector) / torch.sum(neighborhood_vector)
        ).double()

        self.si_calculator = SelectiveInferenceNorm(input_vec, var, eta, use_torch=True)

        if np.isnan(self.si_calculator.stat):
            raise NoHypothesisError

    def algorithm(
        self, si_model: NN, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, list]:
        """
        Parameters
        ----------
        si_model : NN
            The model for the selective inference.
        a : torch.Tensor
            The input tensor a.
        b : torch.Tensor
            The input tensor b.
        z : float
            The test statistic z.

        Returns
        -------
        roi_vector : torch.Tensor
            The flattened roi.
        [l, u] : list(float)
            The over-conditioning interval that contains z.
        """
        return self._base_algorithm(si_model, a, b, z)

    def model_selector(self, roi_vector: torch.Tensor) -> bool:
        """
        Parameters
        ----------
        roi_vector : torch.Tensor
            should be a tensor of int
        Returns
        -------
        bool
            True if the input roi_vector is the same as the roi_vector in the construct_hypothesis method
        """
        return torch.all(torch.eq(self.roi_vector, roi_vector))


class ReferenceMeanDiff(PresetHypothesis):
    def __init__(
        self,
        threshold: float,
        i_idx: int = 0,
        o_idx: int = 0,
        post_process: nn.Module | list[nn.Module] = [],
        use_norm: bool = False,
        **kwargs,
    ):
        """Hypothesis for the mean difference between the ROI and the reference data in the input data.

        Parameters
        ----------
        threshold : float
            Threshold value for `roi`.
        reference_data : torch.Tensor
            The reference data for comparison with the input data.
        i_idx : int
            The index of the input to use for the test statistic.
            This option is for models with multiple inputs.
            Defaults to 0.
        o_idx : int
            The index of the output to use for the calculation of the `roi`.
            This option is for models with multiple outputs.
            Defaults to 0.
        use_norm : bool, optional
            Whether to apply min-max normalization to the `score_map`.
        """
        super().__init__(threshold, i_idx, o_idx, post_process, use_norm, **kwargs)
        # self.reference_data = reference_data

    def construct_hypothesis(
        self,
        si_model: NN,
        X: tuple,
        var: float | torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        si_model : NN
            The model for the selective inference.
        X : tuple
            The tuple of the input data and the reference data.
            (input_data, reference_data)
        var: float | torch.Tensor | numpy.ndarray
            The variance of the noise.
        mask : torch.Tensor | numpy.ndarray | None, optional
            The mask can be used to specify the region that may be used for the hypothesis test.
            The mask to apply the logical AND operator to the `roi` and `non_roi`.
            Defaults to None.
        """
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError(
                f"Input must be a tuple of length 2, got {type(X).__name__}"
            )
        self.mask = mask
        self.reference_data = X[1]
        if self.reference_data is None:
            raise ValueError("reference_data is not set.")

        input_x, roi, roi_vector = self._construct_roi_vector(si_model, X[0])

        input_vec = torch.cat(
            [input_x.reshape(-1), self.reference_data.reshape(-1)]
        ).double()

        eta = torch.cat(
            [
                roi_vector / torch.sum(roi_vector),
                -roi_vector / torch.sum(roi_vector),
            ]
        ).double()

        self.si_calculator = SelectiveInferenceNorm(input_vec, var, eta, use_torch=True)

        if np.isnan(self.si_calculator.stat):
            raise NoHypothesisError

    def algorithm(
        self, si_model: NN, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, list]:
        """
        Parameters
        ----------
        si_model : NN
            The model for the selective inference.
        a : torch.Tensor
            The input tensor a.
        b : torch.Tensor
            The input tensor b.
        z : float
            The test statistic z.

        Returns
        -------
        roi_vector : torch.Tensor
            The flattened roi.
        [l, u] : list(float)
            The over-conditioning interval that contains z.
        """
        a = a[: len(a) // 2]
        b = b[: len(b) // 2]
        return self._base_algorithm(si_model, a, b, z)

    def model_selector(self, roi_vector):
        """
        Parameters
        ----------
        roi_vector : torch.Tensor
            should be a tensor of int
        Returns
        -------
        bool
            True if the input roi_vector is the same as the roi_vector in the construct_hypothesis method
        """
        return torch.all(torch.eq(self.roi_vector, roi_vector))
