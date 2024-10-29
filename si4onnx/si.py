from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from onnx import ModelProto
from sicore import (
    SelectiveInferenceNorm,
    SelectiveInferenceChi,
    SelectiveInferenceResult,
)

from . import nn
from .utils import to_numpy
from .hypothesis import PresetHypothesis


@dataclass
class InferenceResult(SelectiveInferenceResult):
    """A class extending SelectiveInferenceResult with ROI and output information.

    Attributes
    ----------
    roi : np.ndarray
        Region of interest for the inference.
    output : np.ndarray
        Output of the inference process.
    score_map : np.ndarray
        Score map obtained from the inference process.
    non_roi : np.ndarray
        Region outside of the ROI.
    stat : float
        Test statistic value.
    p_value : float
        Selective p-value.
    inf_p : float
        Lower bound of selective p-value.
    sup_p : float
        Upper bound of selective p-value.
    searched_intervals : list[list[float]]
        Intervals where the search was performed.
    truncated_intervals : list[list[float]]
        Intervals where the selected model is obtained.
    search_count : int
        Number of times the search was performed.
    detect_count : int
        Number of times the selected model was obtained.
    null_rv : rv_continuous
        Null distribution of the unconditional test statistic.
    alternative : Literal["two-sided", "less", "greater"]
        Type of the alternative hypothesis.
    """

    output: np.ndarray
    score_map: np.ndarray
    roi: np.ndarray
    non_roi: np.ndarray | None = None


class SIModel(ABC):
    def __init__(self):
        self.si_calculator: SelectiveInferenceNorm | SelectiveInferenceChi = None

    @abstractmethod
    def construct_hypothesis(
        self,
        input: torch.Tensor | np.ndarray | list | tuple,
        var: int | float | np.ndarray | torch.Tensor,
        **kwargs,
    ):
        """Abstruct method for construct hypothesis from the observed output of NN.

        Parameters
        ----------
        input : torch.Tensor | np.ndarray | list | tuple
            Input of NN
        var : int | float | np.ndarray | torch.Tensor
            Covariance matrix of input
            Treated as the diagonal of the covariance matrix, representing independent variances for each dimension.

        Raises
        ------
        NoHypothesisError
            If the hypothesis is not obtained from observartion, please raise this error
        """
        pass

    @abstractmethod
    def algorithm(
        self, a: torch.Tensor, b: torch.Tensor, z: float, **kwargs
    ) -> tuple[object, tuple[float, float]]:
        """
        Parameters
        ----------
        a : torch.Tensor
            A vector of nuisance parameter
        b : torch.Tensor
            A vector of the direction of test statistic
        z : float
            A test statistic

        Returns
        -------
        tuple[object, tuple[float,float]]
            First Elements is outputs obtained in the value of z. Second Element is a obtained truncated interval
        """
        pass

    @abstractmethod
    def model_selector(
        self,
        roi_vector: torch.Tensor | np.ndarray | list | tuple | int | float,
        **kwargs,
    ) -> bool:
        """Abstruct method for compare whether same model are obtained from output and observed output(self.output)

        Parameters
        ----------
        roi_vector : Any
            roi obtained from the output of NN

        Returns
        -------
        bool
            If same models are obtained from output and observed output(self.output), Return value should be true. If not, return value should be false.
        """
        pass

    def forward(self, input: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.si_model.forward(input)

    def inference(
        self,
        input: torch.Tensor | np.ndarray,
        var: float | np.ndarray | torch.Tensor,
        **kwargs,
    ) -> SelectiveInferenceResult:
        """
        Parameters
        ----------
        input : torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray] | tuple[torch.Tensor | np.ndarray, ...]
            Input of NN
        var : float | np.ndarray | torch.Tensor
            Covariance matrix of the noise of input.
        **kwargs : Any
        """
        self.construct_hypothesis(input, var)
        result = self.si_calculator.inference(
            algorithm=self.algorithm,
            model_selector=self.model_selector,
            **kwargs,
        )
        return result


class PresetSIModel(SIModel):
    def __init__(
        self,
        model: ModelProto,
        hypothesis: PresetHypothesis,
        seed: int = None,
        memoization: bool = True,
        **kwargs,
    ):
        """SIModel class for Preset Hypothesis

        Parameters
        ----------
        model : onnx.ModelProto
            The onnx model instance.
        hypothesis : PresetHypothesis
            The hypothesis setting.
        seed : int, optional
            The seed of random number generator.
            If the onnx model contains RandomNormalLike layers, the seed is used to generate the same random numbers.
            Default to None.
        memoization : bool, optional
            Whether to use memoization.
            If True, the memoization is enabled. Default to True.
        """
        self.si_model = nn.NN(model=model, seed=seed, memoization=memoization)
        self.hypothesis = hypothesis
        self.si_calculator = None
        self.output = None
        self.score_map = None
        self.roi = None

    def construct_hypothesis(
        self,
        input: torch.Tensor
        | np.ndarray
        | list[torch.Tensor | np.ndarray]
        | tuple[torch.Tensor | np.ndarray, ...],
        var: float | np.ndarray | torch.Tensor,
        mask: torch.Tensor | np.ndarray = None,
        **kwargs,
    ):
        self.hypothesis.construct_hypothesis(
            si_model=self.si_model, X=input, var=var, mask=mask, **kwargs
        )
        self.si_calculator = self.hypothesis.si_calculator
        self.output = self.hypothesis.output
        self.score_map = self.hypothesis.score_map
        self.roi = self.hypothesis.roi
        if hasattr(self.hypothesis, "non_roi"):
            self.non_roi = self.hypothesis.non_roi
        else:
            self.non_roi = None

    def algorithm(self, a, b, z, **kwargs):
        return self.hypothesis.algorithm(self.si_model, a, b, z, **kwargs)

    def model_selector(self, roi, **kwargs):
        return self.hypothesis.model_selector(roi, **kwargs)

    def inference(
        self,
        input: torch.Tensor
        | np.ndarray
        | list[torch.Tensor | np.ndarray]
        | tuple[torch.Tensor | np.ndarray, ...],
        var: float | np.ndarray | torch.Tensor,
        mask: torch.Tensor | np.ndarray = None,
        **kwargs,
    ) -> InferenceResult:
        """Inference process for Selective Inference

        Parameters
        ----------
        input : torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray] | tuple[torch.Tensor | np.ndarray, ...]
            Input of NN
        var : float | np.ndarray | torch.Tensor
            Covariance matrix of the noise of input.
        mask : torch.Tensor | numpy.ndarray | None, optional
            The mask can be used to specify the region that may be used for the hypothesis test.
            The mask to apply the logical AND operator to the `roi` and `non_roi`.
            Defaults to None.
        **kwargs : Any

        Returns
        -------
        InferenceResult
            The result of the inference.
        """
        self.construct_hypothesis(input, var, mask)
        result = self.si_calculator.inference(
            algorithm=self.algorithm,
            model_selector=self.model_selector,
            **kwargs,
        )
        result.output = to_numpy(self.output)
        result.score_map = to_numpy(self.score_map)
        result.roi = to_numpy(self.roi)
        result.non_roi = to_numpy(self.non_roi)
        return result


def load(
    model: ModelProto,
    hypothesis: PresetHypothesis,
    seed: float = None,
    memoization: bool = True,
) -> PresetSIModel:
    """Load onnx model and hypothesis setting to SIModel

    Parameters
    ----------
    model : onnx.ModelProto
        The onnx model instance.
    hypothesis : PresetHypothesis
        The hypothesis setting.
        You can choose an instance of the class "PresetHypothesis" for preset hypothesis setting.
    seed : float, optional
        The seed of random number generator.
        If the onnx model contains RandomNormalLike layers, the seed is used to generate the same random numbers.
        Default to None.
    memoization : bool, optional
        Whether to use memoization.
        If True, the memoization is enabled

    Returns
    -------
    si_model : PresetSIModel
        The Selective Inference model

    Raises
    ------
    ValueError
        If hypothesis is not an instance of PresetHypothesis
    """

    if isinstance(hypothesis, PresetHypothesis):
        si_model = PresetSIModel(model, hypothesis, seed, memoization)
    else:
        raise ValueError("hypothesis should be an instance of PresetHypothesis")
    return si_model
