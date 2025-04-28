import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import truncated_interval
from abc import ABC, abstractmethod


class Operator(ABC):
    def __init__(self, *args, **kwargs):
        """The class is the base class for all operators."""
        super().__init__()

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def forward_si(self, x, a, b, l, u, z, **kwargs):
        pass


class InputDiff(Operator):
    def __init__(self, *args, **kwargs):
        """
        The class computes the difference with the input that is selected 'o_idx'.
        """
        super().__init__()

    def forward(self, x, input_x):
        return x - input_x

    def forward_si(self, x, a, b, l, u, z, input_x, input_a, input_b):
        """Compute (x - input_x), (a - input_a), (b - input_b)

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            input tensor or tensor list
        a : torch.Tensor | list[torch.Tensor]
            a tensor or tensor list
        b : torch.Tensor | list[torch.Tensor]
            b tensor or tensor list
        l : torch.Tensor | list[torch.Tensor]
            l tensor or tensor list
        u : torch.Tensor | list[torch.Tensor]
            u tensor or tensor list
        z : float
        Input_x : torch.Tensor | list[torch.Tensor]
            input tensor or tensor list
        Input_a : torch.Tensor | list[torch.Tensor]
            a tensor or tensor list
        Input_b : torch.Tensor | list[torch.Tensor]
            b tensor or tensor list

        Returns
        -------
        output_x : torch.Tensor | list[torch.Tensor]
            output tensor or tensor list
        output_a : torch.Tensor | list[torch.Tensor]
            output a tensor or tensor list
        output_b : torch.Tensor | list[torch.Tensor]
            output b tensor or tensor list
        l : torch.Tensor | list[torch.Tensor]
            lower bound of the truncated interval
        u : torch.Tensor | list[torch.Tensor]
            upper bound of the truncated interval
        """
        x = x - input_x
        a = a - input_a
        b = b - input_b
        return x, a, b, l, u


class Neg(Operator):
    def __init__(self, *args, **kwargs):
        """
        The class computes the negative value of the input.
        """
        super().__init__()

    def forward(self, x):
        return torch.neg(x)

    def forward_si(self, x, a, b, l, u, z):
        """
        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            input tensor or tensor list
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
        torch.Tensor | list[torch.Tensor]
            -x tensor or tensor list
        torch.Tensor | list[torch.Tensor]
            -a tensor or tensor list
        torch.Tensor | list[torch.Tensor]
            -b tensor or tensor list
        l : torch.Tensor | list[torch.Tensor]
            lower bound of the truncated interval
        u : torch.Tensor | list[torch.Tensor]
            upper bound of the truncated interval
        """
        return torch.neg(x), torch.neg(a), torch.neg(b), l, u


class Abs(Operator):
    def __init__(self, *args, **kwargs):
        """The class computes the absolute value of the input and the interval of the truncated interval."""
        super().__init__()

    def forward(self, x):
        return torch.abs(x)

    def forward_si(self, x, a, b, l, u, z):
        """Compute the interval.

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            input tensor or tensor list
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
        output_x : torch.Tensor | list[torch.Tensor]
            output tensor or tensor list
        output_a : torch.Tensor | list[torch.Tensor]
            output a tensor or tensor list
        output_b : torch.Tensor | list[torch.Tensor]
            output b tensor or tensor list
        l : torch.Tensor | list[torch.Tensor]
            lower bound of the truncated interval
        u : torch.Tensor | list[torch.Tensor]
            upper bound of the truncated interval
        """

        negative_index = x < 0
        tTa = a
        tTb = b
        tTa = torch.where(negative_index, -tTa, tTa)
        tTb = torch.where(negative_index, -tTb, tTb)
        l_negative, u_negative = truncated_interval(tTa, tTb)
        l = torch.max(l, l_negative)
        u = torch.min(u, u_negative)
        assert l <= z <= u

        positive_index = x > 0
        tTa = a
        tTb = b
        tTa = torch.where(positive_index, tTa, -tTa)
        tTb = torch.where(positive_index, tTb, -tTb)
        l_positive, u_positive = truncated_interval(tTa, tTb)
        l = torch.max(l, l_positive)
        u = torch.min(u, u_positive)
        assert l <= z <= u

        output_x = torch.abs(x)
        output_a = torch.where(positive_index, a, -a)
        output_b = torch.where(positive_index, b, -b)
        return output_x, output_a, output_b, l, u


class AverageFilter(Operator):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def _create_kernel(self, dim, channels):
        if dim == 3:  # 1D input
            kernel = torch.ones((channels, 1, self.kernel_size), dtype=torch.float64)
            kernel = kernel / self.kernel_size
        elif dim == 4:  # 2D input with potentially multiple channels (e.g., RGB)
            kernel = torch.ones(
                (channels, 1, self.kernel_size, self.kernel_size), dtype=torch.float64
            )
            kernel = kernel / (self.kernel_size**2)
        else:
            raise ValueError("Unsupported shape")
        return nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        channels = x.shape[1]
        kernel = self._create_kernel(x.dim(), channels)
        conv_func = {3: F.conv1d, 4: F.conv2d}[x.dim()]
        return conv_func(x, kernel, padding=self.padding, groups=channels)

    def forward_si(self, x, a, b, l, u, z):
        """
        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            input tensor or tensor list
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
        output_x : torch.Tensor | list[torch.Tensor]
            output tensor or tensor list
        output_a : torch.Tensor | list[torch.Tensor]
            output a tensor or tensor list
        output_b : torch.Tensor | list[torch.Tensor]
            output b tensor or tensor list
        l : torch.Tensor | list[torch.Tensor]
            lower bound of the truncated interval
        u : torch.Tensor | list[torch.Tensor]
            upper bound of the truncated interval
        """
        output_x = self.forward(x)
        output_a = self.forward(a)
        output_b = self.forward(b)
        return output_x, output_a, output_b, l, u


class GaussianFilter(Operator):
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        """Apply a Gaussian filter to the input tensor.

        Parameters
        ----------
        kernel_size : int
            size of the Gaussian kernel. Default to 3.
        sigma : float | int
            standard deviation of the Gaussian kernel. Default to 1.0.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2

    def _create_gaussian_kernel(self, dim):
        if dim == 3:  # 1D input
            x = torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1)
            kernel = torch.exp(-(x**2) / (2 * self.sigma**2))
            kernel = kernel / kernel.sum()
            return kernel.unsqueeze(0).unsqueeze(0).double()
        else:  # 2D input
            x = torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1)
            x = x.unsqueeze(0).repeat(self.kernel_size, 1)
            y = x.transpose(0, 1)
            kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
            kernel = kernel / kernel.sum()
            return kernel.unsqueeze(0).unsqueeze(0).double()

    def forward(self, x):
        """Compute the convolution of the input tensor with a Gaussian kernel.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        output : torch.Tensor
            output tensor
        """
        kernel = self._create_gaussian_kernel(x.dim())
        if x.dim() == 3:  # 1D input
            return F.conv1d(
                x,
                kernel.expand(x.shape[1], -1, -1),
                padding=self.padding,
                groups=x.shape[1],
            )
        else:  # 2D input
            return F.conv2d(
                x,
                kernel.expand(x.shape[1], -1, -1, -1),
                padding=self.padding,
                groups=x.shape[1],
            )

    def forward_si(self, x, a, b, l, u, z):
        """
        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            input tensor or tensor list
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
        output_x : torch.Tensor | list[torch.Tensor]
            output tensor or tensor list
        output_a : torch.Tensor | list[torch.Tensor]
            output a tensor or tensor list
        output_b : torch.Tensor | list[torch.Tensor]
            output b tensor or tensor list
        l : torch.Tensor | list[torch.Tensor]
            lower bound of the truncated interval
        u : torch.Tensor | list[torch.Tensor]
            upper bound of the truncated interval
        """
        output_x = self.forward(x)
        output_a = self.forward(a)
        output_b = self.forward(b)
        return output_x, output_a, output_b, l, u


class L1Norm(Operator):
    def __init__(self, *args, **kwargs):
        """
        Compute the L1 norm along the channel dimension (automatically detected).
        """
        super().__init__()

    def forward(self, x):
        """Compute the L1 norm of the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        output : torch.Tensor
            output tensor
        """
        if x.dim() == 3:
            self.channel_dim = 0
            self.channels = x.shape[0]
        elif x.dim() == 4:
            self.channel_dim = 1
            self.channels = x.shape[1]
        else:
            raise ValueError(f"Unsupported shape: {x.shape}")

        output = (
            x.abs()
            .sum(dim=self.channel_dim, keepdim=True)
            .repeat_interleave(self.channels, dim=self.channel_dim)
        )
        return output

    def forward_si(self, x, a, b, l, u, z):
        """
        Apply L1 norm across the channel dimension for selective inference components.
        
        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            input tensor or tensor list
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
        output_x : torch.Tensor | list[torch.Tensor]
            output tensor or tensor list
        output_a : torch.Tensor | list[torch.Tensor]
            output a tensor or tensor list
        output_b : torch.Tensor | list[torch.Tensor]
            output b tensor or tensor list
        l : torch.Tensor | list[torch.Tensor]
            lower bound of the truncated interval
        u : torch.Tensor | list[torch.Tensor]
            upper bound of the truncated interval
        """
        x, a, b, l, u = Abs().forward_si(x, a, b, l, u, z)
        x = x.sum(dim=self.channel_dim, keepdim=True).repeat_interleave(
            self.channels, dim=self.channel_dim
        )
        a = a.sum(dim=self.channel_dim, keepdim=True).repeat_interleave(
            self.channels, dim=self.channel_dim
        )
        b = b.sum(dim=self.channel_dim, keepdim=True).repeat_interleave(
            self.channels, dim=self.channel_dim
        )

        return x, a, b, l, u
