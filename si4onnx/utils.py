import torch
import numpy as np


def to_torch_tensor(input: torch.Tensor | np.ndarray):
    """Convert input to PyTorch tensor if it's a numpy array.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input data (numpy array or torch tensor)

    Returns
    -------
    torch.Tensor
        torch.Tensor of type double
    """
    if isinstance(input, torch.Tensor):
        return input.detach().double()
    elif isinstance(input, np.ndarray):
        return torch.from_numpy(input)


def to_numpy(
    input: torch.Tensor
    | np.ndarray
    | list[torch.Tensor | np.ndarray]
    | tuple[torch.Tensor | np.ndarray, ...],
):
    """Convert input to numpy array if it's a PyTorch tensor.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor | List[np.ndarray | torch.Tensor] | Tuple[np.ndarray | torch.Tensor, ...]
        Input data (numpy array or torch tensor)

    Returns
    -------
    numpy.ndarray or None
        Numpy array of the input data
    """
    if isinstance(input, torch.Tensor):
        return input.detach().numpy()
    elif isinstance(input, (list, tuple)):
        return [[x.numpy() if isinstance(x, torch.Tensor) else x for x in input]]
    elif input is None:
        return None


@torch.jit.script
def truncated_interval(a: torch.Tensor, b: torch.Tensor):
    """Compute the interval [l, u] = {z | a + b * z > 0}.

    Parameters
    ----------
    a : torch.Tensor
        a tensor
    b : torch.Tensor
        b tensor
    Returns
    -------
    float
        lower bound of the truncated interval
    float
        upper bound of the truncated interval
    """
    INF = torch.tensor([torch.inf]).double()
    plus_index = torch.greater(b, 0)
    minus_index = torch.less(b, 0)

    divided = torch.div(torch.neg(a), b)

    lowers = divided.masked_select(plus_index)
    l = torch.max(torch.cat([lowers, -INF]))

    uppers = divided.masked_select(minus_index)
    u = torch.min(torch.cat([uppers, INF]))

    return l, u


def thresholding(
    threshold: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    use_sigmoid: bool = False,
    use_norm: bool = False,
):
    """Threshold the input tensor x with the given threshold (thr) and return the indices greater than thr.

    Parameters
    ----------
    threshold : torch.Tensor
        threshold tensor
    x : torch.Tensor
        input tensor
    a : torch.Tensor
        a tensor
    b : torch.Tensor
        b tensor
    use_sigmoid : bool, optional
        Whether to use sigmoid or not in the output layer. Defaults to False
    use_norm : bool, optional
        Whether to apply min-max normalization. Defaults to False

    Returns
    -------
    torch.Tensor
        threshold index, which is greater than the threshold
        The shape of the tensor is flattened.
    float
        lower bound of the truncated interval
    float
        upper bound of the truncated interval
    """
    a = torch.flatten(a)
    b = torch.flatten(b)
    x = a + b * z

    if use_sigmoid:
        tau = torch.logit(threshold)
        threshold_index = x > tau
        threshold_index = torch.flatten(threshold_index).bool()

        tTa = a - tau
        tTb = b

        tTa = torch.where(threshold_index, tTa, -tTa)
        tTb = torch.where(threshold_index, tTb, -tTb)

        _l, _u = truncated_interval(tTa, tTb)
        l = torch.max(l, _l)
        u = torch.min(u, _u)
        assert l <= z <= u

        return threshold_index.int(), float(l.item()), float(u.item())

    tau = threshold
    if use_norm:
        # calculate the interval where the same maximum and minimum elements are selected.
        max_index = torch.argmax(x)
        _l, _u = truncated_interval(a[max_index] - a, b[max_index] - b)
        l = torch.max(l, _l)
        u = torch.min(u, _u)
        assert l <= z <= u

        min_index = torch.argmin(x)
        _l, _u = truncated_interval(a - a[min_index], b - b[min_index])
        l = torch.max(l, _l)
        u = torch.min(u, _u)
        assert l <= z <= u

        x_max = torch.max(x)
        x_min = torch.min(x)
        x = (x - x_min) / (x_max - x_min)

        a_max = a[max_index]
        a_min = a[min_index]
        b_max = b[max_index]
        b_min = b[min_index]

        a_bias = tau * (a_max - a_min)
        b_bias = tau * (b_max - b_min)
        a = a - a_min
        b = b - b_min

        tTa = a - a_bias
        tTb = b - b_bias
    else:
        # calculate the interval "a + b * z > tau".
        tTa = a - tau
        tTb = b

    threshold_index = x > tau
    threshold_index = torch.flatten(threshold_index).bool()
    tTa = torch.where(threshold_index, tTa, -tTa)
    tTb = torch.where(threshold_index, tTb, -tTb)
    _l, _u = truncated_interval(tTa, tTb)
    l = torch.max(l, _l)
    u = torch.min(u, _u)
    assert l <= z <= u

    return threshold_index.int(), float(l.item()), float(u.item())
