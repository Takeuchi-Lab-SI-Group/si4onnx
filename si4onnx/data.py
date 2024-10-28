import numpy as np
import torch
from torch.utils.data import Dataset


def generate_iid_data(
    size, shape, loc=0, scale=1, local_signal=0, local_size=None, seed=0
):
    """Generate synthetic data with iid Gaussian noise.

    Parameters
    ----------
    size : int
        The number of samples.
    shape : int or tuple
        The shape of the data.
    loc : float, optional
        The mean of the Gaussian noise. Default is 0.0.
    scale : float, optional
        The standard deviation of the Gaussian noise. Default is 1.0.
    signal : float, optional
        The signal strength. Default is 1.0.
    seed : int, optional
        The seed of the random number generator. Default is None.

    Returns
    -------
    np.ndarray
        The generated data.
    np.ndarray
        The masks of the signal.
    np.ndarray
        The labels indicating the presence of the signal.
    """
    if local_size is None:
        local_size = shape[-1] // 3

    rng = np.random.default_rng(seed=seed)

    if isinstance(shape, int):
        shape = (shape,)

    data = rng.normal(loc, scale, (size, *shape))

    masks = np.zeros((size, *shape))

    for dim in shape[1:]:
        assert (
            local_size < dim
        ), "local_size must be less than to the shape of the data."

    local_shape = tuple(max(dim, local_size) for dim in shape[1:])

    local_positions = [
        tuple(rng.integers(0, max(1, data_dim - local_dim), size))
        for data_dim, local_dim in zip(shape, local_shape)
    ]

    for i in range(size):
        slices = tuple(
            slice(pos[i], pos[i] + local_dim)
            for pos, local_dim in zip(local_positions, local_shape)
        )
        data[i][slices] += local_signal
        if local_signal != 0:
            masks[i][slices] = 1

    labels = (local_signal != 0) * np.ones(size, dtype=int)

    return data, masks, labels


class SyntheticDataset(Dataset):
    """A PyTorch Dataset for generating synthetic data with iid Gaussian noise and local signals.

    This dataset generates samples containing Gaussian noise with optional local signals
    added at random positions.

    The dataset generates three components for each sample:
        1. Data: Tensor containing Gaussian noise with optional local signals
        2. Mask: Binary tensor indicating the position of local signals
        3. Label: Binary value indicating presence of local signal

    Attributes
    ----------
    data : numpy.ndarray
        Array containing the generated synthetic data.
    masks : numpy.ndarray
        Array containing the masks for local signals.
    labels : numpy.ndarray
        Array containing the binary labels.

    Examples
    --------
    >>> dataset = SyntheticDataset(
    ...     size=1000,
    ...     shape=(32, 32),
    ...     loc=0,
    ...     scale=1,
    ...     local_signal=2.0
    ... )
    >>> data, mask, label = dataset[0]
    >>> print(data.shape, mask.shape, label.shape)
    torch.Size([32, 32]) torch.Size([32, 32]) torch.Size([])
    """

    def __init__(
        self,
        size,
        shape,
        loc=0,
        scale=1,
        local_signal=0,
        local_size=None,
        seed=0,
    ):
        self.data, self.masks, self.labels = generate_iid_data(
            size,
            shape,
            loc,
            scale,
            local_signal,
            local_size,
            seed,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).float()
        label = torch.tensor(self.labels[idx]).float()
        return data, mask, label
