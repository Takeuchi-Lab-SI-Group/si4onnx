import numpy as np
import torch
from torch.utils.data import Dataset


def generate_iid_data(
    n_samples, shape, loc=0, scale=1, local_signal=0, local_size=None, seed=0
):
    """Generate synthetic data with iid Gaussian noise and local signals.

    Parameters
    ----------
    n_samples : int
        The number of samples.
    shape : tuple
        The shape of the data (channels, height, width).
    loc : float, optional
        The mean of the Gaussian noise. Default is 0.0.
    scale : float, optional
        The standard deviation of the Gaussian noise. Default is 1.0.
    local_signal : float, optional
        The strength of the local signal to add. Default is 0.0.
    local_size : int or None, optional
        The size of the local signal region. If None, uses min(height, width) // 3.
    seed : int, optional
        The seed for reproducibility. Default is 0.

    Returns
    -------
    np.ndarray
        The generated data of shape (size, channels, height, width).
    np.ndarray
        Binary masks indicating signal locations of shape (size, channels, height, width).
    np.ndarray
        Binary labels indicating presence of signal of shape (size,).
    """
    if isinstance(shape, int):
        shape = (1, shape, shape)  # (channels, height, width)
    elif len(shape) == 2:
        shape = (1, *shape)  # Add channel dimension if not specified

    channels, height, width = shape

    if local_size is None:
        local_size = min(height, width) // 3

    # Validate local_size against spatial dimensions only
    assert (
        local_size < height
    ), f"local_size ({local_size}) must be less than height ({height})"
    assert (
        local_size < width
    ), f"local_size ({local_size}) must be less than width ({width})"

    rng = np.random.default_rng(seed=seed)

    # Generate base noise
    data = rng.normal(loc, scale, (n_samples, channels, height, width))

    # Initialize masks
    masks = np.zeros((n_samples, channels, height, width))

    # Add local signals for each sample
    for i in range(n_samples):
        # Generate random position for the top-left corner of the local signal
        # Only for spatial dimensions (height, width)
        h_start = rng.integers(0, height - local_size)
        w_start = rng.integers(0, width - local_size)

        # Apply signal to all channels at the same spatial location
        data[i, :, h_start : h_start + local_size, w_start : w_start + local_size] += (
            local_signal
        )

        if local_signal != 0:
            masks[
                i, :, h_start : h_start + local_size, w_start : w_start + local_size
            ] = 1

    labels = (local_signal != 0) * np.ones(n_samples, dtype=int)

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
    ...     n_samples=1000,
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
        n_samples,
        shape,
        loc=0,
        scale=1,
        local_signal=0,
        local_size=None,
        seed=0,
    ):
        self.data, self.masks, self.labels = generate_iid_data(
            n_samples,
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
