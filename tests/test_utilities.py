import math
import tempfile
import pathlib
import numpy as np
import h5py
import scri
import pytest


def generate_bit_widths(bit_width):
    possible_widths = 2 ** np.arange(0, int(np.log2(bit_width)))
    bit_widths = []
    while np.sum(bit_widths) < bit_width:
        next_width = np.random.choice(possible_widths)
        if np.sum(bit_widths) + next_width <= bit_width:
            bit_widths.append(next_width)
    return tuple(bit_widths)


@pytest.mark.parametrize("bit_width", [8, 16, 32, 64])
def test_multishuffle_reversibility(bit_width):
    dt = np.dtype(f"u{bit_width//8}")
    np.random.seed(123)
    data = np.random.randint(0, high=2 ** bit_width, size=5_000, dtype=dt)
    for bit_widths in [(1,) * bit_width, (8,) * (bit_width // 8)] + [generate_bit_widths(bit_width) for _ in range(10)]:
        shuffle = scri.utilities.multishuffle(bit_widths)
        unshuffle = scri.utilities.multishuffle(bit_widths, forward=False)
        assert np.array_equal(data, unshuffle(shuffle(data))), bit_widths


@pytest.mark.parametrize("bit_width", [8, 16, 32, 64])
def test_multishuffle_like_hdf5(bit_width):
    dt = np.dtype(f"u{bit_width//8}")
    np.random.seed(1234)
    data = np.random.randint(0, high=2 ** bit_width, size=5_000, dtype=dt)

    # Save the data to file via h5py, then extract the raw data to see what
    # HDF5's shuffle looks like
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = pathlib.Path(temp_dir) / "test.h5"
        with h5py.File(file_name, "w") as f:
            f.create_dataset("data", data=data, shuffle=True, chunks=(data.size,))
        with h5py.File(file_name, "r") as f:
            ds = f["data"]
            filter_mask, raw_data_bytes = ds.id.read_direct_chunk((0,))
            hdf5_raw_data = np.frombuffer(raw_data_bytes, dtype=dt)

    # Shuffle with our function
    shuffle = scri.utilities.multishuffle((8,) * (bit_width // 8))
    scri_shuffle_data = shuffle(data)

    # Check that they are equal
    assert np.array_equal(scri_shuffle_data, hdf5_raw_data)
