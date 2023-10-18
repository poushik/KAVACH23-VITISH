"""This module contains a training procedure for video feature extraction."""
from typing import List

import numpy as np



def to_segments(
    data, n_segments: int = 32
) -> List[np.ndarray]:
    """These code is taken from:

        # https://github.com/rajanjitenpatel/C3D_feature_extraction/blob/b5894fa06d43aa62b3b64e85b07feb0853e7011a/extract_C3D_feature.py#L805

    Args:
        data (Union[Tensor, np.ndarray]): List of features of a certain video
        n_segments (int, optional): Number of segments

    Returns:
        List[np.ndarray]: List of `num` segments
    """
    data = np.array(data)
    Segments_Features = []
    thirty2_shots = np.round(np.linspace(0, len(data) - 1, num=n_segments + 1)).astype(
        int
    )
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
        if ss == ee:
            temp_vect = data[min(ss, data.shape[0] - 1), :]
        else:
            temp_vect = data[ss:ee, :].mean(axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)

        if np.linalg.norm(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features