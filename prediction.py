import torch
import numpy as np
from torch import Tensor, nn

from network.anomaly_detector_model import AnomalyDetector
from network.c3d import C3D
from network.TorchUtils import TorchModel
from utils.types import Device, FeatureExtractor
from feature_extractor import to_segments
from network.TorchUtils import get_torch_device
from utils.load_model import load_models
from utils.utils import build_transforms

anomaly_detector, feature_extractor = load_models(
    "./models/c3d.pickle",
    "./models/anamoly_detector.pt",
    features_method="c3d",
    device=get_torch_device(),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transforms = build_transforms(mode="c3d")


def features_extraction(
    frames,
    model ,
    device,
    frame_stride: int = 1,
    transforms = None,
):
    frames = torch.tensor(frames, device=device)  # pylint: disable=not-callable
    if transforms is not None:
        frames = transforms(frames)
    data = frames[:, range(0, frames.shape[1], frame_stride), ...]
    data = data.unsqueeze(0)

    with torch.no_grad():
        outputs = model(data.to(device)).detach().cpu()

    out = outputs.numpy()
    # print(out[0])

    return to_segments(out, 1)


def ad_prediction(
    model: nn.Module, features: Tensor, device  = "cuda"
) -> np.ndarray:

    features = torch.tensor(features).to(device)  # pylint: disable=not-callable
    with torch.no_grad():
        preds = model(features)

    return preds.detach().cpu().numpy().flatten()

def perform_prediction(batch):
        features = features_extraction(
            frames=batch,
            model=feature_extractor,
            device=device,
            transforms=transforms,
        )
        
        # print(len(features))
        # print(features[0])
        # print(features)


        local_pred = ad_prediction(
            model=anomaly_detector,
            features=features,
            device=device,
        )

        new_pred = local_pred[0]
        return new_pred

