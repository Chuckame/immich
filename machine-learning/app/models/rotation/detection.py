from typing import Any

import numpy as np
from numpy.typing import NDArray

from huggingface_hub import snapshot_download

from pathlib import Path
from app.models.base import InferenceModel
from app.models.transforms import decode_pil, normalize, to_numpy
from app.schemas import ModelTask, ModelType, ModelFormat
from PIL import Image

IMAGE_SIZE = 224
MODEL_NAME = "Chuckame/deep-image-orientation-angle-detection"

class AngleDetector(InferenceModel):
    depends = []
    identity = (ModelType.ANGLE_PREDICTION, ModelTask.ANGLE_PREDICTION)

    def __init__(self, model_name: str, **model_kwargs: Any) -> None:
        super().__init__(MODEL_NAME, model_format=ModelFormat.ONNX, **model_kwargs)

    def _download(self) -> None:
        snapshot_download(
            MODEL_NAME,
            cache_dir=self.cache_dir,
            local_dir=self.cache_dir,
            ignore_patterns=[],
        )

    @property
    def model_path(self) -> Path:
        return self.cache_dir / f"deep-image-orientation-angle-detection.onnx"

    def _transform(self, image: Image.Image) -> dict[str, NDArray[np.float32]]:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        # v2 - to check
        # image = resize_pil(image, IMAGE_SIZE)
        # image = crop_pil(image, IMAGE_SIZE)
        image_np = to_numpy(image)
        image_np = normalize(image_np, 0.5, 0.5)
        return np.expand_dims(image_np.transpose(2, 0, 1), 0)

    def _predict(self, inputs: Image.Image | bytes, **kwargs: Any) -> np.float32:
        image = decode_pil(inputs)
        res: np.float32 = self.session.run(None, {"image": self._transform(image)})[0][0][0]
        return res
