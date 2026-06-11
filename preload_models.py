import gc
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("DOCTR_CACHE_DIR", os.path.expanduser("~/.cache/doctr"))

import torch
from doctr.models import detection_predictor, recognition_predictor
from doctr.models.classification.zoo import crop_orientation_predictor, page_orientation_predictor


DET_ARCHS = [
    "fast_base",
    "fast_small",
    "fast_tiny",
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]

RECO_ARCHS = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "parseq",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "viptr_tiny",
]


def release(model: object) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    print(f"Using DOCTR_CACHE_DIR={os.environ.get('DOCTR_CACHE_DIR')}")

    for arch in DET_ARCHS:
        print(f"Preloading detector: {arch}", flush=True)
        model = detection_predictor(arch=arch, pretrained=True)
        release(model)

    for arch in RECO_ARCHS:
        print(f"Preloading recognizer: {arch}", flush=True)
        model = recognition_predictor(arch=arch, pretrained=True)
        release(model)

    print("Preloading page orientation predictor", flush=True)
    model = page_orientation_predictor(pretrained=True)
    release(model)

    print("Preloading crop orientation predictor", flush=True)
    model = crop_orientation_predictor(pretrained=True)
    release(model)

    print("docTR model preload complete", flush=True)


if __name__ == "__main__":
    main()
