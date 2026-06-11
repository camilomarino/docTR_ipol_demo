import argparse
import csv
import json
import math
import os
import random
import shutil
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("DOCTR_CACHE_DIR", os.path.expanduser("~/.cache/doctr"))

import cv2
import numpy as np
import torch
from doctr.io.elements import Document
from doctr.models import ocr_predictor
from doctr.models._utils import get_language
from doctr.utils.geometry import detach_scores
from PIL import Image, ImageDraw, ImageFont


OUTPUT_IMAGE_FILES = [
    "ocr_page.png",
    "detector_input.png",
    "detector_probability_map.png",
    "detector_binary_map.png",
    "detector_components.png",
    "detector_word_boxes.png",
    "recognizer_raw_crops_stack.png",
    "recognizer_input_crops_stack.png",
    "recognizer_crops_contact_sheet.png",
    "reading_order_words.png",
    "line_grouping.png",
    "block_grouping.png",
    "overlay_words.png",
    "overlay_lines.png",
    "overlay_blocks.png",
    "overlay_all.png",
]

OUTPUT_TEXT_FILES = [
    "result_raw_doctr.json",
    "result.json",
    "result.txt",
    "summary.txt",
    "structure_debug.json",
    "words.csv",
    "result.hocr",
    "hocr_error.txt",
]

PARAMETER_EFFECTS = {
    "det_arch": "Passed to doctr.models.ocr_predictor(det_arch=...). It selects the detection model before any inference.",
    "reco_arch": "Passed to doctr.models.ocr_predictor(reco_arch=...). It selects the recognition model before any inference.",
    "det_input_size": "If not 'model_default', applied after model construction as predictor.det_predictor.pre_processor.resize.size = (size, size). The default leaves docTR unchanged.",
    "preserve_aspect_ratio": "Passed to doctr.models.ocr_predictor(preserve_aspect_ratio=...). docTR forwards it to the detector preprocessor resize transform.",
    "symmetric_pad": "Passed to doctr.models.ocr_predictor(symmetric_pad=...). docTR forwards it to the detector preprocessor resize transform.",
    "reco_preserve_aspect_ratio": "Applied after model construction as predictor.reco_predictor.pre_processor.resize.preserve_aspect_ratio. docTR's recognizer default is True.",
    "reco_symmetric_pad": "Applied after model construction as predictor.reco_predictor.pre_processor.resize.symmetric_pad. docTR's recognizer default is False.",
    "assume_straight_pages": "Passed to doctr.models.ocr_predictor(assume_straight_pages=...). It controls straight boxes versus rotated crop preparation.",
    "export_as_straight_boxes": "Passed to doctr.models.ocr_predictor(export_as_straight_boxes=...). It is used by docTR's DocumentBuilder during export.",
    "straighten_pages": "Passed to doctr.models.ocr_predictor(straighten_pages=...). When enabled, docTR estimates orientation and runs detection again on straightened pages.",
    "detect_orientation": "Passed to doctr.models.ocr_predictor(detect_orientation=...). It adds page orientation metadata to the exported document.",
    "detect_language": "Passed to doctr.models.ocr_predictor(detect_language=...). It adds language metadata from recognized text.",
    "disable_page_orientation": "Passed to doctr.models.ocr_predictor(disable_page_orientation=...). It disables the page orientation predictor while preserving docTR's pipeline shape.",
    "disable_crop_orientation": "Passed to doctr.models.ocr_predictor(disable_crop_orientation=...). It disables crop orientation prediction when rotated crop rectification is active.",
    "resolve_lines": "Passed through ocr_predictor(..., resolve_lines=...) to docTR's DocumentBuilder.",
    "resolve_blocks": "Passed through ocr_predictor(..., resolve_blocks=...) to docTR's DocumentBuilder.",
    "paragraph_break": "Passed through ocr_predictor(..., paragraph_break=...) to docTR's DocumentBuilder line/block grouping.",
    "bin_thresh": "Applied after model construction as predictor.det_predictor.model.postprocessor.bin_thresh. It thresholds the detector response map.",
    "box_thresh": "Applied after model construction as predictor.det_predictor.model.postprocessor.box_thresh. It filters detector boxes by score.",
    "unclip_ratio": "Applied after model construction as predictor.det_predictor.model.postprocessor.unclip_ratio when available. It controls detector box expansion.",
    "draw_labels": "Visualization-only. It does not affect docTR inference or JSON outputs.",
    "draw_confidence": "Visualization-only. It does not affect docTR inference or JSON outputs.",
    "min_confidence_display": "Visualization-only. It hides low-confidence words in overlays but never filters JSON, CSV, hOCR, or reading order text.",
    "recognizer_sample_count": "Visualization-only. It selects how many recognized crops are shown in diagnostic stack/contact-sheet images.",
    "visualization_seed": "Visualization-only. It seeds random crop sampling for diagnostic images.",
}

CSV_HEADER = [
    "page",
    "block",
    "line",
    "word",
    "text",
    "confidence",
    "objectness_score",
    "geometry",
]

PALETTE = [
    (32, 90, 180),
    (224, 119, 32),
    (47, 140, 70),
    (184, 64, 108),
    (102, 84, 180),
    (39, 150, 150),
    (202, 170, 48),
    (150, 75, 40),
]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def empty_to(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return value


def coerce_bool(value: Any, default: bool) -> bool:
    value = empty_to(value, default)
    try:
        return str2bool(value)
    except argparse.ArgumentTypeError:
        return default


def coerce_float(value: Any, default: float) -> float:
    value = empty_to(value, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: int) -> int:
    value = empty_to(value, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def add_optional_value(parser: argparse.ArgumentParser, flag: str, default: Any) -> None:
    parser.add_argument(flag, nargs="?", default=default, const="")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.det_arch = empty_to(args.det_arch, "fast_base")
    args.reco_arch = empty_to(args.reco_arch, "crnn_vgg16_bn")
    args.det_input_size = empty_to(args.det_input_size, "model_default")
    args.preserve_aspect_ratio = coerce_bool(args.preserve_aspect_ratio, True)
    args.symmetric_pad = coerce_bool(args.symmetric_pad, True)
    args.reco_preserve_aspect_ratio = coerce_bool(args.reco_preserve_aspect_ratio, True)
    args.reco_symmetric_pad = coerce_bool(args.reco_symmetric_pad, False)
    args.assume_straight_pages = coerce_bool(args.assume_straight_pages, True)
    args.export_as_straight_boxes = coerce_bool(args.export_as_straight_boxes, False)
    args.straighten_pages = coerce_bool(args.straighten_pages, False)
    args.detect_orientation = coerce_bool(args.detect_orientation, False)
    args.detect_language = coerce_bool(args.detect_language, False)
    args.disable_page_orientation = coerce_bool(args.disable_page_orientation, False)
    args.disable_crop_orientation = coerce_bool(args.disable_crop_orientation, False)
    args.resolve_lines = coerce_bool(args.resolve_lines, True)
    args.resolve_blocks = coerce_bool(args.resolve_blocks, False)
    args.paragraph_break = coerce_float(args.paragraph_break, 0.035)
    args.bin_thresh = coerce_float(args.bin_thresh, 0.1)
    args.box_thresh = coerce_float(args.box_thresh, 0.1)
    args.unclip_ratio = coerce_float(args.unclip_ratio, 1.0)
    args.det_bs = coerce_int(args.det_bs, 2)
    args.reco_bs = coerce_int(args.reco_bs, 128)
    args.draw_labels = coerce_bool(args.draw_labels, True)
    args.draw_confidence = coerce_bool(args.draw_confidence, False)
    args.min_confidence_display = coerce_float(args.min_confidence_display, 0.0)
    args.recognizer_sample_count = coerce_int(args.recognizer_sample_count, 24)
    args.visualization_seed = coerce_int(args.visualization_seed, 0)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IPOL docTR demo runner")
    parser.add_argument("--input", required=True)
    add_optional_value(parser, "--det-arch", "fast_base")
    add_optional_value(parser, "--reco-arch", "crnn_vgg16_bn")
    add_optional_value(parser, "--det-input-size", "model_default")
    add_optional_value(parser, "--preserve-aspect-ratio", True)
    add_optional_value(parser, "--symmetric-pad", True)
    add_optional_value(parser, "--reco-preserve-aspect-ratio", True)
    add_optional_value(parser, "--reco-symmetric-pad", False)
    add_optional_value(parser, "--assume-straight-pages", True)
    add_optional_value(parser, "--export-as-straight-boxes", False)
    add_optional_value(parser, "--straighten-pages", False)
    add_optional_value(parser, "--detect-orientation", False)
    add_optional_value(parser, "--detect-language", False)
    add_optional_value(parser, "--disable-page-orientation", False)
    add_optional_value(parser, "--disable-crop-orientation", False)
    add_optional_value(parser, "--resolve-lines", True)
    add_optional_value(parser, "--resolve-blocks", False)
    add_optional_value(parser, "--paragraph-break", 0.035)
    add_optional_value(parser, "--bin-thresh", 0.1)
    add_optional_value(parser, "--box-thresh", 0.1)
    add_optional_value(parser, "--unclip-ratio", 1.0)
    add_optional_value(parser, "--det-bs", 2)
    add_optional_value(parser, "--reco-bs", 128)
    add_optional_value(parser, "--draw-labels", True)
    add_optional_value(parser, "--draw-confidence", False)
    add_optional_value(parser, "--min-confidence-display", 0.0)
    add_optional_value(parser, "--recognizer-sample-count", 24)
    add_optional_value(parser, "--visualization-seed", 0)
    return normalize_args(parser.parse_args())


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array[..., :3]).save(path)


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def save_placeholder(path: str | Path, title: str, message: str = "", size: tuple[int, int] = (900, 280)) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(190, 190, 190), width=2)
    draw.text((24, 24), title, fill=(40, 40, 40), font=font())
    if message:
        wrapped = wrap_text(message, 105)
        draw.multiline_text((24, 58), wrapped, fill=(80, 80, 80), spacing=5, font=font())
    image.save(path)


def wrap_text(text: str, width: int) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(part) + 1 for part in current) + len(word) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def geometry_to_points(geometry: Any) -> list[tuple[float, float]]:
    if geometry is None:
        return []
    array = np.asarray(geometry, dtype=float)
    if array.size == 0:
        return []
    array = np.squeeze(array)
    if array.ndim == 1:
        if array.size == 4:
            x1, y1, x2, y2 = array.tolist()
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if array.size % 2 == 0:
            array = array.reshape((-1, 2))
        else:
            return []
    if array.ndim == 2:
        if array.shape == (2, 2):
            x1, y1 = array[0].tolist()
            x2, y2 = array[1].tolist()
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if array.shape[1] == 2:
            return [(float(x), float(y)) for x, y in array.tolist()]
    return []


def scale_points(points: list[tuple[float, float]], size: tuple[int, int]) -> list[tuple[int, int]]:
    width, height = size
    if not points:
        return []
    max_abs = max(max(abs(x), abs(y)) for x, y in points)
    if max_abs <= 2.0:
        scaled = [(int(round(x * width)), int(round(y * height))) for x, y in points]
    else:
        scaled = [(int(round(x)), int(round(y))) for x, y in points]
    return [(max(0, min(width - 1, x)), max(0, min(height - 1, y))) for x, y in scaled]


def bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def union_geometries(geometries: list[Any]) -> list[tuple[float, float]]:
    boxes = [bbox_from_points(geometry_to_points(geometry)) for geometry in geometries]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return []
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def draw_geometry(
    draw: ImageDraw.ImageDraw,
    geometry: Any,
    page_size: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 2,
    label: str | None = None,
) -> None:
    points = scale_points(geometry_to_points(geometry), page_size)
    if not points:
        return
    draw.line(points + [points[0]], fill=color, width=width)
    if label:
        x = min(point[0] for point in points)
        y = min(point[1] for point in points)
        text_pos = (x + 2, max(0, y - 12))
        draw.rectangle(
            [text_pos[0] - 1, text_pos[1] - 1, text_pos[0] + min(360, 6 * len(label)) + 4, text_pos[1] + 12],
            fill=(255, 255, 255),
        )
        draw.text(text_pos, label[:60], fill=color, font=font())


def primary_map(out_map: Any) -> np.ndarray:
    array = np.asarray(out_map)
    array = np.squeeze(array)
    if array.ndim == 3:
        if array.shape[0] <= 4:
            array = array[0]
        else:
            array = array[..., 0]
    if array.ndim != 2:
        array = np.zeros((128, 128), dtype=np.float32)
    return np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def heatmap_rgb(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    if data.size == 0:
        data = np.zeros((128, 128), dtype=np.float32)
    if float(np.max(data)) <= 1.0 and float(np.min(data)) >= 0.0:
        gray = (data * 255).clip(0, 255).astype(np.uint8)
    else:
        lo = float(np.min(data))
        hi = float(np.max(data))
        gray = ((data - lo) / max(hi - lo, 1e-6) * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def binary_map(array: np.ndarray, threshold: float) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    return (data > threshold).astype(np.uint8) * 255


def tensor_to_rgb(tensor: torch.Tensor, preprocessor: Any) -> np.ndarray:
    sample = tensor.detach().cpu().float()
    if sample.ndim == 4:
        sample = sample[0]
    mean = torch.tensor(getattr(preprocessor.normalize, "mean", (0.0, 0.0, 0.0))).view(-1, 1, 1)
    std = torch.tensor(getattr(preprocessor.normalize, "std", (1.0, 1.0, 1.0))).view(-1, 1, 1)
    sample = sample * std + mean
    sample = sample.clamp(0, 1)
    array = sample.permute(1, 2, 0).numpy()
    return (array * 255).round().astype(np.uint8)


def preprocessed_images(preprocessor: Any, images: list[np.ndarray]) -> list[np.ndarray]:
    if not images:
        return []
    processed: list[np.ndarray] = []
    for batch in preprocessor(images):
        for sample in batch:
            processed.append(tensor_to_rgb(sample, preprocessor))
    return processed


def detector_input_preview(preprocessor: Any, page: np.ndarray) -> np.ndarray:
    processed = preprocessed_images(preprocessor, [page])
    if processed:
        return processed[0]
    return page


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    height, original_width = image.shape[:2]
    if original_width == 0 or height == 0:
        return np.zeros((32, width, 3), dtype=np.uint8)
    new_height = max(1, int(round(height * width / original_width)))
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)


def make_vertical_stack(images: list[np.ndarray], labels: list[str], path: str, title: str) -> None:
    if not images:
        save_placeholder(path, title, "No word crops were detected.")
        return
    target_width = 520
    label_width = 190
    rows: list[np.ndarray] = []
    for idx, image in enumerate(images):
        resized = resize_to_width(image, target_width)
        row = np.ones((max(resized.shape[0], 34), target_width + label_width, 3), dtype=np.uint8) * 255
        row[: resized.shape[0], label_width:, :] = resized
        pil = Image.fromarray(row)
        draw = ImageDraw.Draw(pil)
        draw.text((8, 8), labels[idx][:28] if idx < len(labels) else str(idx), fill=(30, 30, 30), font=font())
        rows.append(np.array(pil))
    stack = np.vstack(rows)
    save_rgb(path, stack)


def make_contact_sheet(images: list[np.ndarray], labels: list[str], path: str) -> None:
    if not images:
        save_placeholder(path, "Recognizer crop contact sheet", "No word crops were detected.")
        return
    columns = min(4, max(1, len(images)))
    tile_w, tile_h = 260, 90
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, image in enumerate(images):
        x = (idx % columns) * tile_w
        y = (idx // columns) * tile_h
        crop = Image.fromarray(image)
        crop.thumbnail((tile_w - 16, tile_h - 30), Image.Resampling.LANCZOS)
        sheet.paste(crop, (x + 8, y + 8))
        draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(210, 210, 210))
        label = labels[idx][:34] if idx < len(labels) else str(idx)
        draw.text((x + 8, y + tile_h - 18), label, fill=(30, 30, 30), font=font())
    sheet.save(path)


def flatten_pages_crops(crops: list[list[np.ndarray]]) -> list[np.ndarray]:
    return [crop for page_crops in crops for crop in page_crops]


def sample_indices(total: int, count: int, seed: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    rng = random.Random(seed)
    count = min(total, count)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:count])


def export_pages(document_export: dict[str, Any]) -> list[dict[str, Any]]:
    return list(document_export.get("pages", []))


def word_rows(document_export: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page_idx, page in enumerate(export_pages(document_export)):
        for block_idx, block in enumerate(page.get("blocks", [])):
            for line_idx, line in enumerate(block.get("lines", [])):
                for word_idx, word in enumerate(line.get("words", [])):
                    rows.append(
                        {
                            "page": page_idx,
                            "block": block_idx,
                            "line": line_idx,
                            "word": word_idx,
                            "text": word.get("value", ""),
                            "confidence": word.get("confidence"),
                            "objectness_score": word.get("objectness_score"),
                            "geometry": word.get("geometry"),
                        }
                    )
    return rows


def write_words_csv(rows: list[dict[str, Any]], path: str = "words.csv") -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "page": row["page"],
                    "block": row["block"],
                    "line": row["line"],
                    "word": row["word"],
                    "text": row["text"],
                    "confidence": row["confidence"],
                    "objectness_score": row["objectness_score"],
                    "geometry": json.dumps(to_jsonable(row["geometry"]), ensure_ascii=False),
                }
            )


def line_geometry(line: dict[str, Any]) -> Any:
    if line.get("geometry") is not None:
        return line["geometry"]
    return union_geometries([word.get("geometry") for word in line.get("words", [])])


def block_geometry(block: dict[str, Any]) -> Any:
    if block.get("geometry") is not None:
        return block["geometry"]
    return union_geometries([line_geometry(line) for line in block.get("lines", [])])


def word_label(row: dict[str, Any], draw_confidence: bool) -> str:
    text = str(row.get("text", ""))
    if draw_confidence and row.get("confidence") is not None:
        return f"{text} {float(row['confidence']):.2f}"
    return text


def save_overlay(
    path: str,
    page: np.ndarray,
    document_export: dict[str, Any],
    level: str,
    draw_labels: bool,
    draw_confidence: bool,
    min_confidence: float,
) -> None:
    image = Image.fromarray(page.copy())
    draw = ImageDraw.Draw(image)
    page_size = image.size
    pages = export_pages(document_export)
    if not pages:
        image.save(path)
        return
    page_export = pages[0]
    if level in {"blocks", "all"}:
        for block_idx, block in enumerate(page_export.get("blocks", [])):
            draw_geometry(draw, block_geometry(block), page_size, (32, 90, 180), 4, f"B{block_idx}" if level != "all" else None)
    if level in {"lines", "all"}:
        for block_idx, block in enumerate(page_export.get("blocks", [])):
            for line_idx, line in enumerate(block.get("lines", [])):
                draw_geometry(
                    draw,
                    line_geometry(line),
                    page_size,
                    (224, 119, 32),
                    3,
                    f"L{block_idx}.{line_idx}" if level != "all" else None,
                )
    if level in {"words", "all"}:
        for row in word_rows(document_export):
            if row["page"] != 0:
                continue
            confidence = row.get("confidence")
            if confidence is not None and float(confidence) < min_confidence:
                continue
            label = word_label(row, draw_confidence) if draw_labels else None
            draw_geometry(draw, row["geometry"], page_size, (47, 140, 70), 2, label)
    image.save(path)


def save_reading_order(path: str, page: np.ndarray, document_export: dict[str, Any]) -> None:
    image = Image.fromarray(page.copy())
    draw = ImageDraw.Draw(image)
    for idx, row in enumerate([row for row in word_rows(document_export) if row["page"] == 0]):
        draw_geometry(draw, row["geometry"], image.size, (47, 140, 70), 2, str(idx + 1))
    image.save(path)


def save_line_grouping(path: str, page: np.ndarray, document_export: dict[str, Any]) -> None:
    image = Image.fromarray(page.copy())
    draw = ImageDraw.Draw(image)
    pages = export_pages(document_export)
    if pages:
        for block_idx, block in enumerate(pages[0].get("blocks", [])):
            for line_idx, line in enumerate(block.get("lines", [])):
                color = PALETTE[(block_idx * 11 + line_idx) % len(PALETTE)]
                draw_geometry(draw, line_geometry(line), image.size, color, 4, f"L{block_idx}.{line_idx}")
                for word in line.get("words", []):
                    draw_geometry(draw, word.get("geometry"), image.size, color, 2)
    image.save(path)


def save_block_grouping(path: str, page: np.ndarray, document_export: dict[str, Any]) -> None:
    image = Image.fromarray(page.copy())
    draw = ImageDraw.Draw(image)
    pages = export_pages(document_export)
    if pages:
        for block_idx, block in enumerate(pages[0].get("blocks", [])):
            color = PALETTE[block_idx % len(PALETTE)]
            draw_geometry(draw, block_geometry(block), image.size, color, 5, f"B{block_idx}")
            for line in block.get("lines", []):
                draw_geometry(draw, line_geometry(line), image.size, color, 2)
    image.save(path)


def save_detector_components(path: str, prob_map: np.ndarray, threshold: float) -> None:
    binary = binary_map(prob_map, threshold)
    num_labels, labels = cv2.connectedComponents(binary)
    output = np.ones((*labels.shape, 3), dtype=np.uint8) * 255
    rng = random.Random(17)
    for label_idx in range(1, num_labels):
        color = np.array([rng.randrange(40, 230), rng.randrange(40, 230), rng.randrange(40, 230)], dtype=np.uint8)
        output[labels == label_idx] = color
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 0, 0), 1)
    save_rgb(path, output)


def save_detector_word_boxes(path: str, page: np.ndarray, boxes: np.ndarray, objectness_scores: np.ndarray | None) -> None:
    image = Image.fromarray(page.copy())
    draw = ImageDraw.Draw(image)
    for idx, box in enumerate(np.asarray(boxes)):
        label = str(idx + 1)
        if objectness_scores is not None and idx < len(objectness_scores):
            label = f"{idx + 1}:{float(objectness_scores[idx]):.2f}"
        draw_geometry(draw, box, image.size, (47, 140, 70), 2, label)
    image.save(path)


@torch.inference_mode()
def run_pipeline(predictor: Any, pages: list[np.ndarray]) -> tuple[Document, dict[str, Any]]:
    if any(page.ndim != 3 for page in pages):
        raise ValueError("all pages must be RGB images")

    origin_page_shapes = [page.shape[:2] for page in pages]
    loc_preds_dicts, out_maps = predictor.det_predictor(pages, return_maps=True)

    seg_maps = [
        np.where(out_map > getattr(predictor.det_predictor.model.postprocessor, "bin_thresh"), 255, 0).astype(np.uint8)
        for out_map in out_maps
    ]

    if predictor.detect_orientation:
        general_pages_orientations, origin_pages_orientations = predictor._get_orientations(pages, seg_maps)
        orientations = [{"value": orientation_page, "confidence": None} for orientation_page in origin_pages_orientations]
    else:
        orientations = None
        general_pages_orientations = None
        origin_pages_orientations = None

    if predictor.straighten_pages:
        pages = predictor._straighten_pages(pages, seg_maps, general_pages_orientations, origin_pages_orientations)
        origin_page_shapes = [page.shape[:2] for page in pages]
        loc_preds_dicts, out_maps = predictor.det_predictor(pages, return_maps=True)
        seg_maps = [
            np.where(out_map > getattr(predictor.det_predictor.model.postprocessor, "bin_thresh"), 255, 0).astype(np.uint8)
            for out_map in out_maps
        ]

    if not all(len(loc_pred) == 1 for loc_pred in loc_preds_dicts):
        raise RuntimeError("docTR OCR predictor should output one detection class per page")

    loc_preds = [list(loc_pred.values())[0] for loc_pred in loc_preds_dicts]
    loc_preds, objectness_scores = detach_scores(loc_preds)

    for hook in predictor.hooks:
        loc_preds = hook(loc_preds)

    detector_boxes = [np.array(page_boxes, copy=True) for page_boxes in loc_preds]
    detector_scores = [np.array(page_scores, copy=True) for page_scores in objectness_scores]

    raw_crops, loc_preds = predictor._prepare_crops(
        pages,
        loc_preds,
        assume_straight_pages=predictor.assume_straight_pages,
        assume_horizontal=predictor._page_orientation_disabled,
    )
    final_crops = raw_crops

    crop_orientations: Any = []
    if not predictor.assume_straight_pages:
        final_crops, loc_preds, raw_crop_orientations = predictor._rectify_crops(raw_crops, loc_preds)
        crop_orientations = [{"value": orientation[0], "confidence": orientation[1]} for orientation in raw_crop_orientations]

    flat_crops = flatten_pages_crops(final_crops)
    word_preds = predictor.reco_predictor(flat_crops) if flat_crops else []
    if not crop_orientations:
        crop_orientations = [{"value": 0, "confidence": None} for _ in word_preds]

    boxes, text_preds, crop_orientations = predictor._process_predictions(loc_preds, word_preds, crop_orientations)

    if predictor.detect_language:
        languages = [get_language(" ".join([item[0] for item in text_pred])) for text_pred in text_preds]
        languages_dict = [{"value": lang[0], "confidence": lang[1]} for lang in languages]
    else:
        languages_dict = None

    document = predictor.doc_builder(
        pages,
        boxes,
        objectness_scores,
        text_preds,
        origin_page_shapes,
        crop_orientations,
        orientations,
        languages_dict,
    )

    diagnostics = {
        "pages": pages,
        "origin_page_shapes": origin_page_shapes,
        "out_maps": out_maps,
        "seg_maps": seg_maps,
        "detector_boxes": detector_boxes,
        "detector_scores": detector_scores,
        "raw_crops": raw_crops,
        "final_crops": final_crops,
        "word_preds": word_preds,
        "text_preds": text_preds,
        "boxes": boxes,
        "crop_orientations": crop_orientations,
        "orientations": orientations,
        "languages": languages_dict,
    }
    return document, diagnostics


def build_predictor(args: argparse.Namespace) -> Any:
    # DDL model/runtime/geometry/structure parameters that docTR accepts at construction time.
    # det_arch, reco_arch, preserve_aspect_ratio, symmetric_pad, assume_straight_pages,
    # export_as_straight_boxes, straighten_pages, detect_orientation, detect_language,
    # disable_page_orientation, disable_crop_orientation, det_bs, reco_bs,
    # resolve_lines, resolve_blocks, and paragraph_break are passed here.
    predictor = ocr_predictor(
        det_arch=args.det_arch,
        reco_arch=args.reco_arch,
        pretrained=True,
        assume_straight_pages=args.assume_straight_pages,
        preserve_aspect_ratio=args.preserve_aspect_ratio,
        symmetric_pad=args.symmetric_pad,
        export_as_straight_boxes=args.export_as_straight_boxes,
        detect_orientation=args.detect_orientation,
        straighten_pages=args.straighten_pages,
        detect_language=args.detect_language,
        disable_page_orientation=args.disable_page_orientation,
        disable_crop_orientation=args.disable_crop_orientation,
        det_bs=args.det_bs,
        reco_bs=args.reco_bs,
        resolve_lines=args.resolve_lines,
        resolve_blocks=args.resolve_blocks,
        paragraph_break=args.paragraph_break,
    )

    # DDL detector preprocessing parameter. The docTR default path is to leave
    # predictor.det_predictor.pre_processor.resize.size untouched.
    if args.det_input_size != "model_default":
        size = int(args.det_input_size)
        predictor.det_predictor.pre_processor.resize.size = (size, size)

    # DDL recognizer preprocessing parameters. docTR's public ocr_predictor()
    # does not expose these directly, but they are the recognizer Resize fields
    # created by recognition_predictor().
    predictor.reco_predictor.pre_processor.resize.preserve_aspect_ratio = args.reco_preserve_aspect_ratio
    predictor.reco_predictor.pre_processor.resize.symmetric_pad = args.reco_symmetric_pad

    # DDL detection post-processing parameters. docTR stores both thresholds on
    # the detector model postprocessor, so they are applied after construction.
    predictor.det_predictor.model.postprocessor.bin_thresh = args.bin_thresh
    predictor.det_predictor.model.postprocessor.box_thresh = args.box_thresh
    if hasattr(predictor.det_predictor.model.postprocessor, "unclip_ratio"):
        predictor.det_predictor.model.postprocessor.unclip_ratio = args.unclip_ratio
    return predictor


def selected_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "det_arch": args.det_arch,
        "reco_arch": args.reco_arch,
        "det_input_size": args.det_input_size,
        "preserve_aspect_ratio": args.preserve_aspect_ratio,
        "symmetric_pad": args.symmetric_pad,
        "reco_preserve_aspect_ratio": args.reco_preserve_aspect_ratio,
        "reco_symmetric_pad": args.reco_symmetric_pad,
        "assume_straight_pages": args.assume_straight_pages,
        "export_as_straight_boxes": args.export_as_straight_boxes,
        "straighten_pages": args.straighten_pages,
        "detect_orientation": args.detect_orientation,
        "detect_language": args.detect_language,
        "disable_page_orientation": args.disable_page_orientation,
        "disable_crop_orientation": args.disable_crop_orientation,
        "resolve_lines": args.resolve_lines,
        "resolve_blocks": args.resolve_blocks,
        "paragraph_break": args.paragraph_break,
        "bin_thresh": args.bin_thresh,
        "box_thresh": args.box_thresh,
        "unclip_ratio": args.unclip_ratio,
        "draw_labels": args.draw_labels,
        "draw_confidence": args.draw_confidence,
        "min_confidence_display": args.min_confidence_display,
        "recognizer_sample_count": args.recognizer_sample_count,
        "visualization_seed": args.visualization_seed,
    }


def write_hocr(document: Document) -> str:
    try:
        xml_exports = document.export_as_xml(file_title="docTR IPOL demo hOCR")
        if not xml_exports:
            raise RuntimeError("docTR returned no XML pages")
        xml_bytes = xml_exports[0][0]
        Path("result.hocr").write_bytes(xml_bytes)
        status = "hOCR generated successfully."
    except Exception as exc:  # noqa: BLE001
        status = f"hOCR export failed; placeholder file written. Error: {exc}"
        placeholder = "<html><body><!-- hOCR export failed; see hocr_error.txt --></body></html>\n"
        Path("result.hocr").write_text(placeholder, encoding="utf-8")
    Path("hocr_error.txt").write_text(status + "\n", encoding="utf-8")
    return status


def write_summary(
    args: argparse.Namespace,
    input_shape: tuple[int, int, int],
    ocr_shape: tuple[int, int, int],
    rows: list[dict[str, Any]],
    document_export: dict[str, Any],
    timings: dict[str, float],
    hocr_status: str,
) -> None:
    pages = export_pages(document_export)
    page_export = pages[0] if pages else {}
    block_count = len(page_export.get("blocks", []))
    line_count = sum(len(block.get("lines", [])) for block in page_export.get("blocks", []))
    text = [
        "docTR IPOL demo summary",
        "",
        f"Input shape: {input_shape[1]}x{input_shape[0]}",
        f"OCR page shape: {ocr_shape[1]}x{ocr_shape[0]}",
        f"Detector: {args.det_arch}",
        f"Recognizer: {args.reco_arch}",
        f"Detector input size: {args.det_input_size}",
        f"Words: {len(rows)}",
        f"Lines: {line_count}",
        f"Blocks: {block_count}",
        f"Orientation metadata: {page_export.get('orientation')}",
        f"Language metadata: {page_export.get('language')}",
        f"hOCR: {hocr_status}",
        "",
        "Timings (seconds):",
    ]
    for name, value in timings.items():
        text.append(f"- {name}: {value:.3f}")
    text.extend(["", "Parameters:"])
    for key, value in selected_params(args).items():
        text.append(f"- {key}: {value}")
    text.extend(["", "How parameters are applied:"])
    for key in selected_params(args):
        text.append(f"- {key}: {PARAMETER_EFFECTS[key]}")
    Path("summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")


def crop_labels(indices: list[int], word_preds: list[tuple[str, float]]) -> list[str]:
    labels: list[str] = []
    for index in indices:
        if index < len(word_preds):
            value, confidence = word_preds[index]
            labels.append(f"{index + 1}: {value} ({confidence:.2f})")
        else:
            labels.append(str(index + 1))
    return labels


def write_outputs(args: argparse.Namespace, input_page: np.ndarray, predictor: Any) -> None:
    timings: dict[str, float] = {}

    start = perf_counter()
    document, diagnostics = run_pipeline(predictor, [input_page])
    timings["ocr_pipeline"] = perf_counter() - start

    ocr_page = diagnostics["pages"][0]
    save_rgb("ocr_page.png", ocr_page)

    start = perf_counter()
    detector_preview = detector_input_preview(predictor.det_predictor.pre_processor, ocr_page)
    save_rgb("detector_input.png", detector_preview)

    prob_map = primary_map(diagnostics["out_maps"][0])
    save_rgb("detector_probability_map.png", heatmap_rgb(prob_map))
    save_rgb("detector_binary_map.png", binary_map(prob_map, args.bin_thresh))
    save_detector_components("detector_components.png", prob_map, args.bin_thresh)

    detector_boxes = diagnostics["detector_boxes"][0] if diagnostics["detector_boxes"] else np.empty((0, 4))
    detector_scores = diagnostics["detector_scores"][0] if diagnostics["detector_scores"] else None
    save_detector_word_boxes("detector_word_boxes.png", ocr_page, detector_boxes, detector_scores)
    timings["diagnostic_images_detector"] = perf_counter() - start

    document_export = document.export()
    rows = word_rows(document_export)
    write_words_csv(rows)
    save_json("result_raw_doctr.json", document_export)
    Path("result.txt").write_text(document.render(), encoding="utf-8")

    start = perf_counter()
    save_overlay("overlay_words.png", ocr_page, document_export, "words", args.draw_labels, args.draw_confidence, args.min_confidence_display)
    save_overlay("overlay_lines.png", ocr_page, document_export, "lines", args.draw_labels, args.draw_confidence, args.min_confidence_display)
    save_overlay("overlay_blocks.png", ocr_page, document_export, "blocks", args.draw_labels, args.draw_confidence, args.min_confidence_display)
    save_overlay("overlay_all.png", ocr_page, document_export, "all", args.draw_labels, args.draw_confidence, args.min_confidence_display)
    save_reading_order("reading_order_words.png", ocr_page, document_export)
    save_line_grouping("line_grouping.png", ocr_page, document_export)
    save_block_grouping("block_grouping.png", ocr_page, document_export)
    timings["diagnostic_images_structure"] = perf_counter() - start

    start = perf_counter()
    raw_crops = flatten_pages_crops(diagnostics["raw_crops"])
    final_crops = flatten_pages_crops(diagnostics["final_crops"])
    indices = sample_indices(len(final_crops), args.recognizer_sample_count, args.visualization_seed)
    sampled_raw = [raw_crops[index] for index in indices if index < len(raw_crops)]
    sampled_final = [final_crops[index] for index in indices if index < len(final_crops)]
    sampled_network = preprocessed_images(predictor.reco_predictor.pre_processor, sampled_final)
    labels = crop_labels(indices, diagnostics["word_preds"])
    make_vertical_stack(sampled_raw, labels, "recognizer_raw_crops_stack.png", "Raw word crops")
    make_vertical_stack(sampled_network, labels, "recognizer_input_crops_stack.png", "Recognizer network input crops")
    make_contact_sheet(sampled_network, labels, "recognizer_crops_contact_sheet.png")
    timings["diagnostic_images_recognizer"] = perf_counter() - start

    hocr_status = write_hocr(document)

    structure_debug = {
        "detector_boxes": diagnostics["detector_boxes"],
        "detector_objectness_scores": diagnostics["detector_scores"],
        "sampled_crop_indices": indices,
        "word_predictions_flat": diagnostics["word_preds"],
        "word_rows": rows,
        "document_structure": document_export,
        "diagnostic_notes": {
            "detector_probability_map": "First channel of the detector response map.",
            "detector_binary_map": f"Detector response thresholded at bin_thresh={args.bin_thresh}.",
            "recognizer_input_crops_stack": "Sampled crops after docTR recognizer preprocessing and denormalization for display.",
        },
    }
    save_json("structure_debug.json", structure_debug)

    full_result = {
        "metadata": {
            "demo": "docTR IPOL demo",
            "runtime": "CPU",
            "input_file": str(args.input),
            "input_shape": input_page.shape,
            "ocr_page_shape": ocr_page.shape,
            "output_files": OUTPUT_IMAGE_FILES + OUTPUT_TEXT_FILES,
        },
        "parameters": selected_params(args),
        "parameter_effects": {key: PARAMETER_EFFECTS[key] for key in selected_params(args)},
        "counts": {
            "pages": len(export_pages(document_export)),
            "words": len(rows),
            "detector_boxes": int(len(detector_boxes)),
            "raw_crops": len(raw_crops),
            "recognizer_crops": len(final_crops),
        },
        "timings_seconds": timings,
        "hocr_status": hocr_status,
        "doctr": document_export,
    }
    save_json("result.json", full_result)
    write_summary(args, input_page.shape, ocr_page.shape, rows, document_export, timings, hocr_status)


def write_failure_outputs(args: argparse.Namespace, exc: BaseException) -> None:
    message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    for path in OUTPUT_IMAGE_FILES:
        save_placeholder(path, "docTR demo error", message)
    error_payload = {
        "error": str(exc),
        "traceback": message,
        "parameters": selected_params(args),
    }
    save_json("result.json", error_payload)
    save_json("result_raw_doctr.json", {})
    save_json("structure_debug.json", error_payload)
    Path("result.txt").write_text("", encoding="utf-8")
    Path("summary.txt").write_text("docTR demo failed\n\n" + message, encoding="utf-8")
    with open("words.csv", "w", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=CSV_HEADER).writeheader()
    Path("result.hocr").write_text("<html><body><!-- docTR demo failed --></body></html>\n", encoding="utf-8")
    Path("hocr_error.txt").write_text("hOCR was not generated because the run failed.\n", encoding="utf-8")


def ensure_referenced_files() -> None:
    for path in OUTPUT_IMAGE_FILES:
        if not Path(path).exists():
            save_placeholder(path, Path(path).stem, "This output was not produced.")
    for path in OUTPUT_TEXT_FILES:
        if not Path(path).exists():
            if path.endswith(".json"):
                save_json(path, {})
            elif path == "words.csv":
                with open(path, "w", encoding="utf-8", newline="") as file:
                    csv.DictWriter(file, fieldnames=CSV_HEADER).writeheader()
            else:
                Path(path).write_text("", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    try:
        if input_path.resolve() != Path("input_0.png").resolve():
            shutil.copyfile(input_path, "input_0.png")
        input_page = load_rgb(input_path)
        predictor = build_predictor(args)
        write_outputs(args, input_page, predictor)
        ensure_referenced_files()
        return 0
    except Exception as exc:  # noqa: BLE001
        write_failure_outputs(args, exc)
        ensure_referenced_files()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
