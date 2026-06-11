# docTR IPOL demo: inputs, parameters, outputs

This document is the working contract for the IPOL demo. It records the
reasonable knobs to expose, the outputs to generate, and the implementation
constraints that should guide the first public version.

## Scope

Goal: expose a reproducible CPU-first docTR OCR demo for one uploaded document
image, with configurable model choices, geometry/orientation options,
post-processing thresholds, document-structure options, and downloadable OCR
results.

Initial target:

- one image input, converted by IPOL to `input_0.png`;
- the DDL `run` command passes that input as `$input_0`, which IPOL resolves to
  the execution-directory path before running the container command;
- PyTorch CPU execution in `registry.ipol.im/ipol:v2-py3.11-pytorch`;
- pretrained docTR models downloaded during Docker build;
- fixed output filenames in `/workdir/exec`;
- no runtime network dependency.

Out of scope for the first version:

- arbitrary custom weights uploaded by users;
- GPU execution;
- multi-page PDF input;
- training/fine-tuning;
- free-form model names typed by the user;
- KIE/layout-specific docTR predictors, unless we explicitly add a separate mode.

## IPOL Input Files

### Primary Input: Document Image

DDL input:

```json
{
  "description": "input document image",
  "max_pixels": "3000*3000",
  "max_weight": "25*1024*1024",
  "dtype": "3x8i",
  "ext": ".png",
  "type": "image"
}
```

Recommended defaults:

- `type`: `image`
- `dtype`: `3x8i`
- `ext`: `.png`
- `max_pixels`: `3000*3000` for the first version.
- `max_weight`: optional, around `25*1024*1024`.

Rationale:

- docTR accepts images and PDFs, but IPOL image inputs give format checking,
  color conversion, and pixel limits.
- A single page keeps result filenames simple and avoids dynamic result loops.
- CPU OCR on very large pages can be slow; IPOL is not a compute cluster.

### Optional Future Input: PDF

Potential DDL input:

```json
{
  "description": "input PDF document",
  "max_weight": "25*1024*1024",
  "ext": ".pdf",
  "type": "data"
}
```

Do not add this in the first version unless we also solve:

- multi-page result naming;
- DDL result display with repeat expressions;
- page count limits;
- output archive size;
- PDF rasterization resolution.

## Parameter Groups

IPOL has no native collapsible advanced section, so use `label` parameters to
separate groups and keep names clear. Prefer `selection_radio` controls over
free text for model names because the current IPOL client renders that format
reliably.

For checkbox defaults, keep both `default` and `default_value` with the same
boolean value. The current IPOL client initializes checkbox state from
`default_value`, while older examples and documentation use `default`.

For numeric parameters, include `values.step` even though the DDL examples do
not require it. The current IPOL client writes that field directly into the
HTML input `step` attribute.

## Detector and Recognizer Model Parameters

### `det_arch`

Purpose: choose text detection architecture.

DDL type: selection_radio.

User-facing explanation:

> Selects the neural network used to localize text regions before recognition.
> FAST models are usually good CPU defaults; DBNet and LinkNet are useful for
> comparing older or differently behaved detectors.

Allowed values:

- `fast_base` default, good modern default and reasonable CPU latency.
- `fast_small`, faster/lighter variant.
- `fast_tiny`, fastest/lighter variant.
- `db_resnet50`, common historical docTR example.
- `db_resnet34`, lighter DBNet.
- `db_mobilenet_v3_large`, mobile DBNet variant.
- `linknet_resnet18`, useful for rotated/non-straight page tests.
- `linknet_resnet34`, heavier LinkNet variant.
- `linknet_resnet50`, heaviest LinkNet variant exposed by docTR 1.0.1.

The DDL intentionally exposes the full public docTR 1.0.1 detection set, and
`preload_models.py` downloads weights for all of them during Docker build.

CLI mapping:

```bash
--det-arch $det_arch
```

docTR mapping:

```python
ocr_predictor(det_arch=args.det_arch, ...)
```

### `reco_arch`

Purpose: choose text recognition architecture.

DDL type: selection_radio.

User-facing explanation:

> Selects the neural network used to read each detected word crop. Lightweight
> CRNN models are faster; transformer-style recognizers such as PARSeq or ViTSTR
> may improve difficult words but are heavier.

Allowed values:

- `crnn_vgg16_bn` default, stable docTR baseline.
- `crnn_mobilenet_v3_small`, faster/lighter.
- `crnn_mobilenet_v3_large`, still light and often useful.
- `parseq`, strong transformer-style recognizer.
- `master`, heavier attention-based recognizer.
- `sar_resnet31`, heavier recognizer.
- `vitstr_small`, compact ViT recognizer.
- `vitstr_base`, heavier ViT recognizer.
- `viptr_tiny`, VIPTR recognizer variant exposed by docTR 1.0.1.

The DDL intentionally exposes the full public docTR 1.0.1 recognition set, and
`preload_models.py` downloads weights for all of them during Docker build.

CLI mapping:

```bash
--reco-arch $reco_arch
```

docTR mapping:

```python
ocr_predictor(reco_arch=args.reco_arch, ...)
```

## Detector and Recognizer Preprocessing Parameters

### `det_input_size`

Purpose: set the square image size used by docTR's detector preprocessor.

DDL type: numeric.

Recommended values:

- `512`
- `768`
- `1024`
- `1280`
- `1536`

Default: `1024`, matching the detector resize size used by docTR 1.0.1 for
every detector exposed in this demo.

User-facing explanation:

> Size, in pixels, of the square tensor sent to the text detector. Default 1024
> matches docTR 1.0.1 for the exposed detectors. Recommended values to try: 512,
> 768, 1024, 1280, 1536. Smaller values are faster but may miss small text;
> larger values can recover small text but cost more CPU time and memory.

Implementation:

```python
args.det_input_size = normalize_det_input_size(args.det_input_size)
size = int(args.det_input_size)
predictor.det_predictor.pre_processor.resize.size = (size, size)
```

Visualization impact:

- `detector_input.png` must show the de-normalized detector tensor after docTR
  resizing/padding, so users can see exactly what this parameter changed.

First version decision: expose this parameter. Do not add a separate
`page_max_side`; it is redundant for this demo and hides the actual detector
input size.

### `preserve_aspect_ratio`

Purpose: preserve page aspect ratio while resizing before detector inference.

DDL type: checkbox.

Default: `true`.

User-facing explanation:

> Resize the page to fit inside the detector square without distorting the
> document. If disabled, the page is stretched to the detector input size.

docTR mapping:

```python
ocr_predictor(preserve_aspect_ratio=args.preserve_aspect_ratio, ...)
```

Note: docTR does not expose an independent "enable padding" flag. Padding is the
consequence of preserving aspect ratio when the page ratio differs from the
detector tensor ratio.

### `symmetric_pad`

Purpose: detector symmetric padding. If detector aspect ratio is preserved, pad
symmetrically rather than only on one side.

DDL type: checkbox.

Default: `true`.

User-facing explanation:

> When preserving aspect ratio, add padding on both sides rather than only at
> the bottom/right. This changes where the page content lies inside the detector
> square and can slightly affect detection near borders.

docTR mapping:

```python
ocr_predictor(symmetric_pad=args.symmetric_pad, ...)
```

### `reco_preserve_aspect_ratio`

Purpose: recognizer padding switch. Preserve aspect ratio while resizing word
crops for the recognition network.

DDL type: checkbox.

Default: `true`, matching docTR's recognition preprocessor.

User-facing explanation:

> Resize word crops to the recognizer tensor without distorting the crop. If
> disabled, crops are stretched to the recognizer input size.

docTR mapping after predictor creation:

```python
predictor.reco_predictor.pre_processor.resize.preserve_aspect_ratio = args.reco_preserve_aspect_ratio
```

### `reco_symmetric_pad`

Purpose: recognizer symmetric padding. If recognizer aspect ratio is preserved,
pad word crops symmetrically rather than only at the bottom/right.

DDL type: checkbox.

Default: `false`, matching `recognition_predictor(..., symmetric_pad=False)`.

User-facing explanation:

> When recognizer padding is enabled, place padding on both sides of each word
> crop instead of only after the crop content.

docTR mapping after predictor creation:

```python
predictor.reco_predictor.pre_processor.resize.symmetric_pad = args.reco_symmetric_pad
```

### `split_wide_crops`

Purpose: recognizer wide-crop post-processing switch. When enabled, docTR
splits very wide detector crops into overlapping sub-crops before recognition
and remaps the sub-crop predictions back into one word prediction.

DDL type: checkbox.

Default: `true`, matching docTR's `RecognitionPredictor` default.

User-facing explanation:

> Split very wide word crops into overlapping sub-crops before recognition,
> then merge the sub-crop predictions.

docTR mapping after predictor creation:

```python
predictor.reco_predictor.split_wide_crops = args.split_wide_crops
```

Diagnostic output: `recognizer_split_crops.png` shows the original recognizer
crop, the split boundaries, and the sub-crops sent to the recognition network.
`recognizer_split_overlay.png` shows the same recognizer boxes on the detector
canvas and marks split boxes in red.

### `crop_split_critical_ar`

Purpose: width/height threshold above which a recognizer crop is considered
wide enough to split.

DDL type: range.

Default: `8`, matching docTR.

Suggested range:

- min: `2`
- max: `20`
- step: `0.5`

User-facing explanation:

> Recognizer crops with width/height above this threshold are split when
> wide-crop splitting is enabled.

docTR mapping after predictor creation:

```python
predictor.reco_predictor.critical_ar = args.crop_split_critical_ar
```

### `crop_split_target_ar`

Purpose: target width/height ratio for each sub-crop produced by the wide-crop
splitter.

DDL type: numeric.

Default: `6`, matching docTR.

Suggested range:

- min: `1`
- max: `20`

User-facing explanation:

> Target width/height ratio for each recognizer sub-crop created from a wide
> word crop.

docTR mapping after predictor creation:

```python
predictor.reco_predictor.target_ar = args.crop_split_target_ar
```

### `crop_split_overlap_ratio`

Purpose: overlap fraction between adjacent sub-crops created from one wide
recognizer crop.

DDL type: range.

Default: `0.5`, matching docTR.

Suggested range:

- min: `0.05`
- max: `0.95`
- step: `0.05`

User-facing explanation:

> Horizontal overlap fraction between adjacent recognizer sub-crops.

docTR mapping after predictor creation:

```python
predictor.reco_predictor.overlap_ratio = args.crop_split_overlap_ratio
```

## Detector Geometry and Orientation Parameters

### `assume_straight_pages`

Purpose: use straight text boxes and skip rotated crop orientation logic.

DDL type: checkbox.

Default: `true`.

User-facing explanation:

> Use for normal scans/photos with horizontal text. Disable for rotated or
> skewed text regions; this may return polygon boxes and is slower.

docTR mapping:

```python
ocr_predictor(assume_straight_pages=args.assume_straight_pages, ...)
```

### `export_as_straight_boxes`

Purpose: when rotated/polygon mode is enabled, export final predictions as
axis-aligned boxes.

DDL type: checkbox.

Default: `false`.

User-facing explanation:

> Convert rotated/polygon boxes into ordinary rectangular boxes in the exported
> JSON and hOCR. This is easier to consume downstream, but loses the exact
> rotated geometry.

Visibility: only useful when `assume_straight_pages` is false.

docTR mapping:

```python
ocr_predictor(export_as_straight_boxes=args.export_as_straight_boxes, ...)
```

Output impact:

- `true`: easier hOCR/XML and rectangular JSON.
- `false`: preserves rotated quadrilaterals when available.

### `straighten_pages`

Purpose: estimate page-level orientation/skew and rerun detection on a
straightened page.

DDL type: checkbox.

Default: `false`.

User-facing explanation:

> Try to rotate the page into a straighter orientation before OCR. This can help
> uniformly rotated pages, but it is slower because detection may be executed
> more than once.

docTR mapping:

```python
ocr_predictor(straighten_pages=args.straighten_pages, ...)
```

Preload impact:

- downloads page orientation classifier weights.

Runtime impact:

- slower, because detection may run again.

### `detect_orientation`

Purpose: add estimated page orientation to each page output.

DDL type: checkbox.

Default: `false`.

User-facing explanation:

> Estimate and report the page orientation in the JSON output. This is mostly
> diagnostic unless combined with page straightening.

docTR mapping:

```python
ocr_predictor(detect_orientation=args.detect_orientation, ...)
```

Preload impact:

- downloads page orientation classifier weights.

Output impact:

- `result.json` page object contains orientation value and confidence.
- `summary.txt` should report orientation.

### `disable_page_orientation`

Purpose: expert speed option for workflows that do not need page orientation
classification even though rotated-page logic is active.

DDL type: checkbox.

Default: `false`.

User-facing explanation:

> Expert option. Skip the page-orientation classifier when rotated-page logic is
> active. This can be faster, but orientation metadata and straightening quality
> may be worse.

Visibility: expert only; useful with `assume_straight_pages=false`,
`straighten_pages=true`, or `detect_orientation=true`.

docTR mapping:

```python
ocr_predictor(disable_page_orientation=args.disable_page_orientation, ...)
```

First version recommendation: include only if we expose an "expert" group.

### `disable_crop_orientation`

Purpose: expert speed option for non-straight mode when crop orientation
classification is unnecessary.

DDL type: checkbox.

Default: `false`.

User-facing explanation:

> Expert option. Skip orientation correction of individual detected word crops.
> This is faster in non-straight mode, but rotated words may be recognized less
> accurately.

Visibility: useful only when `assume_straight_pages=false`.

docTR mapping:

```python
ocr_predictor(disable_crop_orientation=args.disable_crop_orientation, ...)
```

First version recommendation: include only if we expose an "expert" group.

### `detect_language`

Purpose: add document-level language prediction from recognized text.

DDL type: checkbox.

Default: `false`.

User-facing explanation:

> Infer the language from the recognized text and store it in the JSON summary.
> This does not change OCR predictions; it only adds metadata.

docTR mapping:

```python
ocr_predictor(detect_language=args.detect_language, ...)
```

Output impact:

- `result.json` page object contains language value and confidence.
- `summary.txt` should report language.

## Post-processing Structure Parameters

### `resolve_lines`

Purpose: group recognized words into lines.

DDL type: checkbox.

Default: `true`.

User-facing explanation:

> Group detected words into text lines after recognition. Disable this to inspect
> a flatter word-level structure.

docTR mapping:

```python
ocr_predictor(resolve_lines=args.resolve_lines, ...)
```

Output impact:

- if false, line-level output may be less meaningful.

First version recommendation: expose.

### `resolve_blocks`

Purpose: group lines into document blocks.

DDL type: checkbox.

Default: `false`.

User-facing explanation:

> Group resolved lines into larger text blocks or paragraph-like regions. Enable
> this to inspect docTR's block post-processing and block-level boxes.

docTR mapping:

```python
ocr_predictor(resolve_blocks=args.resolve_blocks, ...)
```

Output impact:

- `blocks.png` is most useful when true.
- If false, docTR still returns block containers, but they may not correspond
  to paragraphs/regions.

First version recommendation: expose because "use blocks or not" is one of the
main desired controls.

### `paragraph_break`

Purpose: threshold for splitting horizontal gaps into paragraphs/sub-lines.

DDL type: range.

Default: `0.035`.

User-facing explanation:

> Relative gap threshold used while grouping words and lines. Smaller values
> split text more aggressively; larger values merge words/lines across wider
> spaces.

Suggested range:

- min: `0.005`
- max: `0.20`
- step: `0.005`

docTR mapping:

```python
ocr_predictor(paragraph_break=args.paragraph_break, ...)
```

Visibility: useful when `resolve_lines=true`; more meaningful with
`resolve_blocks=true`.

## Post-processing Detection Parameters

### `det_threshold_mode`

Purpose: choose whether detector thresholds come from the selected docTR model
or from the custom sliders.

DDL type: selection_radio.

Default: `model_default`.

Allowed values:

- `model_default`: keep docTR's own postprocessor defaults for the selected
  detector architecture.
- `custom`: apply `bin_thresh`, `box_thresh`, and `unclip_ratio` after model
  construction.

User-facing explanation:

> Use detector defaults to keep the thresholds defined by docTR for the selected
> detector. Use custom values to override them with the sliders below. When
> detector defaults is selected, the three slider values are ignored.

Implementation:

```python
post = predictor.det_predictor.model.postprocessor
model_defaults = detector_postprocessor_state(predictor)
if args.det_threshold_mode == "custom":
    post.bin_thresh = args.bin_thresh
    post.box_thresh = args.box_thresh
    post.unclip_ratio = args.unclip_ratio
```

The run exports both `model_defaults` and `applied` values in `result.json` and
`structure_debug.json`.

### `bin_thresh`

Purpose: threshold for detector binarization map.

DDL type: range.

Default slider value: `0.1`, matching FAST detector defaults. This value is
used only when `det_threshold_mode=custom`; otherwise docTR's selected-detector
default is used.

User-facing explanation:

> Threshold applied to the detector probability map before extracting candidate
> text components. Lower values are more permissive; higher values keep only
> stronger detector responses.

Suggested range:

- min: `0.01`
- max: `0.95`
- step: `0.01`

docTR mapping after predictor creation, only in custom mode:

```python
predictor.det_predictor.model.postprocessor.bin_thresh = args.bin_thresh
```

### `box_thresh`

Purpose: threshold for keeping detected boxes.

DDL type: range.

Default: `0.1`, matching docTR's `fast_base` detector postprocessor default.
This value is used only when `det_threshold_mode=custom`.

User-facing explanation:

> Minimum score required to keep a detected text box after detector
> post-processing. Lower values can add weak boxes; higher values remove
> uncertain detections.

Suggested range:

- min: `0.01`
- max: `0.95`
- step: `0.01`

docTR mapping after predictor creation, only in custom mode:

```python
predictor.det_predictor.model.postprocessor.box_thresh = args.box_thresh
```

### `unclip_ratio`

Purpose: detector box expansion ratio used by post-processing after connected
components are extracted.

DDL type: range.

Default slider value: `1.0`, matching FAST detector defaults. This value is used
only when `det_threshold_mode=custom`; DBNet and LinkNet keep their own `1.5`
default in model-default mode.

Suggested range:

- min: `0.5`
- max: `3.0`
- step: `0.05`

docTR mapping after predictor creation, only in custom mode:

```python
predictor.det_predictor.model.postprocessor.unclip_ratio = args.unclip_ratio
```

## Visualization Parameters

These parameters affect only generated overlays, not OCR computation.

### `draw_labels`

Purpose: draw recognized words on `overlay_words.png` and `overlay_all.png`.

DDL type: checkbox.

Default: `true`.

User-facing explanation:

> Write recognized word strings next to word boxes in the overlay images.

### `draw_confidence`

Purpose: include confidence in word labels.

DDL type: checkbox.

Default: `false`.

User-facing explanation:

> Append recognition confidence values to word labels in the visual overlays.

### `min_confidence_display`

Purpose: hide low-confidence words in overlays only.

DDL type: range.

Default: `0.0`.

User-facing explanation:

> Minimum recognition confidence for drawing a word in diagnostic overlays. This
> does not remove the word from JSON, CSV, hOCR, or text outputs.

Suggested range:

- min: `0`
- max: `1`
- step: `0.01`

Important: do not filter `result_raw_doctr.json` with this parameter. If we
create `result_filtered.json`, make the difference explicit.

### `recognizer_sample_count`

Purpose: number of detected word crops to show in recognizer diagnostic
visualizations.

DDL type: numeric.

Default: `24`.

Suggested range:

- min: `1`
- max: `64`

User-facing explanation:

> Number of word crops sampled for the recognizer stack/contact-sheet
> visualizations. This only changes diagnostic images, not OCR results.

Implementation:

- sample without replacement when there are more crops than the requested count;
- use `visualization_seed` for reproducibility;
- use the same sampled indices in `recognizer_raw_crops_stack.png`,
  `recognizer_split_crops.png`, `recognizer_input_crops_stack.png`, and
  `recognizer_crops_contact_sheet.png`.
- prioritize crops that are actually split by `split_wide_crops`, then fill the
  remaining slots with deterministic random crops.
- when `split_wide_crops` creates multiple network crops from one detector
  crop, `recognizer_input_crops_stack.png` and
  `recognizer_crops_contact_sheet.png` show the post-split network inputs.

### `visualization_seed`

Purpose: deterministic seed for selecting crop examples in visual diagnostics.

DDL type: numeric.

Default: `0`.

Suggested range:

- min: `0`
- max: `999999`

User-facing explanation:

> Random seed used only to choose which detected word crops appear in the
> recognizer diagnostic images. Keeping it fixed makes archived runs
> reconstructible.

### `overlay_line_width`

Purpose: bounding box line thickness.

DDL type: range or numeric.

Default: `2`.

User-facing explanation:

> Thickness, in pixels, of bounding box outlines in generated overlay images.

Suggested range:

- min: `1`
- max: `8`
- step: `1`

First version recommendation: hardcode unless visual tests show a need.

## Recommended First DDL Parameter Set

This is the smallest useful set that still exposes the requested behavior:

- `det_arch`
- `reco_arch`
- `det_input_size`
- `assume_straight_pages`
- `export_as_straight_boxes`
- `straighten_pages`
- `detect_orientation`
- `detect_language`
- `preserve_aspect_ratio`
- `symmetric_pad`
- `reco_preserve_aspect_ratio`
- `reco_symmetric_pad`
- `resolve_lines`
- `resolve_blocks`
- `paragraph_break`
- `det_threshold_mode`
- `bin_thresh`
- `box_thresh`
- `unclip_ratio`
- `draw_labels`
- `draw_confidence`
- `min_confidence_display`
- `recognizer_sample_count`
- `visualization_seed`

- `disable_page_orientation`
- `disable_crop_orientation`

Optional first-version extras:

- `overlay_line_width`

## Output Files

All outputs must be written in the current execution directory, next to
`input_0.png`.

### Always Generated

#### `detector_input.png`

De-normalized visualization of the tensor sent to the detector after docTR's
resize and padding step.

This is the main image-size diagnostic. It shows the effect of:

- `det_input_size`;
- `preserve_aspect_ratio`;
- `symmetric_pad`.

Display: gallery.

Archive: yes.

#### `detector_probability_map.png`

Continuous detector response map before binarization. This is the closest
visualization to the raw detector output exposed by docTR.

For side-by-side comparison, save this image with the same pixel dimensions as
`detector_input.png`.

For multi-channel detector maps, save:

- the primary text/objectness channel as the default image;
- any extra channel metadata in `structure_debug.json` if needed.

Display: gallery.

Archive: yes.

#### `detector_binary_map.png`

Detector response after applying the effective detector `bin_thresh`.

For side-by-side comparison, save this image with the same pixel dimensions as
`detector_input.png`.

Purpose:

- explain why lowering `bin_thresh` creates more candidate text components;
- explain why raising `bin_thresh` removes weak regions.

Display: gallery.

Archive: yes.

#### `detector_components.png`

Connected components or contours extracted from `detector_binary_map.png`,
overlaid on the detector input.

For side-by-side comparison, save this image with the same pixel dimensions as
`detector_input.png`.

Purpose:

- make detector post-processing visible before final box filtering;
- show which detector blobs can become word boxes.

Display: gallery.

Archive: yes.

#### `detector_word_boxes.png`

Post-processed detector boxes, before recognition, overlaid on the detector
diagnostic canvas. The page content is mapped into the same padded/resized
layout as `detector_input.png` so all detector-panel images have matching
dimensions.

This is the key bridge between detection and recognition:

- every box shown here should correspond to a crop candidate sent to the
  recognizer, unless the crop is empty or filtered by docTR internals;
- optionally draw an integer crop index to match the crop stack outputs.

Display: gallery.

Archive: yes.

#### `recognizer_raw_crops_stack.png`

Vertical stack of a deterministic random sample of word crops immediately after
cropping from the page, before recognizer resizing and normalization.

Sampling policy:

- fixed seed, e.g. `0`, for reproducibility;
- maximum `recognizer_sample_count`, default `24`;
- if there are fewer crops, show all crops.

Display: gallery.

Archive: yes.

#### `recognizer_split_overlay.png`

Page-level overlay on the same detector diagnostic canvas as
`detector_word_boxes.png`.

This image is the bridge between detector and recognizer:

- green boxes are recognizer crops kept as single crops;
- red boxes are recognizer crops split by `split_wide_crops`;
- red internal lines show the horizontal sub-crop boundaries used before
  recognizer preprocessing;
- labels on red boxes show how many network crops are created from one detector
  crop.

Display: gallery.

Archive: yes.

#### `recognizer_split_crops.png`

Diagnostic view of the recognizer crop post-processing applied after detector
boxes are converted into word crops.

For each sampled crop, the image shows:

- the original detector crop that enters the recognizer stage;
- its width/height ratio;
- whether docTR keeps it as one crop or splits it with `split_wide_crops`;
- red split boundaries on wide crops;
- the overlapping sub-crops that will be passed to the recognition network.

This output explains the effect of `split_wide_crops`,
`crop_split_critical_ar`, `crop_split_target_ar`, and
`crop_split_overlap_ratio`.

Display: gallery.

Archive: yes.

#### `recognizer_input_crops_stack.png`

Vertical stack of the sampled crops after wide-crop splitting and recognizer
preprocessing, de-normalized for display. This shows the image tensors as they
enter the recognition network, including resize, padding, aspect-ratio effects,
and additional sub-crops created by `split_wide_crops`.

This is the visualization requested for "palabras tal cual entran a la red".

Display: gallery.

Archive: yes.

#### `recognizer_crops_contact_sheet.png`

Grid/contact sheet of the sampled recognizer input crops with crop indices and,
after inference, recognized text and confidence.

Purpose:

- quickly inspect many recognizer inputs at once;
- match crop quality with recognition confidence.

Display: gallery.

Archive: yes.

#### `reading_order_words.png`

Word boxes overlaid with reading-order indices.

Purpose:

- inspect docTR's word sorting;
- understand why rendered text appears in a given order.

Display: gallery.

Archive: yes.

#### `line_grouping.png`

Words colored by resolved line, with line boxes and line indices.

Purpose:

- explain how word detections are grouped into lines;
- show the effect of `resolve_lines` and `paragraph_break`.

Display: gallery.

Archive: yes.

#### `block_grouping.png`

Lines colored by resolved block, with block boxes and block indices.

Purpose:

- explain how lines are grouped into blocks;
- show the effect of `resolve_blocks`.

Display: gallery.

Archive: yes.

#### `overlay_words.png`

Input image with word-level boxes.

Color convention:

- words: blue.

Labels:

- controlled by `draw_labels`;
- optionally include confidence if `draw_confidence` is added.

Display: gallery.

Archive: yes.

#### `overlay_lines.png`

Input image with line-level boxes.

Color convention:

- lines: red.

Display: gallery.

Archive: yes.

#### `overlay_blocks.png`

Input image with block-level boxes.

Color convention:

- blocks: green.

Display: gallery.

Archive: yes.

#### `overlay_all.png`

Combined overlay:

- blocks: green;
- lines: red;
- words: blue.

Display: gallery.

Archive: yes.

#### `result_raw_doctr.json`

Exact `result.export()` output from docTR.

Coordinate convention:

- coordinates are relative to page size;
- straight boxes use `((xmin, ymin), (xmax, ymax))`;
- rotated boxes/polygons may use four points.

Display:

- `text_file` if small enough;
- `file_download` always.

Archive: yes.

#### `result.json`

Enriched demo output with metadata and raw docTR result.

Recommended top-level shape:

```json
{
  "metadata": {
    "demo": "doctr_ipol_demo",
    "doctr_version": "...",
    "params": {},
    "input": {
      "original_filename": "input_0.png",
      "original_size": [0, 0],
      "processed_filename": null,
      "processed_size": null,
      "detector_input_filename": "detector_input.png",
      "detector_input_size": [0, 0]
    },
    "diagnostics": {
      "detector_probability_map": "detector_probability_map.png",
      "detector_binary_map": "detector_binary_map.png",
      "detector_components": "detector_components.png",
      "detector_word_boxes": "detector_word_boxes.png",
      "recognizer_sample_indices": []
    },
    "timings": {
      "load_seconds": 0.0,
      "detector_debug_seconds": 0.0,
      "inference_seconds": 0.0,
      "render_seconds": 0.0,
      "total_seconds": 0.0
    },
    "counts": {
      "pages": 1,
      "blocks": 0,
      "lines": 0,
      "words": 0
    }
  },
  "doctr": {}
}
```

Display:

- `file_download`.

Archive: yes.

#### `result.txt`

Plain text OCR output from `result.render()`.

Display: `text_file`.

Archive: yes.

#### `summary.txt`

Human-readable run summary:

- selected models;
- major parameters;
- input dimensions and detector input dimensions;
- counts;
- detected orientation/language if enabled;
- runtime.

Display: `text_file`.

Archive: yes.

#### `structure_debug.json`

Structured diagnostic file explaining the hierarchy built from detector boxes:

- detector boxes before recognition;
- crop indices and sampled recognizer crops;
- recognized words;
- word-to-line assignment;
- line-to-block assignment;
- line and block geometries;
- geometry in both relative coordinates and pixel coordinates.

Purpose:

- make the post-processing from boxes to lines and blocks auditable without
  relying only on images;
- connect `detector_word_boxes.png`, crop stack images, `line_grouping.png`,
  `block_grouping.png`, and final JSON.

Display:

- `file_download`.

Archive: yes.

#### `words.csv`

Flat word table for downstream use.

Suggested columns:

- `page_idx`
- `block_idx`
- `line_idx`
- `word_idx`
- `value`
- `confidence`
- `objectness_score`
- `crop_orientation_value`
- `crop_orientation_confidence`
- `geometry_type`
- `geometry_relative_json`
- `geometry_pixel_json`

Display:

- `file_download`.

Archive: yes.

#### `result.hocr`

HTML hOCR/XML export.

First version decision: include hOCR support.

Generation rule:

- `assume_straight_pages=true`; or
- `export_as_straight_boxes=true`; or
- the exported geometry is otherwise compatible with docTR's hOCR exporter.

If hOCR export is not possible, still create `hocr_error.txt`, record the issue
in `summary.txt`, and keep JSON/text outputs as the authoritative result.

Display:

- `file_download`.

Archive: yes.

### Conditionally Generated

#### `processed_input.png`

Image actually sent to docTR before its internal detector preprocessing, only
if the wrapper performs an external image transformation.

Current decision:

- omit this file when no wrapper-level resize or conversion is applied;
- do not create a redundant copy of `input_0.png`;
- use `detector_input.png` to understand detector resizing/padding.

Display:

- no first-version DDL display unless wrapper preprocessing is later added.

Archive:

- only if generated.

#### `detector_extra_maps.png`

Additional detector maps or channels, if the selected detector exposes more than
the primary probability map.

Status: optional/debug output.

Implementation note:

- `detector_probability_map.png` and `detector_binary_map.png` are the required
  detector diagnostics;
- this optional file is only for extra channels or architecture-specific maps
  that are useful after implementation/testing.

First version recommendation:

- do not add a separate DDL control;
- generate only if it is cheap and informative.

#### `result_filtered.json`

Output filtered by `min_confidence_display` or a future OCR confidence
threshold.

First version recommendation:

- avoid, to keep raw OCR output unambiguous.

If added:

- never replace `result_raw_doctr.json`;
- record the filter threshold in metadata.

## DDL Results Plan

Recommended result blocks:

```json
[
  {
    "type": "gallery",
    "label": "<h3>Input and final overlays</h3>",
    "contents": {
      "Input": { "img": "input_0.png" },
      "Words": { "img": "overlay_words.png" },
      "Lines": { "img": "overlay_lines.png" },
      "Blocks": { "img": "overlay_blocks.png" },
      "All boxes": { "img": "overlay_all.png" }
    }
  },
  {
    "type": "gallery",
    "label": "<h3>Detector diagnostics</h3>",
    "contents": {
      "Detector input": { "img": "detector_input.png" },
      "Detector response": { "img": "detector_probability_map.png" },
      "Binarized response": { "img": "detector_binary_map.png" },
      "Detector components": { "img": "detector_components.png" },
      "Boxes sent to recognizer": { "img": "detector_word_boxes.png" }
    }
  },
  {
    "type": "gallery",
    "label": "<h3>Recognizer inputs and crop post-processing</h3>",
    "contents": {
      "Boxes sent to recognizer": { "img": "detector_word_boxes.png" },
      "Recognizer split overlay": { "img": "recognizer_split_overlay.png" },
      "Raw word crops": { "img": "recognizer_raw_crops_stack.png" },
      "Split crop details": { "img": "recognizer_split_crops.png" },
      "Network input crops": { "img": "recognizer_input_crops_stack.png" },
      "Crop contact sheet": { "img": "recognizer_crops_contact_sheet.png" }
    }
  },
  {
    "type": "gallery",
    "label": "<h3>Document structure diagnostics</h3>",
    "contents": {
      "Word reading order": { "img": "reading_order_words.png" },
      "Line grouping": { "img": "line_grouping.png" },
      "Block grouping": { "img": "block_grouping.png" }
    }
  },
  {
    "type": "text_file",
    "label": "<h3>Reading Order</h3>",
    "contents": "result.txt",
    "style": "{'width':'100%','height':'18em','white-space':'pre-wrap'}"
  },
  {
    "type": "text_file",
    "label": "<h3>Raw docTR JSON</h3>",
    "contents": "result_raw_doctr.json",
    "style": "{'width':'100%','height':'24em','white-space':'pre-wrap'}"
  },
  {
    "type": "text_file",
    "label": "<h3>Run summary</h3>",
    "contents": "summary.txt",
    "style": "{'width':'100%','height':'14em','white-space':'pre-wrap'}"
  },
  {
    "type": "file_download",
    "label": "<h3>Download structured outputs</h3>",
    "contents": {
      "JSON": "result.json",
      "Raw docTR JSON": "result_raw_doctr.json",
      "hOCR": "result.hocr",
      "Words CSV": "words.csv",
      "Structure debug JSON": "structure_debug.json",
      "Reading Order": "result.txt"
    }
  }
]
```

If `result.hocr` cannot be generated for a run, `hocr_error.txt` must explain
why. The file-download block can still include `result.hocr` if the runner
creates a small placeholder hOCR file with a clear comment, but the preferred
implementation is valid hOCR whenever straight-box export is available.

## Archive Plan

Archive files:

- `input_0.png`: Input
- `detector_input.png`: Detector input
- `detector_probability_map.png`: Detector response
- `detector_binary_map.png`: Binarized detector response
- `detector_components.png`: Detector components
- `detector_word_boxes.png`: Detector word boxes
- `recognizer_split_overlay.png`: Recognizer split overlay
- `recognizer_raw_crops_stack.png`: Raw sampled recognizer crops
- `recognizer_split_crops.png`: Recognizer wide-crop splitting
- `recognizer_input_crops_stack.png`: Sampled crops as sent to recognizer
- `recognizer_crops_contact_sheet.png`: Recognizer crop contact sheet
- `reading_order_words.png`: Word reading order
- `line_grouping.png`: Line grouping
- `block_grouping.png`: Block grouping
- `overlay_words.png`: Word boxes
- `overlay_lines.png`: Line boxes
- `overlay_blocks.png`: Block boxes
- `overlay_all.png`: All boxes
- `result.json`: Full JSON
- `result_raw_doctr.json`: Raw docTR JSON
- `structure_debug.json`: Detector/crop/line/block diagnostic JSON
- `words.csv`: Word table
- `result.txt`: Reading order
- `summary.txt`: Summary
- `result.hocr`: hOCR
- `hocr_error.txt`: hOCR error explanation, if generated
- `processed_input.png`: Processed input, only if wrapper preprocessing is later added

Archive params:

- all DDL parameters exposed in the final DDL.

Archive info:

```json
{
  "run_time": "run time"
}
```

Enable reconstruct:

```json
"enable_reconstruct": true
```

## Intermediate Diagnostics Implementation Notes

The runner should make docTR's pipeline understandable by exposing intermediate
states. The intended flow is:

1. Load `input_0.png` as RGB.
2. Instantiate `ocr_predictor` with the selected detector, recognizer, geometry,
   structure, and batch options.
3. Apply `det_input_size`; apply detector thresholds only when
   `det_threshold_mode=custom`, otherwise keep the selected detector's docTR
   defaults.
4. Capture detector preprocessor output and save `detector_input.png`.
5. Run the detector with `return_maps=True` to collect:
   - raw location predictions;
   - detector probability maps.
6. Save detector visual diagnostics:
   - `detector_probability_map.png`;
   - `detector_binary_map.png`;
   - `detector_components.png`;
   - `detector_word_boxes.png`.
7. Capture the crops that will be sent to the recognizer:
   - raw crops from page coordinates;
   - preprocessed recognizer crops after resize/padding/normalization, displayed
     after de-normalization.
8. Run full OCR and export final docTR results.
9. Save post-processing diagnostics:
   - `reading_order_words.png`;
   - `line_grouping.png`;
   - `block_grouping.png`;
   - `structure_debug.json`.

Implementation caution:

- A simple implementation may run the detector once for diagnostics and once
  inside `ocr_predictor`. That is acceptable for the first version if runtime is
  still reasonable and the summary records this cost.
- A later optimized implementation can subclass or wrap docTR internals to reuse
  the same detector pass for diagnostics and OCR.
- Diagnostic detector boxes must be generated with the same selected
  `assume_straight_pages`, `preserve_aspect_ratio`, `symmetric_pad`,
  `det_input_size`, detector threshold mode, and effective detector thresholds
  as the final OCR run.
- The sampled crop indices must be stored in `result.json` and
  `structure_debug.json` so that crop stack images can be traced back to word
  boxes and OCR predictions.

## Docker Model Preload Contract

Runtime should not download models. The Docker build should run
`preload_models.py` after installing `python-doctr` and after setting:

```dockerfile
ENV HOME /home/ipol
ENV DOCTR_CACHE_DIR /home/ipol/.cache/doctr
```

`preload_models.py` must instantiate:

- every detection architecture exposed in `det_arch`;
- every recognition architecture exposed in `reco_arch`;
- page orientation classifier if `detect_orientation`, `straighten_pages`, or
  `assume_straight_pages=false` can be selected;
- crop orientation classifier if `assume_straight_pages=false` can be selected.

The model lists in this document and the preload script must stay synchronized.

## Resolved Decisions

- Do not expose `page_max_side`; use `det_input_size` instead.
- Expose `det_input_size` in the first DDL.
- Do not expose runtime batch sizes in the first DDL; keep `det_bs=2` and
  `reco_bs=128` as internal docTR defaults.
- Add hOCR support in the first version.
- Use `max_pixels: "3000*3000"` for the first version.
- Omit `processed_input.png` when no wrapper-level preprocessing is applied.
- Always generate the detector, recognizer, reading-order, line-grouping, and
  block-grouping diagnostic images listed above.

## Open Decisions

- Exact first-version model list after Docker size/time tests.
