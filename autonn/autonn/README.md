# AutoNN/tango Wheel (tango package)

## Build wheel (from this directory, in a build venv)
```
python -m venv .build-venv
source .build-venv/bin/activate
python -m pip install --upgrade pip build
python -m build
```
Wheel: `dist/tango-<version>-py3-none-any.whl`

## Install into a fresh venv (separate from build venv)
```
python -m venv .install-venv
source .install-venv/bin/activate
pip install --upgrade pip
pip install dist/tango-*-py3-none-any.whl
```
Dependencies are not bundled in the wheel. Ensure you have `torch`, `torchvision` installed. If CUDA wheels are needed, install the matching torch/torchvision first.

## Use with detection_app.py and a torch.save() model
```
python detection_app.py \
  --weights bestmodel.pt \
  --source car.mp4 \
  --imgsz 640 \
  --save-img --save-dir results
```

Notes:
- The wheel exposes `tango` as the top-level package, so pickled modules like `tango.common.models...` resolve when loading `bestmodel.pt`.
- TorchScript (`.torchscript`/`.ts`) weights are also supported by `detection_app.py`.

## Sharing with others
- You only need to share the wheel file (`dist/tango-*-py3-none-any.whl`), not the entire `dist` directory.
- Place the wheel, `detection_app.py`, your `bestmodel.pt`, and the input file (e.g., `car.mp4`) in one directory, then instruct:
  1) Create/activate a venv.
  2) `pip install <wheel-file>`
  3) Run `python detection_app.py --weights bestmodel.pt --source car.mp4 --imgsz 640 --save-img --save-dir results`
- No extra `import tango` or `sys.path` tweaks are needed once the wheel is installed.
