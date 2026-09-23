# Local dataset integration

The supplied Kaggle extraction is intentionally kept outside Git because it is
approximately 1.2 GB and contains 22,120 JPEG images. Configure
`config/config.yaml` or pass `--root` to the audit with the local directory:

```text
C:\Users\venne\Downloads\archive (7)\images\images\train
```

The current extraction contains no YOLO annotation files. The audit therefore
reports zero annotations and training refuses to start until a labeled,
six-class dataset passes mapping validation. Do not infer or fabricate labels
from the images.

Run:

```powershell
python training/dataset_audit.py --root "C:\Users\venne\Downloads\archive (7)\images\images\train"
```

The compact tracked facts are recorded in `data/dataset_manifest.json`.
