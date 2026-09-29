"""Step 1: download OpenVINO helper code + PaddleOCR-VL-1.5 weights, convert to OpenVINO IR.

Follows openvinotoolkit/openvino_notebooks notebooks/paddleocr_vl (pinned commit below).
Run once; re-running skips finished steps.
"""
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
NB_COMMIT = "5f55c05ee771ff152a39d86196eff40461f935dd"   # openvino_notebooks, notebooks/paddleocr_vl
NB_URL = f"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/{NB_COMMIT}/notebooks/paddleocr_vl"
HELPER_FILES = ["ov_paddleocr_vl.py", "image_processing_paddleocr_vl.py", "modeling_paddleocr_vl.py"]
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"

HELPER_DIR = ROOT / "ov_helper"
PRETRAINED_DIR = ROOT / "models" / "PaddleOCR-VL-1.5"
OV_DIR = ROOT / "models" / "ov_paddleocr_vl_1_5"          # INT8 LLM (speed test)
OV_DIR_FP = ROOT / "models" / "ov_paddleocr_vl_1_5_fp"    # uncompressed LLM (full pipeline)


def main():
    HELPER_DIR.mkdir(parents=True, exist_ok=True)
    for f in HELPER_FILES:
        dst = HELPER_DIR / f
        if not dst.exists():
            print(f"[1/3] download {f}")
            urllib.request.urlretrieve(f"{NB_URL}/{f}", dst)

    if not (PRETRAINED_DIR / "config.json").exists():
        print(f"[2/3] download {MODEL_ID} (~2 GB)")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, local_dir=str(PRETRAINED_DIR))
    # the notebook replaces the model repo's modeling file with its patched one
    target, backup = PRETRAINED_DIR / "modeling_paddleocr_vl.py", PRETRAINED_DIR / "modeling_paddleocr_vl.py.bak"
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(HELPER_DIR / "modeling_paddleocr_vl.py", target)

    sys.path.insert(0, str(HELPER_DIR))
    # the converter writes llm_stateful.xml only when no compression is requested -> two separate exports
    for out, int8, marker in [(OV_DIR, True, "llm_stateful_int8.xml"), (OV_DIR_FP, False, "llm_stateful.xml")]:
        if (out / marker).exists():
            print(f"[3/3] {out.name} already converted — skip")
            continue
        print(f"[3/3] convert to OpenVINO -> {out.name} ({'INT8' if int8 else 'uncompressed'} LLM, several minutes)")
        from ov_paddleocr_vl import PaddleOCR_VL_OV
        conv = PaddleOCR_VL_OV(pretrained_model_path=str(PRETRAINED_DIR), ov_model_path=str(out),
                               device="CPU", llm_int4_compress=False, llm_int8_compress=int8,
                               vision_int8_quant=False)
        conv.export_paddleocr_vl_to_ov()
        conv.close()
    print("DONE — next: run_test.bat")


if __name__ == "__main__":
    main()
