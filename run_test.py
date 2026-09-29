"""Step 2: speed test of PaddleOCR-VL-1.5 (OpenVINO) on OmniDocBench v1.6 pages.

Uses the same first-N pages (sorted by name) as Colab batch 1, so timings can be compared.
NOTE: this is the OpenVINO notebook's whole-page "OCR:" mode (no layout detection),
not the PaddleOCR pipeline used for the OmniDocBench score — use it for speed only.

  python run_test.py --device GPU --n 20
  python run_test.py --device CPU --n 20
  python run_test.py --device GPU --llm fp     # no INT8 weight compression
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "ov_helper"))
OV_DIR = ROOT / "models" / "ov_paddleocr_vl_1_5"
IMG_DIR = ROOT / "omnidocbench_images"
HF_REV = "d386947f7fc3bafdcd756c8485845a2f43a19875"   # OmniDocBench v1.6 (same as Colab)


def get_images(n):
    from huggingface_hub import hf_hub_download, list_repo_files
    names = sorted(f for f in list_repo_files("opendatalab/OmniDocBench", repo_type="dataset", revision=HF_REV)
                   if f.startswith("images/"))[:n]
    paths = []
    for f in names:
        p = IMG_DIR / Path(f).name
        if not p.exists():
            hf_hub_download("opendatalab/OmniDocBench", f, repo_type="dataset", revision=HF_REV, local_dir=str(ROOT / "_hf"))
            IMG_DIR.mkdir(exist_ok=True)
            (ROOT / "_hf" / f).replace(p)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="GPU", help="GPU (Intel Arc iGPU) | CPU | AUTO")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--llm", choices=["int8", "fp"], default="int8", help="int8 = notebook default")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    import openvino as ov
    from PIL import Image
    from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM

    core = ov.Core()
    print("OpenVINO", ov.get_version(), "| devices:", core.available_devices)
    if args.device not in core.available_devices and args.device != "AUTO":
        sys.exit(f"device {args.device} not available on this machine")
    if args.device != "AUTO":
        print("using:", core.get_property(args.device, "FULL_DEVICE_NAME"))

    images = get_images(args.n)
    t_load = time.perf_counter()
    model = OVPaddleOCRVLForCausalLM(core=core, ov_model_path=str(OV_DIR), device=args.device,
                                     llm_int4_compress=False, llm_int8_compress=(args.llm == "int8"),
                                     vision_int8_quant=False,
                                     # llm_int8_quant adds DYNAMIC_QUANTIZATION_GROUP_SIZE, which the NPU plugin rejects
                                     llm_int8_quant=(args.llm == "int8" and args.device != "NPU"),
                                     llm_infer_list=[], vision_infer=[])
    print(f"model load/compile: {time.perf_counter() - t_load:.1f}s")
    gen = {"bos_token_id": model.tokenizer.bos_token_id, "eos_token_id": model.tokenizer.eos_token_id,
           "pad_token_id": model.tokenizer.pad_token_id, "max_new_tokens": args.max_new_tokens, "do_sample": False}

    tag = f"{args.device.lower()}_{args.llm}"
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(images, 1):
        msg = [{"role": "user", "content": [{"type": "image", "image": Image.open(p).convert("RGB")},
                                            {"type": "text", "text": "OCR:"}]}]
        t0 = time.perf_counter()
        try:
            text, _ = model.chat(messages=msg, generation_config=gen)
            status = "ok"
        except Exception as e:
            text, status = "", f"failed: {e!r}"
        sec = time.perf_counter() - t0
        (out / f"{p.stem}.md").write_text(text or "", encoding="utf-8")
        rows.append({"page": p.name, "sec": round(sec, 2), "chars": len(text or ""), "status": status})
        print(f"[{i}/{len(images)}] {sec:6.1f}s  {len(text or ''):5d} chars  {status}  {p.name[:60]}")

    with open(out / "timing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page", "sec", "chars", "status"])
        w.writeheader(); w.writerows(rows)
    ok = [r["sec"] for r in rows if r["status"] == "ok"]
    rest = ok[1:] or ok                     # first page includes warm-up
    print("\n===== SUMMARY =====")
    print(f"device={args.device} llm={args.llm} pages={len(rows)} ok={len(ok)}")
    if rest:
        print(f"avg {sum(rest) / len(rest):.1f} s/page (excluding first page)")
    print(f"outputs + timing.csv: {out}")


if __name__ == "__main__":
    main()
