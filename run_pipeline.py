"""Full PaddleOCR-VL-1.5 pipeline on OmniDocBench v1.6, with the VLM running on OpenVINO (Intel iGPU).

Same pipeline as the Colab run (pipeline_version="v1.5", layout detection PP-DocLayoutV3,
per-block prompts, pretty=False markdown); only the VLM compute moves to ov_server.py.
Layout detection runs with Paddle on CPU. Resumable: pages with an existing .md are skipped.

  .venv-paddle\\Scripts\\python.exe run_pipeline.py --limit 20      (pilot)
  .venv-paddle\\Scripts\\python.exe run_pipeline.py                 (all 1651 pages)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
HF_REV = "d386947f7fc3bafdcd756c8485845a2f43a19875"   # OmniDocBench v1.6 (same as Colab)
DATA = ROOT / "omnidocbench_v1.6"
N_PAGES = 1651


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def ensure_dataset():
    imgs = DATA / "images"
    if (DATA / "OmniDocBench.json").exists() and imgs.is_dir() and len(os.listdir(imgs)) == N_PAGES:
        return
    print("[data] downloading OmniDocBench v1.6 (~1.4 GB, once)")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="opendatalab/OmniDocBench", repo_type="dataset", revision=HF_REV,
                      local_dir=str(DATA), allow_patterns=["OmniDocBench.json", "images/*"], max_workers=8)
    assert len(os.listdir(imgs)) == N_PAGES, "dataset incomplete"


def start_server(args, log_path):
    cmd = [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "ov_server.py"), "--device", args.device,
           "--precision", args.precision, "--llm", args.llm, "--port", str(args.port)]
    srv = subprocess.Popen(cmd, cwd=ROOT, stdout=open(log_path, "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{args.port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < 900:
        if srv.poll() is not None:
            sys.exit(f"[server] exited early — see {log_path}\n" + Path(log_path).read_text(encoding="utf-8")[-2000:])
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"[server] ready ({time.time() - t0:.0f}s)")
            return srv
        except Exception:
            time.sleep(3)
    srv.terminate()
    sys.exit("[server] not ready after 15 min")


def force_png_blocks():
    """Server backends send block crops as JPEG (lossy); the Colab native run uses raw pixels -> send PNG."""
    import paddlex.inference.models.doc_vlm.predictor as P
    orig = P.DocVLMGenAIClientPredictor._doc_vlm_genai_build_request_specs

    def patched(self, client, data, image_format, *a, **k):
        return orig(self, client, data, "PNG", *a, **k)
    P.DocVLMGenAIClientPredictor._doc_vlm_genai_build_request_specs = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--precision", choices=["f16", "f32"], default="f16")
    ap.add_argument("--llm", choices=["fp", "int8"], default="fp")
    ap.add_argument("--limit", type=int, default=0, help="only first N pages (pilot)")
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--pages", default="", help="text file with image names (one per line) to run only those pages")
    ap.add_argument("--tag", default="", help="suffix for the output folder name")
    args = ap.parse_args()

    ensure_dataset()
    run = f"paddleocr_vl15_ov_{args.device.lower()}_{args.precision}_{args.llm}" + (f"_{args.tag}" if args.tag else "")
    out = ROOT / "pipeline_results" / run
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in (DATA / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        images = images[: args.limit]
    if args.pages:
        keep = {l.strip() for l in open(args.pages, encoding="utf-8") if l.strip()}
        images = [p for p in images if p.name in keep]
    todo = [p for p in images if not (out / f"{p.stem}.md").exists()]
    print(f"[run] {run}: {len(images)} pages | done {len(images) - len(todo)} | todo {len(todo)}")
    if not todo:
        return

    srv = start_server(args, out.parent / f"server_{run}.log")
    try:
        force_png_blocks()
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
        from paddleocr import PaddleOCRVL
        pipe = PaddleOCRVL(pipeline_version="v1.5", use_doc_orientation_classify=False, use_doc_unwarping=False,
                           vl_rec_backend="vllm-server", vl_rec_server_url=f"http://127.0.0.1:{args.port}/v1",
                           vl_rec_max_concurrency=1, device="cpu")
        log = open(out.parent / f"{run}_log.jsonl", "a", encoding="utf-8")
        t_start, done_now = time.time(), 0
        for k, img in enumerate(todo, 1):
            t0 = time.time()
            try:
                for res in pipe.predict(str(img)):
                    text = res._to_markdown(pretty=False)["markdown_texts"]
                tmp = out / f"{img.stem}.md.tmp"
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, out / f"{img.stem}.md")
                status = "ok"
                done_now += 1
            except Exception as e:
                status = f"failed: {e!r}"
            sec = time.time() - t0
            log.write(json.dumps({"image": img.name, "sec": round(sec, 2), "status": status}) + "\n")
            log.flush()
            avg = (time.time() - t_start) / k
            total_done = len(images) - len(todo) + done_now
            print(f"[{total_done}/{len(images)}] {sec:6.1f}s {status[:40]:40s} | avg {avg:.1f}s/page "
                  f"| ETA {fmt(avg * (len(todo) - k))} | {img.name[:50]}", flush=True)
    finally:
        srv.terminate()
    print(f"[run] finished — outputs in {out}")


if __name__ == "__main__":
    main()
