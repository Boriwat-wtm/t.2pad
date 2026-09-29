"""Score the iGPU pipeline outputs with OmniDocBench v1.6 (Edit_dist + TEDS, no CDM) — same config as Colab.

  .venv-eval\\Scripts\\python.exe eval_pipeline.py [run_name]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
ODB = ROOT / "OmniDocBench"
PUB = {"text_edit": 0.038, "table_teds": 91.67, "table_teds_s": 94.37, "read_order_edit": 0.130}
PUB_NO_CDM = ((1 - PUB["text_edit"]) * 100 + PUB["table_teds"]) / 2      # 93.94


def main():
    runs = sorted(p.name for p in (ROOT / "pipeline_results").iterdir() if p.is_dir())
    run = sys.argv[1] if len(sys.argv) > 1 else runs[0]
    pred = ROOT / "pipeline_results" / run
    n = sum(1 for f in pred.iterdir() if f.suffix == ".md")
    print(f"[eval] {run}: {n}/1651 predictions")
    if n < 1651:
        print("  WARNING: missing pages are scored as EMPTY pages -> score will be lower")

    cfg = ROOT / f"cfg_{run}.yaml"
    cfg.write_text(f"""end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist]
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: 8
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {(ROOT / 'omnidocbench_v1.6' / 'OmniDocBench.json').as_posix()}
    prediction:
      data_path: {pred.as_posix()}
    match_method: quick_match
    match_workers: 8
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
""", encoding="utf-8")
    (ODB / "result").mkdir(exist_ok=True)
    rc = subprocess.run([sys.executable, "pdf_validation.py", "--config", str(cfg)], cwd=ODB).returncode
    if rc != 0:
        sys.exit(f"[eval] failed (exit {rc})")

    rep = json.load(open(ODB / "result" / f"{run}_quick_match_run_summary.json", encoding="utf-8"))
    m = rep["notebook_metric_summary"]["metrics"]
    te = m["text_block_Edit_dist"]["notebook_value"]
    teds = m["table_TEDS"]["notebook_value"]
    ours = ((1 - te) * 100 + teds) / 2
    rows = [("Overall (no CDM)", ours, PUB_NO_CDM), ("Text Edit (lower=better)", te, PUB["text_edit"]),
            ("Table TEDS", teds, PUB["table_teds"]),
            ("Table TEDS-S", m["table_TEDS_structure_only"]["notebook_value"], PUB["table_teds_s"]),
            ("Read Order Edit (lower=better)", m["reading_order_Edit_dist"]["notebook_value"], PUB["read_order_edit"])]
    print(f"\n===== SCORE {run} =====")
    print(f"{'metric':32s} {'ours':>9s} {'published':>9s} {'diff':>8s}")
    for name, o, p in rows:
        print(f"{name:32s} {o:9.4f} {p:9.4f} {o - p:+8.4f}")
    out = ROOT / "pipeline_results" / f"score_{run}.json"
    out.write_text(json.dumps({"run": run, "pages": n, "rows": rows}, indent=1), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
