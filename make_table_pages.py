"""Write N OmniDocBench v1.6 pages that contain a table, evenly spaced over all table pages
(so every document type is represented), to pages_tables<N>.txt."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
gt = json.load(open(ROOT / "omnidocbench_v1.6" / "OmniDocBench.json", encoding="utf-8"))
names = sorted(Path(s["page_info"]["image_path"]).name for s in gt
               if any(d.get("category_type") == "table" for d in s.get("layout_dets", [])))
out = ROOT / f"pages_tables{n}.txt"
# evenly spaced over all table pages so every document type is represented
step = max(len(names) / n, 1)
pick = [names[int(i * step)] for i in range(min(n, len(names)))]
out.write_text("\n".join(pick) + "\n", encoding="utf-8")
print(f"{len(names)} pages have tables; wrote {len(pick)} evenly spaced to {out.name}")
