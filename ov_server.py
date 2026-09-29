"""OpenAI-compatible server that runs PaddleOCR-VL-1.5 with OpenVINO.

PaddleOCR's own pipeline (layout detection + per-block prompts) sends each cropped
block here via its "vllm-server" backend, so everything except the VLM compute is
identical to the Colab (Paddle GPU) run. Started automatically by run_pipeline.py.

  POST /v1/chat/completions   (image as data URL + prompt text; mm_processor_kwargs min/max_pixels)
  GET  /v1/models
"""
import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "ov_helper"))
OV_DIRS = {"int8": ROOT / "models" / "ov_paddleocr_vl_1_5", "fp": ROOT / "models" / "ov_paddleocr_vl_1_5_fp"}
MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--precision", choices=["f16", "f32"], default="f16", help="GPU inference precision")
    ap.add_argument("--llm", choices=["fp", "int8"], default="fp")
    ap.add_argument("--port", type=int, default=8111)
    args = ap.parse_args()

    import openvino as ov
    from PIL import Image
    from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM

    core = ov.Core()
    if args.device == "GPU":
        core.set_property("GPU", {"INFERENCE_PRECISION_HINT": args.precision})
    model = OVPaddleOCRVLForCausalLM(core=core, ov_model_path=str(OV_DIRS[args.llm]), device=args.device,
                                     llm_int4_compress=False, llm_int8_compress=(args.llm == "int8"),
                                     vision_int8_quant=False,
                                     llm_int8_quant=(args.llm == "int8" and args.device != "NPU"),
                                     llm_infer_list=[], vision_infer=[])
    tok = model.tokenizer
    lock = threading.Lock()   # one generation at a time

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._send(404, {"error": {"message": "not found"}})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                image, text = None, ""
                for c in req["messages"][-1]["content"]:
                    if c["type"] == "image_url":
                        b64 = c["image_url"]["url"].split(",", 1)[1]
                        image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                    elif c["type"] == "text":
                        text += c["text"]
                mm = req.get("mm_processor_kwargs") or {}
                ipc = {k: int(mm[k]) for k in ("min_pixels", "max_pixels") if mm.get(k) is not None}
                max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 8192)
                gen = {"bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
                       "pad_token_id": tok.pad_token_id, "max_new_tokens": max_new, "do_sample": False}
                msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
                with lock:
                    out, _ = model.chat(messages=msgs, generation_config=gen, image_processor_config=ipc or None)
                self._send(200, {"id": "chatcmpl-ov", "object": "chat.completion", "created": int(time.time()),
                                 "model": MODEL_NAME,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": out},
                                              "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
            except Exception as e:
                self._send(500, {"error": {"message": repr(e)}})

        def log_message(self, *a):
            pass

    print(f"[ov_server] ready on 127.0.0.1:{args.port} device={args.device} precision={args.precision} llm={args.llm}",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
