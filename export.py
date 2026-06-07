#!/usr/bin/env python3
"""
Exporta o modelo multilíngue de forced alignment (MMS) para ONNX.

Modelo: MahmoudAshraf/mms-300m-1130-forced-aligner
  - Wav2Vec2ForCTC baseado no facebook/mms-300m
  - Vocabulário romanizado (uroman) -> funciona para 1000+ idiomas
  - Entrada:  waveform mono 16 kHz, float32, shape [1, N]
  - Saída:    logits CTC por frame, shape [1, T, V]
              (T = N/320 frames, ou seja ~50 fps / 20 ms por frame; V = tamanho do vocab)

Gera dois artefatos:
  - model.onnx       (fp32, ~1.2 GB)  -> qualidade máxima
  - model.int8.onnx  (int8, ~320 MB)  -> download padrão do app

Mais vocab.json + config.json + tokens.txt (id -> token, ordenado), que o
app usa para mapear o texto transcrito na sequência-alvo do alinhamento.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

MODEL_ID = os.environ.get("MODEL_ID", "MahmoudAshraf/mms-300m-1130-forced-aligner")
OUT = Path("out")
OUT.mkdir(exist_ok=True)

print(f"[export] baixando {MODEL_ID} ...", flush=True)
model = AutoModelForCTC.from_pretrained(MODEL_ID)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# ---- vocabulário / config -------------------------------------------------
vocab = processor.tokenizer.get_vocab()  # token -> id
(OUT / "vocab.json").write_text(
    json.dumps(vocab, ensure_ascii=False, indent=0), encoding="utf-8"
)

# tokens.txt: uma linha por id (0..V-1), facilita o parsing no Rust
id_to_tok = {i: t for t, i in vocab.items()}
lines = [id_to_tok.get(i, "") for i in range(len(id_to_tok))]
(OUT / "tokens.txt").write_text("\n".join(lines), encoding="utf-8")

model.config.to_json_file(str(OUT / "config.json"))
print(f"[export] vocab: {len(vocab)} tokens", flush=True)

# ---- export ONNX (fp32) ---------------------------------------------------
# Áudio dummy de 2 s (32000 amostras) só para traçar o grafo.
dummy = torch.zeros(1, 32000, dtype=torch.float32)

fp32_path = OUT / "model.onnx"
print("[export] exportando ONNX fp32 ...", flush=True)
torch.onnx.export(
    model,
    (dummy,),
    str(fp32_path),
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_values": {0: "batch", 1: "samples"},
        "logits": {0: "batch", 1: "frames"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"[export] ok -> {fp32_path} ({fp32_path.stat().st_size/1e6:.0f} MB)", flush=True)

# ---- validação numérica: torch vs onnxruntime -----------------------------
import onnxruntime as ort  # noqa: E402

print("[export] validando ONNX contra PyTorch ...", flush=True)
test = torch.randn(1, 16000 * 3, dtype=torch.float32)  # 3 s aleatórios
with torch.no_grad():
    ref = model(test).logits.numpy()

sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
got = sess.run(["logits"], {"input_values": test.numpy()})[0]

if ref.shape != got.shape:
    print(f"[export] ERRO: shapes diferentes torch={ref.shape} onnx={got.shape}")
    sys.exit(1)

max_diff = float(np.max(np.abs(ref - got)))
print(f"[export] max|torch-onnx| = {max_diff:.5f}  shape={got.shape}", flush=True)
if max_diff > 1e-2:
    print("[export] ERRO: divergência numérica acima da tolerância")
    sys.exit(1)

# argmax (o que o forced alignment realmente usa) precisa bater
if not np.array_equal(ref.argmax(-1), got.argmax(-1)):
    print("[export] ERRO: argmax por frame divergiu entre torch e onnx")
    sys.exit(1)
print("[export] validação OK (argmax idêntico)", flush=True)

# ---- quantização dinâmica int8 -------------------------------------------
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

int8_path = OUT / "model.int8.onnx"
print("[export] quantizando int8 ...", flush=True)
quantize_dynamic(
    str(fp32_path),
    str(int8_path),
    weight_type=QuantType.QInt8,
)
print(f"[export] ok -> {int8_path} ({int8_path.stat().st_size/1e6:.0f} MB)", flush=True)

# sanidade da int8: argmax deve continuar batendo na maioria dos frames
sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
got8 = sess8.run(["logits"], {"input_values": test.numpy()})[0]
agree = float(np.mean(ref.argmax(-1) == got8.argmax(-1)))
print(f"[export] int8 argmax agreement = {agree:.3%}", flush=True)
if agree < 0.95:
    print("[export] AVISO: concordância int8 baixa; revisar qualidade")

print("[export] concluído.", flush=True)
