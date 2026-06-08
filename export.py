#!/usr/bin/env python3
"""
Exporta um modelo wav2vec2 CTC (Wav2Vec2ForCTC) para ONNX int8, para uso no
forced alignment local do Clipaisso Desktop.

Modelo via env `MODEL_ID`. Pensado para a família permissiva (Apache 2.0)
`jonatasgrosman/wav2vec2-large-xlsr-53-<idioma>` — um modelo por idioma.

  - Entrada:  waveform mono 16 kHz, float32, shape [1, N]
  - Saída:    logits CTC por frame, shape [1, T, V]  (T ≈ N/320 → ~20 ms/frame)

Gera em out/: model.int8.onnx + vocab.json + tokens.txt + config.json.
O fp32 é só intermediário (necessário para quantizar) e é apagado no fim —
distribuímos apenas o int8. Carregamos o vocab pelo TOKENIZER (não pelo
AutoProcessor) de propósito: esses repos trazem um KenLM, e o AutoProcessor
tentaria puxar pyctcdecode/kenlm e quebrar no CI.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer

MODEL_ID = os.environ["MODEL_ID"]
OUT = Path("out")
OUT.mkdir(exist_ok=True)

print(f"[export] baixando {MODEL_ID} ...", flush=True)
model = AutoModelForCTC.from_pretrained(MODEL_ID)
model.eval()

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID)
vocab = tokenizer.get_vocab()  # token -> id
(OUT / "vocab.json").write_text(
    json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
)
id_to_tok = {i: t for t, i in vocab.items()}
lines = [id_to_tok.get(i, "") for i in range(len(id_to_tok))]
(OUT / "tokens.txt").write_text("\n".join(lines), encoding="utf-8")
model.config.to_json_file(str(OUT / "config.json"))
print(f"[export] vocab: {len(vocab)} tokens", flush=True)

# ---- export ONNX fp32 (intermediário) -------------------------------------
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
print(f"[export] ok ({fp32_path.stat().st_size/1e6:.0f} MB)", flush=True)

# ---- validação: argmax do ONNX == do PyTorch ------------------------------
import onnxruntime as ort  # noqa: E402

test = torch.randn(1, 16000 * 3, dtype=torch.float32)
with torch.no_grad():
    ref = model(test).logits.numpy()
sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
got = sess.run(["logits"], {"input_values": test.numpy()})[0]
if ref.shape != got.shape:
    print(f"[export] ERRO shapes torch={ref.shape} onnx={got.shape}")
    sys.exit(1)
if not np.array_equal(ref.argmax(-1), got.argmax(-1)):
    print("[export] ERRO: argmax por frame divergiu torch vs onnx")
    sys.exit(1)
print(f"[export] validação OK (argmax idêntico, shape={got.shape})", flush=True)

# ---- quantização int8 (só MatMul; Conv viraria ConvInteger não suportado) --
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

int8_path = OUT / "model.int8.onnx"
print("[export] quantizando int8 (apenas MatMul) ...", flush=True)
quantize_dynamic(
    str(fp32_path),
    str(int8_path),
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"],
)
print(f"[export] ok ({int8_path.stat().st_size/1e6:.0f} MB)", flush=True)

got8 = (
    ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    .run(["logits"], {"input_values": test.numpy()})[0]
)
agree = float(np.mean(ref.argmax(-1) == got8.argmax(-1)))
print(f"[export] int8 argmax agreement = {agree:.3%}", flush=True)

fp32_path.unlink()  # só distribuímos o int8
print("[export] concluído.", flush=True)
