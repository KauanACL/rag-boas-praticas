"""Exporta o modelo fine-tunado para ONNX (fp32 + int8) no formato do transformers.js.

Saída: models/onnx/{config.json, tokenizer*.json, onnx/model.onnx, onnx/model_quantized.onnx}
"""
import shutil
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
ORIGEM = RAIZ / "models" / "e5-small-boas-praticas"
SAIDA = RAIZ / "models" / "onnx"


class SoEstadoOculto(torch.nn.Module):
    """Recebe os tensores por nome e devolve só last_hidden_state (o pooling fica no JS)."""

    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo

    def forward(self, input_ids, attention_mask):
        return self.modelo(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def main():
    (SAIDA / "onnx").mkdir(parents=True, exist_ok=True)
    modelo = SoEstadoOculto(AutoModel.from_pretrained(ORIGEM).eval())
    tokenizer = AutoTokenizer.from_pretrained(ORIGEM)
    entrada = tokenizer(["query: teste"], return_tensors="pt")
    nomes = ["input_ids", "attention_mask"]  # o tokenizer deste modelo não gera token_type_ids
    eixos = {n: {0: "batch", 1: "seq"} for n in nomes}
    eixos["last_hidden_state"] = {0: "batch", 1: "seq"}

    fp32 = SAIDA / "onnx" / "model.onnx"
    torch.onnx.export(
        modelo,
        tuple(entrada[n] for n in nomes),
        str(fp32),
        input_names=nomes,
        output_names=["last_hidden_state"],
        dynamic_axes=eixos,
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32), str(SAIDA / "onnx" / "model_quantized.onnx"), weight_type=QuantType.QInt8)

    for arquivo in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        shutil.copy(ORIGEM / arquivo, SAIDA / arquivo)
    for p in sorted(SAIDA.rglob("*")):
        if p.is_file():
            print(f"{p.relative_to(SAIDA)}  {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
