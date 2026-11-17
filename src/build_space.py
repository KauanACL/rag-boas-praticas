"""Gera space/chunks.json: chunks + embeddings pré-calculados para o Space estático.

Os vetores saem do mesmo modelo ONNX int8 que o navegador usa para a pergunta,
então a similaridade no Space é a mesma medida em evaluate.py.
"""
import json
from pathlib import Path

from onnx_encoder import CodificadorOnnx
from retriever import PREFIXO_TRECHO, carregar_chunks

RAIZ = Path(__file__).resolve().parent.parent
MODELO_ONNX = RAIZ / "models" / "onnx" / "onnx" / "model_quantized.onnx"
SAIDA = RAIZ / "space" / "chunks.json"


def main():
    chunks = carregar_chunks()
    vetores = CodificadorOnnx(str(MODELO_ONNX)).encode([PREFIXO_TRECHO + c["texto"] for c in chunks])
    registros = [
        {**c, "vetor": [round(float(x), 5) for x in v]}
        for c, v in zip(chunks, vetores)
    ]
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(json.dumps(registros, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(registros)} chunks, dim {len(vetores[0])} -> {SAIDA.relative_to(RAIZ)} "
          f"({SAIDA.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
