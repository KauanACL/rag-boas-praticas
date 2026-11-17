"""Codificador ONNX com a mesma interface de encode() do SentenceTransformer.

Reproduz o que o transformers.js faz no navegador: tokenização, mean pooling
e normalização. Serve para medir o modelo quantizado que roda no Space estático.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class CodificadorOnnx:
    def __init__(self, arquivo_onnx: str):
        pasta = Path(arquivo_onnx).resolve().parent.parent
        self.tokenizer = AutoTokenizer.from_pretrained(pasta)
        self.sessao = ort.InferenceSession(arquivo_onnx, providers=["CPUExecutionProvider"])

    def encode(self, textos, normalize_embeddings=True, batch_size=32, **_):
        saida = []
        for i in range(0, len(textos), batch_size):
            tok = self.tokenizer(textos[i:i + batch_size], padding=True, truncation=True,
                                 max_length=512, return_tensors="np")
            estados = self.sessao.run(None, {"input_ids": tok["input_ids"],
                                             "attention_mask": tok["attention_mask"]})[0]
            mascara = tok["attention_mask"][..., None].astype(np.float32)
            vetores = (estados * mascara).sum(1) / mascara.sum(1)
            if normalize_embeddings:
                vetores /= np.linalg.norm(vetores, axis=1, keepdims=True)
            saida.append(vetores)
        return np.concatenate(saida)
