"""Busca semântica: embeddings (sentence-transformers) + índice FAISS."""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

RAIZ = Path(__file__).resolve().parent.parent
CHUNKS = RAIZ / "data" / "processed" / "chunks.jsonl"
MODELO_BASE = "intfloat/multilingual-e5-small"

# A família e5 foi treinada com estes prefixos; sem eles a qualidade cai.
PREFIXO_PERGUNTA = "query: "
PREFIXO_TRECHO = "passage: "


def carregar_chunks(caminho: Path = CHUNKS) -> list[dict]:
    return [json.loads(l) for l in caminho.open(encoding="utf-8")]


class Buscador:
    def __init__(self, modelo: str = MODELO_BASE, chunks: list[dict] | None = None):
        self.modelo = SentenceTransformer(modelo)
        self.chunks = chunks if chunks is not None else carregar_chunks()
        vetores = self._codificar([PREFIXO_TRECHO + c["texto"] for c in self.chunks])
        # Vetores normalizados: produto interno = similaridade de cosseno.
        self.indice = faiss.IndexFlatIP(vetores.shape[1])
        self.indice.add(vetores)

    def _codificar(self, textos: list[str]) -> np.ndarray:
        vetores = self.modelo.encode(textos, normalize_embeddings=True, batch_size=32)
        return np.asarray(vetores, dtype="float32")

    def buscar(self, pergunta: str, k: int = 5) -> list[dict]:
        return self.buscar_varias([pergunta], k)[0]

    def buscar_varias(self, perguntas: list[str], k: int = 5) -> list[list[dict]]:
        vetores = self._codificar([PREFIXO_PERGUNTA + p for p in perguntas])
        scores, posicoes = self.indice.search(vetores, k)
        return [
            [{**self.chunks[i], "score": float(s)} for s, i in zip(linha_s, linha_i)]
            for linha_s, linha_i in zip(scores, posicoes)
        ]


if __name__ == "__main__":
    import sys

    pergunta = " ".join(sys.argv[1:]) or "Com que frequência devo limpar a caixa d'água?"
    for r in Buscador().buscar(pergunta, k=3):
        print(f"{r['score']:.3f}  {r['id']}  ({r['local']})  {r['texto'][:120]}...")
