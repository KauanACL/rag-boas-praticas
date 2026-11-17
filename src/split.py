"""Separa os chunks em treino e avaliação, antes de gerar qualquer pergunta.

Os chunks vizinhos compartilham 40 palavras. Para reduzir o vazamento, a divisão
é feita em blocos de 3 chunks consecutivos (1 bloco de avaliação a cada 3), e
só as bordas dos blocos de avaliação se sobrepõem a chunks de treino.

Saída: data/processed/split.json
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CHUNKS = RAIZ / "data" / "processed" / "chunks.jsonl"
SAIDA = RAIZ / "data" / "processed" / "split.json"

BLOCO = 3


def main():
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    treino, avaliacao = [], []
    for c in chunks:
        posicao = int(c["id"].rsplit("-", 1)[1])
        (avaliacao if (posicao // BLOCO) % 3 == 1 else treino).append(c["id"])
    SAIDA.write_text(json.dumps({"treino": treino, "avaliacao": avaliacao}, indent=1))
    print(f"treino: {len(treino)} chunks | avaliação: {len(avaliacao)} chunks")


if __name__ == "__main__":
    main()
