"""Mede recall@1, recall@5 e MRR@10 do buscador no conjunto de avaliação.

Acerto = o chunk exato anotado na pergunta (critério estrito: um vizinho que
também contenha a resposta conta como erro).

Uso:
    python src/evaluate.py                       # modelo base
    python src/evaluate.py models/e5-small-bp    # outro modelo (pasta ou id no Hub)
"""
import json
import sys
from pathlib import Path

from retriever import MODELO_BASE, Buscador

RAIZ = Path(__file__).resolve().parent.parent
PERGUNTAS = RAIZ / "eval" / "perguntas.jsonl"
RESULTADOS = RAIZ / "eval" / "resultados.json"
K_MAX = 10


def avaliar(modelo: str) -> dict:
    perguntas = [json.loads(l) for l in PERGUNTAS.open(encoding="utf-8")]
    buscador = Buscador(modelo)
    respostas = buscador.buscar_varias([p["pergunta"] for p in perguntas], k=K_MAX)

    acertos1 = acertos5 = soma_rr = 0.0
    erros = []
    for p, resultado in zip(perguntas, respostas):
        ids = [r["id"] for r in resultado]
        posicao = ids.index(p["chunk_id"]) + 1 if p["chunk_id"] in ids else None
        acertos1 += posicao == 1
        acertos5 += posicao is not None and posicao <= 5
        soma_rr += 1 / posicao if posicao else 0
        if posicao != 1:
            erros.append({"id": p["id"], "esperado": p["chunk_id"], "top1": ids[0], "posicao": posicao})

    n = len(perguntas)
    return {
        "modelo": modelo,
        "n_perguntas": n,
        "recall@1": round(acertos1 / n, 4),
        "recall@5": round(acertos5 / n, 4),
        "mrr@10": round(soma_rr / n, 4),
        "erros_top1": erros,
    }


def main():
    modelo = sys.argv[1] if len(sys.argv) > 1 else MODELO_BASE
    r = avaliar(modelo)
    print(f"{r['modelo']}  n={r['n_perguntas']}  "
          f"recall@1={r['recall@1']:.3f}  recall@5={r['recall@5']:.3f}  mrr@10={r['mrr@10']:.3f}")

    todos = json.loads(RESULTADOS.read_text()) if RESULTADOS.exists() else {}
    todos[modelo] = r
    RESULTADOS.write_text(json.dumps(todos, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
