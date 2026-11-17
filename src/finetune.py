"""Fine-tuning do e5-small com pares (pergunta, trecho) dos chunks de TREINO.

Os chunks e as perguntas de avaliação (eval/perguntas.jsonl) não entram aqui.

Uso:
    python src/finetune.py [epocas]
Saída: models/e5-small-boas-praticas/
"""
import json
import sys
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from retriever import MODELO_BASE, PREFIXO_PERGUNTA, PREFIXO_TRECHO, carregar_chunks

RAIZ = Path(__file__).resolve().parent.parent
PARES = RAIZ / "data" / "processed" / "treino_pares.jsonl"
SPLIT = RAIZ / "data" / "processed" / "split.json"
SAIDA = RAIZ / "models" / "e5-small-boas-praticas"


def montar_dataset() -> Dataset:
    textos = {c["id"]: c["texto"] for c in carregar_chunks()}
    ids_avaliacao = set(json.loads(SPLIT.read_text())["avaliacao"])
    pares = [json.loads(l) for l in PARES.open(encoding="utf-8")]

    vazados = [p for p in pares if p["chunk_id"] in ids_avaliacao]
    assert not vazados, f"pares de treino apontam para chunks de avaliação: {vazados}"

    return Dataset.from_dict({
        "anchor": [PREFIXO_PERGUNTA + p["pergunta"] for p in pares],
        "positive": [PREFIXO_TRECHO + textos[p["chunk_id"]] for p in pares],
    })


def main():
    epocas = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dataset = montar_dataset()
    print(f"{len(dataset)} pares de treino, {epocas} épocas")

    modelo = SentenceTransformer(MODELO_BASE)
    # Negativos in-batch: os outros trechos do lote servem de exemplos errados.
    perda = MultipleNegativesRankingLoss(modelo)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(RAIZ / "models" / "checkpoints"),
        num_train_epochs=epocas,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        # Evita o mesmo trecho duas vezes no lote, o que viraria um falso negativo.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        seed=42,
    )
    SentenceTransformerTrainer(model=modelo, args=args, train_dataset=dataset, loss=perda).train()
    modelo.save(str(SAIDA))
    print(f"modelo salvo em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
