"""Publica modelo, dataset e Space no Hugging Face.

Usa o login feito com `hf auth login`; nenhum token passa por este script.

Uso:
    python src/publish_hub.py modelo|onnx|dataset|space
"""
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

RAIZ = Path(__file__).resolve().parent.parent
HUB = RAIZ / "hub"
USUARIO = "KauanACL"
MODELO = f"{USUARIO}/e5-small-boas-praticas"
DATASET = f"{USUARIO}/boas-praticas-anvisa-qa"
SPACE = f"{USUARIO}/rag-boas-praticas"

api = HfApi()


def publicar(repo_id: str, tipo: str, arquivos: dict[str, Path], mensagem: str, **criar):
    api.create_repo(repo_id, repo_type=tipo, exist_ok=True, **criar)
    ops = [CommitOperationAdd(destino, str(origem)) for destino, origem in arquivos.items()]
    api.create_commit(repo_id, ops, commit_message=mensagem, repo_type=tipo)
    print(f"publicado: https://huggingface.co/{'' if tipo == 'model' else tipo + 's/'}{repo_id}")


def modelo():
    pasta = RAIZ / "models" / "e5-small-boas-praticas"
    arquivos = {str(p.relative_to(pasta)): p for p in pasta.rglob("*") if p.is_file()}
    arquivos["README.md"] = HUB / "model_card.md"
    publicar(MODELO, "model", arquivos, "Modelo fine-tunado + model card")


def onnx():
    # Versão int8 para o transformers.js (Space estático), no mesmo repositório do modelo.
    publicar(MODELO, "model", {
        "onnx/model_quantized.onnx": RAIZ / "models" / "onnx" / "onnx" / "model_quantized.onnx",
        "README.md": HUB / "model_card.md",
    }, "ONNX int8 para transformers.js")


def dataset():
    proc = RAIZ / "data" / "processed"
    publicar(DATASET, "dataset", {
        "README.md": HUB / "dataset_card.md",
        "chunks.jsonl": proc / "chunks.jsonl",
        "treino_pares.jsonl": proc / "treino_pares.jsonl",
        "perguntas_avaliacao.jsonl": RAIZ / "eval" / "perguntas.jsonl",
    }, "Chunks, pares de treino e perguntas de avaliação")


def space():
    # Space estático (grátis): o modelo roda no navegador via transformers.js.
    publicar(SPACE, "space", {
        "README.md": HUB / "space_readme.md",
        "index.html": RAIZ / "space" / "index.html",
        "chunks.json": RAIZ / "space" / "chunks.json",
    }, "Demo estática (transformers.js)", space_sdk="static")


if __name__ == "__main__":
    {"modelo": modelo, "onnx": onnx, "dataset": dataset, "space": space}[sys.argv[1]]()
