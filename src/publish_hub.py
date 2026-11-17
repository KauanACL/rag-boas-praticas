"""Publica modelo, dataset e Space no Hugging Face.

Usa o login feito com `hf auth login`; nenhum token passa por este script.

Uso:
    python src/publish_hub.py modelo|dataset|space
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


def dataset():
    proc = RAIZ / "data" / "processed"
    publicar(DATASET, "dataset", {
        "README.md": HUB / "dataset_card.md",
        "chunks.jsonl": proc / "chunks.jsonl",
        "treino_pares.jsonl": proc / "treino_pares.jsonl",
        "perguntas_avaliacao.jsonl": RAIZ / "eval" / "perguntas.jsonl",
    }, "Chunks, pares de treino e perguntas de avaliação")


def space():
    # A variável vem antes do código, para o primeiro build já usar o modelo fine-tunado.
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(SPACE, "MODELO", MODELO)
    publicar(SPACE, "space", {
        "README.md": HUB / "space_readme.md",
        "requirements.txt": HUB / "space_requirements.txt",
        "app/app.py": RAIZ / "app" / "app.py",
        "src/retriever.py": RAIZ / "src" / "retriever.py",
        "data/processed/chunks.jsonl": RAIZ / "data" / "processed" / "chunks.jsonl",
    }, "Demo Gradio", space_sdk="gradio")


if __name__ == "__main__":
    {"modelo": modelo, "dataset": dataset, "space": space}[sys.argv[1]]()
