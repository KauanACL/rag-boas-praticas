"""Demo: busca de trechos sobre boas práticas em serviços de alimentação (Anvisa)."""
import os
import sys
from pathlib import Path

import gradio as gr

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from retriever import MODELO_BASE, Buscador  # noqa: E402

MODELO_LOCAL = RAIZ / "models" / "e5-small-boas-praticas"
MODELO = os.environ.get("MODELO") or (str(MODELO_LOCAL) if MODELO_LOCAL.exists() else MODELO_BASE)

buscador = Buscador(MODELO)

EXEMPLOS = [
    "Qual a temperatura para manter a comida quente no bufê?",
    "De quanto em quanto tempo devo limpar a caixa d'água?",
    "O cozinheiro pode usar anel ou relógio?",
    "Como descongelar carne com segurança?",
    "Por quanto tempo guardar os registros dos POP?",
]


def buscar(pergunta: str, k: int) -> str:
    if not pergunta.strip():
        return "Digite uma pergunta."
    blocos = []
    for i, r in enumerate(buscador.buscar(pergunta, k=int(k)), start=1):
        blocos.append(
            f"### {i}. {r['titulo']} — {r['local']}\n"
            f"<small>similaridade {r['score']:.3f} · `{r['id']}` · [fonte]({r['url']})</small>\n\n"
            f"> {r['texto']}"
        )
    return "\n\n".join(blocos)


with gr.Blocks(title="Boas Práticas em Serviços de Alimentação") as demo:
    gr.Markdown(
        "# Boas Práticas em Serviços de Alimentação\n"
        "Busca semântica na **RDC 216/2004** e na **Cartilha de Boas Práticas** da Anvisa. "
        "Mostra os trechos mais relevantes com a fonte e a página (cartilha) ou o item (RDC). "
        "Não substitui a leitura da norma nem a orientação da vigilância sanitária."
    )
    with gr.Row():
        pergunta = gr.Textbox(label="Pergunta", placeholder="Ex.: como higienizar verduras?", scale=4)
        k = gr.Slider(1, 10, value=3, step=1, label="Trechos", scale=1)
    botao = gr.Button("Buscar", variant="primary")
    saida = gr.Markdown()
    gr.Examples(EXEMPLOS, inputs=pergunta)
    gr.Markdown(f"<small>Modelo de embedding: `{MODELO}`</small>")

    botao.click(buscar, [pergunta, k], saida)
    pergunta.submit(buscar, [pergunta, k], saida)

if __name__ == "__main__":
    demo.launch()
