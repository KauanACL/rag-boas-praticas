"""Extrai, limpa e divide os documentos de data/raw/ em chunks.

Saída: data/processed/chunks.jsonl, um chunk por linha.
"""
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
SAIDA = RAIZ / "data" / "processed" / "chunks.jsonl"

TAMANHO = 180  # palavras por chunk
SOBREPOSICAO = 40  # palavras repetidas entre chunks vizinhos

DOCUMENTOS = [
    {
        "arquivo": "cartilha_anvisa.pdf",
        "doc": "cartilha",
        "titulo": "Cartilha sobre Boas Práticas para Serviços de Alimentação (Anvisa)",
        "url": "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/alimentos/manuais-guias-e-orientacoes/cartilha-boas-praticas-para-servicos-de-alimentacao.pdf",
        "paginas": (4, 42),  # 1-3 são capa, créditos e sumário; 43-44 são contatos e contracapa
    },
    {
        "arquivo": "rdc_216_2004.html",
        "doc": "rdc216",
        "titulo": "Resolução RDC nº 216/2004 (Anvisa)",
        "url": "https://bvsms.saude.gov.br/bvs/saudelegis/anvisa/2004/res0216_15_09_2004.html",
    },
]


def juntar_hifenizacao(texto: str) -> str:
    # "deve-\nse" -> "deve-se" (pronome oblíquo mantém o hífen)
    texto = re.sub(r"(\w)-\s*\n\s*(se|lo|la|los|las|lhe|lhes|me|te|nos)\b", r"\1-\2", texto)
    # "Alimen-\ntação" -> "Alimentação"
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", texto)


def normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()


def ler_pdf(caminho: Path, paginas_uteis: tuple[int, int]) -> list[tuple[str, str]]:
    """Retorna [(texto, "p. N")] por página, sem cabeçalhos/rodapés repetidos."""
    paginas = [p.extract_text() or "" for p in PdfReader(caminho).pages]
    primeira, ultima = paginas_uteis

    # Linha que aparece (ignorando dígitos) em mais de 30% das páginas é cabeçalho ou rodapé.
    def chave(linha: str) -> str:
        return re.sub(r"\d+", "", linha).strip().lower()

    freq = Counter(k for p in paginas for k in {chave(l) for l in p.splitlines()} if k)
    repetidas = {k for k, n in freq.items() if n > 0.3 * len(paginas)}

    unidades = []
    for num, pagina in enumerate(paginas, start=1):
        if not primeira <= num <= ultima:
            continue
        linhas = [
            l for l in pagina.splitlines()
            if chave(l) not in repetidas
            and not re.fullmatch(r"\s*\d+\s*", l)  # número de página solto
            and not re.fullmatch(r"\s*Capítulo \d+\s*", l)  # marcador de capítulo
        ]
        texto = normalizar(juntar_hifenizacao("\n".join(linhas)))
        if texto:
            unidades.append((texto, f"p. {num}"))
    return unidades


class _Paragrafos(HTMLParser):
    """Coleta o texto de cada <p>, <h1> e <li> do <body>, ignorando o aviso do topo."""

    BLOCOS = {"p", "h1", "h2", "h3", "li"}

    def __init__(self):
        super().__init__()
        self.paragrafos: list[str] = []
        self._atual: list[str] = []
        self._ignorar = 0
        self._no_body = False

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self._no_body = True
        if tag == "div" and dict(attrs).get("id") == "advert":
            self._ignorar = 1
        elif tag == "div" and self._ignorar:
            self._ignorar += 1
        if tag in self.BLOCOS:
            self._fechar()

    def handle_endtag(self, tag):
        if tag == "div" and self._ignorar:
            self._ignorar -= 1
        if tag in self.BLOCOS:
            self._fechar()

    def handle_data(self, data):
        if self._no_body and not self._ignorar:
            self._atual.append(data)

    def _fechar(self):
        texto = normalizar("".join(self._atual))
        if texto:
            self.paragrafos.append(texto)
        self._atual = []


def ler_html(caminho: Path) -> list[tuple[str, str]]:
    """Retorna [(texto, "item X.Y")] por parágrafo, herdando o último item numerado."""
    parser = _Paragrafos()
    parser.feed(caminho.read_bytes().decode("latin-1"))
    parser.close()

    unidades, secao = [], "preâmbulo"
    for par in parser.paragrafos:
        if m := re.match(r"(\d+(?:\.\d+)+)\.?\s", par):
            secao = f"item {m.group(1)}"
        elif m := re.match(r"Art\. (\d+)", par):
            secao = f"art. {m.group(1)}"
        elif par.startswith("ANEXO"):
            secao = "anexo"
        unidades.append((par, secao))
    return unidades


def dividir(unidades: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Janela deslizante de TAMANHO palavras; o local é o da primeira palavra."""
    palavras = [(w, local) for texto, local in unidades for w in texto.split()]
    passo = TAMANHO - SOBREPOSICAO
    chunks = []
    for inicio in range(0, len(palavras), passo):
        janela = palavras[inicio:inicio + TAMANHO]
        chunks.append((" ".join(w for w, _ in janela), janela[0][1]))
        if inicio + TAMANHO >= len(palavras):
            break
    return chunks


def main():
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with SAIDA.open("w", encoding="utf-8") as f:
        for d in DOCUMENTOS:
            caminho = RAW / d["arquivo"]
            if caminho.suffix == ".pdf":
                unidades = ler_pdf(caminho, d["paginas"])
            else:
                unidades = ler_html(caminho)
            chunks = dividir(unidades)
            for i, (texto, local) in enumerate(chunks):
                registro = {
                    "id": f"{d['doc']}-{i:04d}",
                    "doc": d["doc"],
                    "titulo": d["titulo"],
                    "url": d["url"],
                    "local": local,
                    "texto": texto,
                }
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            print(f"{d['doc']}: {len(chunks)} chunks")
            total += len(chunks)
    print(f"total: {total} chunks -> {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
