"""Baixa os documentos oficiais para data/raw/."""
from pathlib import Path

import httpx

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

FONTES = {
    "cartilha_anvisa.pdf": "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/alimentos/manuais-guias-e-orientacoes/cartilha-boas-praticas-para-servicos-de-alimentacao.pdf/@@display-file/file",
    "rdc_216_2004.html": "https://bvsms.saude.gov.br/bvs/saudelegis/anvisa/2004/res0216_15_09_2004.html",
}


def baixar(nome: str, url: str) -> None:
    destino = RAW / nome
    if destino.exists():
        print(f"já existe: {destino.name}")
        return
    resp = httpx.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    conteudo = resp.content
    if nome.endswith(".pdf") and not conteudo.startswith(b"%PDF"):
        raise ValueError(f"{nome}: resposta não é um PDF")
    if nome.endswith(".html") and b"<html" not in conteudo[:2000].lower():
        raise ValueError(f"{nome}: resposta não é HTML")
    destino.write_bytes(conteudo)
    print(f"baixado: {destino.name} ({len(conteudo) / 1024:.0f} KB)")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for nome, url in FONTES.items():
        baixar(nome, url)
