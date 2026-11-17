---
language: pt
license: other
license_name: anvisa-reproducao-com-citacao
pretty_name: Boas Práticas Anvisa — chunks e perguntas
task_categories:
  - question-answering
  - sentence-similarity
tags:
  - rag
  - retrieval
  - food-safety
  - anvisa
size_categories:
  - n<1K
configs:
  - config_name: chunks
    data_files: chunks.jsonl
  - config_name: perguntas
    data_files:
      - split: train
        path: treino_pares.jsonl
      - split: test
        path: perguntas_avaliacao.jsonl
---

# Boas Práticas Anvisa — chunks e perguntas

Corpus e perguntas usados para avaliar e fine-tunar a busca do projeto [rag-boas-praticas](https://github.com/KauanACL/rag-boas-praticas).

## Fontes

- [Cartilha sobre Boas Práticas para Serviços de Alimentação](https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/alimentos/manuais-guias-e-orientacoes/cartilha-boas-praticas-para-servicos-de-alimentacao.pdf), Anvisa, 3ª ed., páginas 4–42. A obra permite reprodução parcial ou total, desde que a fonte seja citada, e proíbe a comercialização.
- [Resolução RDC nº 216, de 15/09/2004](https://bvsms.saude.gov.br/bvs/saudelegis/anvisa/2004/res0216_15_09_2004.html), texto da BVS/Ministério da Saúde. É ato normativo oficial (Lei 9.610/98, art. 8º, IV).

## Arquivos

| Config / split | Arquivo | Linhas | Campos |
|---|---|---|---|
| `chunks` | `chunks.jsonl` | 66 | `id`, `doc`, `titulo`, `url`, `local` (página ou item da norma), `texto` |
| `perguntas` / `train` | `treino_pares.jsonl` | 172 | `pergunta`, `chunk_id` |
| `perguntas` / `test` | `perguntas_avaliacao.jsonl` | 84 | `id`, `pergunta`, `chunk_id` |

Os chunks têm 180 palavras, com sobreposição de 40 entre vizinhos. Na limpeza, foram removidos cabeçalhos e rodapés repetidos, números de página e marcadores de capítulo.

## Como foi construído

1. Os 66 chunks foram divididos em blocos de 3 consecutivos, e 1 a cada 3 blocos foi para avaliação. Deu 21 chunks de avaliação e 45 de treino. Essa divisão foi feita antes de escrever qualquer pergunta.
2. As perguntas foram escritas com ajuda de um LLM (Claude), mirando o miolo de cada chunk (as palavras 40–140, que não se repetem com os vizinhos). Há 4 perguntas por chunk de avaliação e 2 a 4 por chunk de treino.
3. Cada pergunta aponta para um único `chunk_id`. Nenhuma pergunta de treino aponta para um chunk de avaliação.

## Limitações

- As perguntas são sintéticas e tendem a usar o vocabulário do próprio trecho.
- Cada pergunta tem um único chunk-alvo, mas a cartilha e a RDC repetem muitas regras, então outros chunks podem responder a mesma pergunta.
- Chunks de treino vizinhos a um bloco de avaliação compartilham até 40 palavras com ele.
- Os textos são de 2004/2005 e não refletem atualizações posteriores da norma.
