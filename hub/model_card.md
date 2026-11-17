---
language: pt
license: mit
base_model: intfloat/multilingual-e5-small
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - sentence-transformers
  - retrieval
  - rag
  - food-safety
  - anvisa
datasets:
  - KauanACL/boas-praticas-anvisa-qa
---

# e5-small-boas-praticas

[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) fine-tunado para busca de trechos sobre **boas práticas em serviços de alimentação**, com base na RDC 216/2004 e na cartilha da Anvisa. Faz parte do projeto [rag-boas-praticas](https://github.com/KauanACL/rag-boas-praticas) ([demo](https://huggingface.co/spaces/KauanACL/rag-boas-praticas)).

## Uso

Use os mesmos prefixos do e5: `query: ` para perguntas e `passage: ` para trechos.

```python
from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("KauanACL/e5-small-boas-praticas")
q = modelo.encode(["query: Por quanto tempo a comida pode ficar na estufa?"], normalize_embeddings=True)
p = modelo.encode(["passage: Para conservação a quente, os alimentos devem ser submetidos à temperatura superior a 60ºC por, no máximo, 6 horas."], normalize_embeddings=True)
print(q @ p.T)
```

## Dados de treino

- 172 pares (pergunta, trecho) sobre 45 chunks de treino da [cartilha da Anvisa](https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/alimentos/manuais-guias-e-orientacoes/cartilha-boas-praticas-para-servicos-de-alimentacao.pdf) e da [RDC 216/2004](https://bvsms.saude.gov.br/bvs/saudelegis/anvisa/2004/res0216_15_09_2004.html).
- As perguntas são sintéticas, escritas com ajuda de um LLM (Claude) a partir do texto de cada chunk.
- Os 21 chunks de avaliação foram separados **antes** de gerar qualquer pergunta e não aparecem no treino.

## Treinamento

`MultipleNegativesRankingLoss` (negativos dentro do lote, sampler `NO_DUPLICATES`), 3 épocas, batch 16, lr 2e-5, warmup de 10%, seed 42. Rodou em ~2 min num Apple M3 (MPS).

## Avaliação

São 84 perguntas sobre os 21 chunks de avaliação, com busca sobre todos os 66 chunks do corpus. Só conta acerto o chunk exato anotado.

| Modelo | recall@1 | recall@5 | MRR@10 |
|---|---|---|---|
| `multilingual-e5-small` (base) | 0,571 | 0,881 | 0,699 |
| **este modelo** | **0,631** | **0,905** | **0,744** |

Em recall@1, 10 perguntas passaram a acertar e 5 passaram a errar. O teste de McNemar exato dá **p = 0,30**: a melhora **não é estatisticamente significativa** com esse tamanho de avaliação. Também não medi a variância entre seeds, porque houve um único treino.

## Limitações

- O domínio é muito estreito: 2 documentos e 66 chunks. Não espere ganho fora de boas práticas em alimentação, e o modelo pode ter perdido qualidade em outros domínios, o que não foi medido.
- As perguntas sintéticas reaproveitam o vocabulário dos trechos, e perguntas reais de usuários podem ser mais difíceis.
- Os documentos são de 2004/2005 e não incluem as atualizações posteriores da norma.
- O modelo não dá respostas. Ele só ordena trechos, e a decisão sanitária continua sendo da norma vigente e da vigilância local.
