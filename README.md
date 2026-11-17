# RAG de Boas Práticas em Serviços de Alimentação (Anvisa)

Assistente de busca em português sobre as regras sanitárias para restaurantes, lanchonetes, padarias e outros serviços de alimentação. A base são documentos oficiais da Anvisa. O foco do projeto é **medir a qualidade da busca** e tentar melhorá-la com um **modelo de embedding fine-tunado**.

- Demo: [Space no Hugging Face](https://huggingface.co/spaces/KauanACL/rag-boas-praticas) (estático: o modelo roda no seu navegador)
- Modelo: [KauanACL/e5-small-boas-praticas](https://huggingface.co/KauanACL/e5-small-boas-praticas)
- Dataset: [KauanACL/boas-praticas-anvisa-qa](https://huggingface.co/datasets/KauanACL/boas-praticas-anvisa-qa)

## Problema

Quem trabalha com comida precisa seguir a RDC 216/2004. Mas dúvidas práticas, como "por quanto tempo posso deixar a comida na estufa?", se espalham por dezenas de itens da norma e pela cartilha da Anvisa. Um RAG só é útil se a **recuperação** acerta o trecho certo, então esse é o componente medido aqui.

## Arquitetura

```mermaid
flowchart LR
    A[PDF da cartilha<br/>HTML da RDC 216] -->|download.py| B[data/raw]
    B -->|ingest.py<br/>limpeza + chunks de 180 palavras, sobreposição 40| C[chunks.jsonl<br/>66 chunks]
    C -->|split.py| D{treino 45 / avaliação 21}
    D -->|perguntas sobre chunks de treino| E[finetune.py<br/>MNRL, 3 épocas]
    D -->|84 perguntas sobre chunks de avaliação| F[evaluate.py<br/>recall@1, recall@5, MRR]
    C --> G[retriever.py<br/>e5-small + FAISS]
    E --> G
    G --> F
    G --> H[app/app.py<br/>Gradio local: trechos + fonte + página/item]
    E -->|export_onnx.py<br/>ONNX int8| I[space/<br/>HTML + transformers.js<br/>Space estático]
```

**Fontes** (só oficiais):
- [Cartilha sobre Boas Práticas para Serviços de Alimentação](https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/alimentos/manuais-guias-e-orientacoes/cartilha-boas-praticas-para-servicos-de-alimentacao.pdf) (Anvisa, PDF, páginas 4–42)
- [Resolução RDC nº 216/2004](https://bvsms.saude.gov.br/bvs/saudelegis/anvisa/2004/res0216_15_09_2004.html) (texto oficial em HTML na BVS do Ministério da Saúde; cita o item da norma em vez da página)

**Componentes:**
- Embedding: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (118M parâmetros), com os prefixos `query:`/`passage:`.
- Índice: FAISS `IndexFlatIP` sobre vetores normalizados, ou seja, similaridade de cosseno exata.
- Sem framework de orquestração: cerca de 420 linhas de Python em `src/`, contando comentários.

## Avaliação

**Separação feita antes de gerar qualquer pergunta** (`src/split.py`): os 66 chunks são divididos em blocos de 3 consecutivos, e 1 a cada 3 blocos vai para avaliação. O resultado é 21 chunks de avaliação e 45 de treino. As 84 perguntas de avaliação (`eval/perguntas.jsonl`, 4 por chunk) apontam só para os chunks de avaliação. As 172 perguntas de treino apontam só para os chunks de treino, e `finetune.py` recusa rodar se encontrar um par de treino que aponte para um chunk de avaliação.

**Critério estrito:** a busca roda sobre **todos os 66 chunks**, e só conta acerto o chunk exato anotado. Se a resposta aparece em um chunk vizinho ou no outro documento, conta como erro.

### Resultados (84 perguntas)

| Modelo | recall@1 | recall@5 | MRR@10 |
|---|---|---|---|
| `multilingual-e5-small` (baseline) | 0,571 | 0,881 | 0,699 |
| `e5-small-boas-praticas` (fine-tunado) | **0,631** | **0,905** | **0,744** |

Valores de `eval/resultados.json`, gerados por `python src/evaluate.py [modelo]`.

**Versão que roda na demo.** O Space é estático, porque o Hugging Face passou a cobrar por Spaces Gradio. Por isso o modelo é exportado para ONNX e quantizado em int8 (118 MB em vez de 470 MB), para rodar no navegador.

| Versão do modelo fine-tunado | recall@1 | recall@5 | MRR@10 |
|---|---|---|---|
| ONNX fp32 (onnxruntime, Python) | 0,631 | 0,905 | 0,744 |
| ONNX int8 (onnxruntime, Python) | 0,583 | 0,893 | 0,714 |
| ONNX int8 no navegador (transformers.js 4.3.2, o que o Space usa) | 0,595 | 0,881 | 0,721 |

- O fp32 reproduz exatamente o PyTorch, o que valida a exportação.
- **A quantização int8 consome a maior parte do ganho em recall@1**: a demo fica entre o baseline e o modelo fine-tunado completo.
- Testei também a quantização por canal (recall@1 0,619, recall@5 0,869) e não a adotei, porque escolher a variante olhando o conjunto de avaliação seria ajustar ao teste.
- A pequena diferença entre o int8 no Python e no navegador deve vir dos kernels int8 do onnxruntime-web, que diferem dos do onnxruntime em Python, mas não verifiquei isso. O número do navegador foi medido rodando as 84 perguntas no Chrome.

**Leitura honesta do ganho:** em recall@1, o fine-tuning corrigiu 10 perguntas e piorou 5, um ganho líquido de 5 em 84. O teste de McNemar exato sobre essas 15 discordâncias dá **p = 0,30**, então **o ganho não é estatisticamente significativo** com esse tamanho de avaliação. A direção é positiva nas três métricas, mas pode ser ruído. Também não medi a variância entre seeds: houve um único treino, com seed 42.

**Para onde vão os erros de top-1:**

| Modelo | erros | chunk vizinho (mesmo doc) | outro documento | mesmo doc, longe |
|---|---|---|---|---|
| baseline | 36 | 11 | 15 | 10 |
| fine-tunado | 31 | 9 | 13 | 9 |

Cerca de 40% dos erros vão para o **outro documento**. A cartilha e a RDC falam das mesmas regras (60 °C, 5 °C, caixa d'água, uniforme), então muitas vezes o "erro" também traz uma resposta válida. Para um usuário, o recall@1 estrito subestima a utilidade.

## Fine-tuning

- Dados: 172 pares (pergunta, trecho) sobre os 45 chunks de treino.
- Perda: `MultipleNegativesRankingLoss`, com negativos dentro do lote e `NO_DUPLICATES` para não ter o mesmo trecho duas vezes no lote.
- Hiperparâmetros: 3 épocas, batch 16, lr 2e-5, warmup de 10%.
- Hardware: Apple M3 (MPS). O treino levou ~2 min, então não foi preciso usar GPU na nuvem.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/download.py      # baixa as fontes oficiais para data/raw/
python src/ingest.py        # gera data/processed/chunks.jsonl
python src/split.py         # separa treino/avaliação
python src/evaluate.py      # avalia o baseline
python src/finetune.py 3    # fine-tuning (~2 min no M3), salva em models/
python src/evaluate.py models/e5-small-boas-praticas
python app/app.py           # demo Gradio local em http://127.0.0.1:7860

# Space estático
pip install onnxruntime
python src/export_onnx.py   # models/onnx/ (fp32 + int8)
python src/evaluate.py models/onnx/onnx/model_quantized.onnx
python src/build_space.py   # space/chunks.json com embeddings pré-calculados
```

Para usar o modelo publicado no app Gradio: `MODELO=KauanACL/e5-small-boas-praticas python app/app.py`.

## Estrutura

```
src/download.py   baixa as fontes
src/ingest.py     extração (pypdf / html.parser), limpeza e chunking
src/split.py      divisão treino/avaliação por blocos
src/retriever.py  embeddings + FAISS
src/evaluate.py   recall@1, recall@5, MRR@10
src/finetune.py   fine-tuning com sentence-transformers
src/export_onnx.py    exportação ONNX + quantização int8
src/onnx_encoder.py   codificador ONNX (mesmo pooling do transformers.js)
src/build_space.py    embeddings dos chunks para o Space
src/publish_hub.py    publicação no Hugging Face
app/app.py        demo Gradio (local)
space/            Space estático: index.html + chunks.json
eval/             perguntas de avaliação e resultados
data/processed/   chunks, split e pares de treino
```

## Limitações

- **Avaliação pequena:** 84 perguntas sobre 21 chunks, o que deixa intervalos de confiança largos (veja o p-valor acima).
- **Perguntas sintéticas:** as perguntas de avaliação e de treino foram escritas com ajuda de um LLM (Claude), a partir do texto dos chunks. Elas tendem a reaproveitar o vocabulário do trecho, o que facilita a busca em relação a perguntas reais de usuários.
- **Vazamento parcial:** chunks vizinhos compartilham 40 palavras. A divisão em blocos limita isso às bordas dos blocos de avaliação, mas não elimina.
- **Corpus pequeno e de 2004:** são 2 documentos e 66 chunks. A RDC 216 foi complementada depois por outras normas (por exemplo, a RDC 52/2014), que não estão no corpus. O app não substitui a norma vigente nem a vigilância sanitária local.
- **Extração do PDF:** o pypdf às vezes coloca o título de uma seção depois do parágrafo dela, pela diagramação em caixas da cartilha.
- **Sem geração de resposta:** o app só recupera e mostra os trechos com a fonte. A geração por LLM, com citação e recusa quando não há resposta, ficou de fora porque exigiria uma API paga.
- **Demo quantizada:** o Space usa a versão int8, que é mais fraca que o modelo completo em recall@1 (veja a tabela acima). A primeira visita baixa ~118 MB.
