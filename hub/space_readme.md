---
title: Boas Práticas em Serviços de Alimentação
emoji: 🍽️
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: mit
short_description: Busca semântica na RDC 216/2004 e na cartilha da Anvisa
---

Busca semântica em português sobre boas práticas em serviços de alimentação. O modelo [KauanACL/e5-small-boas-praticas](https://huggingface.co/KauanACL/e5-small-boas-praticas) roda **no seu navegador** (ONNX int8 via transformers.js) e compara a pergunta com os embeddings pré-calculados dos 66 trechos. Os resultados vêm com a fonte e a página ou o item da norma.

Código, avaliação e limitações: [github.com/KauanACL/rag-boas-praticas](https://github.com/KauanACL/rag-boas-praticas).
