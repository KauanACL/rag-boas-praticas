---
title: Boas Práticas em Serviços de Alimentação
emoji: 🍽️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app/app.py
pinned: false
license: mit
short_description: Busca semântica na RDC 216/2004 e na cartilha da Anvisa
---

Busca semântica em português sobre boas práticas em serviços de alimentação. Usa o modelo [KauanACL/e5-small-boas-praticas](https://huggingface.co/KauanACL/e5-small-boas-praticas) + FAISS e mostra os trechos com a fonte e a página ou o item da norma.

Código, avaliação e limitações: [github.com/KauanACL/rag-boas-praticas](https://github.com/KauanACL/rag-boas-praticas).
