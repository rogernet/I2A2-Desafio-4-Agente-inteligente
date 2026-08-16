# nota.ask — Interface Inteligente para Consulta de Arquivos CSV

Desafio 4 — InsurMinds / I2A2. Um agente que responde perguntas em linguagem
natural sobre notas fiscais armazenadas em CSV. A pergunta vira SQL, o SQL roda
sobre os dados e a resposta é montada a partir do resultado real da consulta.

## Arquitetura

- **Interface A — Carga** (`app.py`, aba Carga): recebe um `.ZIP` com CSVs e um
  dicionário de dados. O módulo `ingestion.py` descompacta, lê os CSVs e carrega
  cada um como uma tabela em um arquivo DuckDB. O dicionário é extraído como
  texto para orientar o agente.
- **Interface B — Consulta** (`app.py`, aba Consulta): o usuário pergunta em
  português. O módulo `agent.py` usa LangChain para gerar o SQL, valida que a
  consulta é somente de leitura, executa no DuckDB e resume o resultado.
- **Guardrails**: SQL não-SELECT é bloqueado; erros de execução e perguntas sem
  resposta nos dados retornam mensagem clara.

Fluxo: `ZIP → descompacta → DuckDB → schema + dicionário → pergunta → SQL →
execução → resposta (texto + tabela + gráfico)`.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Execução

Rodando local com Ollama (padrão):

```bash
ollama pull qwen2.5-coder:3b
streamlit run app.py
```

Para usar uma API, ajuste `LLM_PROVIDER` no `.env` para `openai` ou `anthropic`
e preencha a chave correspondente.

## Uso

1. Aba **Carga**: suba o `.ZIP` (ex.: `202401_NFs.zip`) e clique em Processar.
2. Aba **Consulta**: faça perguntas como:
   - Quais foram os cinco maiores fornecedores por valor total?
   - Qual foi o total gasto em cada mês?
   - Qual produto teve o maior volume comprado?

## Tecnologias

Python, Streamlit, DuckDB, LangChain, e um provedor de LLM (Ollama local,
OpenAI ou Anthropic).

## Observações

CSVs de notas fiscais que variam entre UTF-8 e latin-1, com separador `;`
e decimal com vírgula. A ingestão detecta a codificação e o separador, normaliza os
nomes de coluna (com transliteração de acentos) e ajusta os tipos automaticamente:
identificadores como texto, valores e quantidades como número, datas como data.

## Licença

MIT.
