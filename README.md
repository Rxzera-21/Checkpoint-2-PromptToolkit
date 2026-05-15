# Prompt Toolkit — FIAP CP02

> **Disciplina:** Prompt Engineering & Artificial Intelligence  
> **Checkpoint:** 02 — Prompt Toolkit  
> **Domínio:** E-commerce · Customer Experience  
> **Stack:** Python 3.10+ · Ollama · tiktoken · matplotlib · pandas

---

## O Projeto

Toolkit Python que aplica automaticamente as 4 técnicas de prompting (**Zero-Shot, Few-Shot, Chain-of-Thought e Role Prompting**) a tarefas de e-commerce, compara resultados e recomenda a melhor abordagem.

## Estrutura

```
prompt-toolkit/
├── main.py                  # Ponto de entrada
├── requirements.txt
├── .env.example
├── src/
│   ├── llm_client.py        # Conexão Ollama (Aula 05)
│   ├── prompt_builder.py    # Anatomia de prompt (Aula 05)
│   ├── techniques.py        # 4 técnicas (Aulas 06+07)
│   ├── tasks.py             # 3 tarefas do domínio (Aula 08)
│   ├── evaluator.py         # Métricas (Aula 09)
│   └── report.py            # Gráficos e recomendações
├── data/
│   ├── inputs.json          # 5–7 inputs reais por tarefa
│   └── examples.json        # Exemplos few-shot
├── prompts/
│   ├── system_prompts.json  # 3 personas detalhadas
│   └── templates.json       # Templates por tarefa
└── output/
    ├── resultados.csv
    ├── recomendacoes.json
    └── graficos/
```

## Instalação

```bash
# 1. Clone o repositório e acesse a pasta
git clone <url-do-repo>
cd prompt-toolkit

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o ambiente
cp .env.example .env
# Edite .env se necessário (host do Ollama)
```

## Pré-requisitos

- [Ollama](https://ollama.com) instalado e rodando
- Modelo disponível localmente:

```bash
ollama serve                  # Iniciar o servidor Ollama
ollama pull gpt-oss:120b      # Baixar o modelo
```

## Execução

```bash
# Executar todas as tarefas com todas as técnicas
python main.py

# Executar apenas uma tarefa específica
python main.py --tarefa classificacao_sentimento
python main.py --tarefa extracao_dados
python main.py --tarefa sumarizacao_reclamacao

# Incluir teste de temperatura (0.1 / 0.5 / 1.0)
python main.py --temp-test

# Testar estrutura sem chamar o LLM
python main.py --dry-run

# Combinações
python main.py --tarefa classificacao_sentimento --temp-test
python main.py --dry-run --temp-test
```

## Tarefas do Domínio

| Tarefa | Tipo | Inputs | Descrição |
|--------|------|--------|-----------|
| `classificacao_sentimento` | Classificação | 7 | POSITIVO / NEGATIVO / NEUTRO / MISTO |
| `extracao_dados` | Extração | 7 | Produto, preço e defeito em JSON |
| `sumarizacao_reclamacao` | Sumarização | 5 | Resumo executivo de reclamação |

## Técnicas Implementadas

| Técnica | Arquivo | Aula |
|---------|---------|------|
| Zero-Shot | `src/techniques.py` | 06 |
| Few-Shot (3 exemplos) | `src/techniques.py` | 06 |
| Chain-of-Thought | `src/techniques.py` | 06 |
| Role Prompting | `src/techniques.py` | 07 |

## Outputs Gerados

Após a execução, os seguintes arquivos são criados em `output/`:

- `resultados.csv` — Tabela completa com todas as métricas
- `recomendacoes.json` — Melhor técnica por tarefa com justificativa
- `graficos/acuracia_por_tecnica.png` — Barras agrupadas por técnica e tarefa
- `graficos/custo_tokens_por_tecnica.png` — Tokens médios por técnica
- `graficos/consistencia_por_temperatura.png` — Consistência × temperatura (se `--temp-test`)

## Métricas Avaliadas

- **Acurácia** — % de respostas corretas vs. esperado
- **Tokens** — Contagem via Ollama + tiktoken (`cl100k_base`)
- **Latência** — Tempo de resposta em ms
- **Consistência** — Variância entre execuções (teste de temperatura)

## Variáveis de Ambiente (.env)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `OLLAMA_HOST` | `http://localhost:11434` | Host do servidor Ollama |
| `OLLAMA_MODEL` | `gpt-oss:120b` | Modelo a utilizar |
| `REQUEST_TIMEOUT` | `120` | Timeout em segundos |
| `MAX_RETRIES` | `3` | Tentativas em caso de falha |
