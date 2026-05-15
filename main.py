"""
main.py — Ponto de entrada do Prompt Toolkit
FIAP · Checkpoint 02 · Prompt Engineering & AI

Fluxo:
  inputs.json → prompt_builder → techniques (ZS/FS/CoT/Role)
              → llm_client → evaluator → report → output/

Uso:
  python main.py
  python main.py --tarefa classificacao_sentimento
  python main.py --temp-test
  python main.py --dry-run
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from src.llm_client import LLMClient
from src.tasks import TAREFAS, get_tarefa
from src.techniques import zero_shot, few_shot, chain_of_thought, role_prompting
from src.evaluator import calcular_metricas_run, testar_temperatura
from src.report import (
    gerar_tabela,
    grafico_acuracia,
    grafico_custo,
    grafico_temperatura,
    recomendar,
    imprimir_relatorio_terminal,
)

# ──────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
INPUTS_PATH = os.path.join(BASE_DIR, "data", "inputs.json")


def carregar_inputs() -> dict:
    with open(INPUTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def executar_tecnica(
    tecnica: str,
    tarefa: dict,
    input_caso: dict,
    llm: LLMClient,
    temp: float = 0.1,
    dry_run: bool = False,
) -> dict:
    """
    Executa uma técnica para um input específico e retorna métricas.
    """
    input_texto = input_caso["input"]
    esperado = input_caso["esperado"]

    # Montar prompt conforme a técnica
    system = ""
    if tecnica == "zero_shot":
        prompt = zero_shot(tarefa, input_texto)

    elif tecnica == "few_shot":
        exemplos = tarefa.get("exemplos_fewshot", [])
        prompt = few_shot(tarefa, input_texto, exemplos)

    elif tecnica == "chain_of_thought":
        passos = tarefa.get("passos_cot", [])
        prompt = chain_of_thought(tarefa, input_texto, passos)

    elif tecnica == "role_prompting":
        persona_key = tarefa.get("persona", "analista_cx")
        system, prompt = role_prompting(tarefa, input_texto, persona_key)

    else:
        raise ValueError(f"Técnica desconhecida: {tecnica}")

    if dry_run:
        # Simula resultado sem chamar o LLM
        resultado_llm = {
            "resposta": f"[DRY-RUN] Esperado: {esperado}",
            "tokens_prompt": len(prompt.split()),
            "tokens_resposta": 5,
            "tokens_total": len(prompt.split()) + 5,
            "tempo_ms": 0,
        }
        metricas = calcular_metricas_run(resultado_llm, esperado, prompt)
        metricas["acuracia"] = 0.8  # Simulado
    else:
        resultado_llm = llm.chat(prompt=prompt, system=system, temp=temp, max_tokens=512)
        metricas = calcular_metricas_run(resultado_llm, esperado, prompt)

    return {
        "tarefa": tarefa["nome"],
        "tecnica": tecnica,
        "input_idx": input_caso.get("_idx", 0),
        "input_texto": input_texto[:80] + "..." if len(input_texto) > 80 else input_texto,
        **metricas,
    }


def main():
    parser = argparse.ArgumentParser(description="Prompt Toolkit — FIAP CP02")
    parser.add_argument(
        "--tarefa",
        type=str,
        default=None,
        help="Nome da tarefa específica a executar (padrão: todas)",
    )
    parser.add_argument(
        "--temp-test",
        action="store_true",
        help="Executar teste de temperatura no melhor prompt encontrado",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular execução sem chamar o LLM (para testes de estrutura)",
    )
    parser.add_argument(
        "--temperatura",
        type=float,
        default=0.1,
        help="Temperatura para execuções principais (padrão: 0.1)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PROMPT TOOLKIT — FIAP · Checkpoint 02")
    print("  Domínio: E-commerce · Customer Experience")
    print("=" * 70)

    # Inicializar cliente LLM
    llm = LLMClient()

    if not args.dry_run:
        print(f"\n[1/5] Verificando conexão com Ollama em {llm.host}...")
        if not llm.health_check():
            print(
                "\n  [ERRO] Ollama não está acessível!\n"
                "  Execute: ollama serve\n"
                "  E certifique-se que o modelo está disponível: "
                f"ollama pull {llm.model}\n"
            )
            sys.exit(1)
        print(f"  [OK] Ollama conectado. Modelo: {llm.model}")
    else:
        print("\n[1/5] Modo DRY-RUN — LLM não será chamado.")

    # Carregar dados
    print("\n[2/5] Carregando inputs e configurações...")
    inputs = carregar_inputs()
    tarefas_exec = [get_tarefa(args.tarefa)] if args.tarefa else TAREFAS
    tecnicas = ["zero_shot", "few_shot", "chain_of_thought", "role_prompting"]
    print(f"  [OK] {len(tarefas_exec)} tarefa(s) · {len(tecnicas)} técnicas")

    # Execução principal
    print("\n[3/5] Executando técnicas × tarefas × inputs...")
    todos_resultados = []
    total_runs = sum(
        len(inputs.get(t["nome"], [])) * len(tecnicas) for t in tarefas_exec
    )
    run_atual = 0

    for tarefa in tarefas_exec:
        nome_tarefa = tarefa["nome"]
        casos = inputs.get(nome_tarefa, [])
        if not casos:
            print(f"  [AVISO] Nenhum input encontrado para '{nome_tarefa}'. Pulando.")
            continue

        print(f"\n  Tarefa: {nome_tarefa} ({len(casos)} inputs × {len(tecnicas)} técnicas)")

        for idx, caso in enumerate(casos):
            caso["_idx"] = idx
            for tecnica in tecnicas:
                run_atual += 1
                progresso = f"[{run_atual}/{total_runs}]"
                print(f"    {progresso} {tecnica} | input {idx+1}...", end=" ", flush=True)

                resultado = executar_tecnica(
                    tecnica=tecnica,
                    tarefa=tarefa,
                    input_caso=caso,
                    llm=llm,
                    temp=args.temperatura,
                    dry_run=args.dry_run,
                )
                todos_resultados.append(resultado)

                acc = resultado["acuracia"]
                tokens = resultado["tokens_total"]
                emoji = "✓" if acc >= 0.8 else ("~" if acc >= 0.5 else "✗")
                print(f"{emoji} acc={acc:.0%} tokens={tokens}")

                if not args.dry_run:
                    time.sleep(0.5)  # Respeitar rate limit do Ollama

    if not todos_resultados:
        print("\n[ERRO] Nenhum resultado gerado. Verifique os inputs e tarefas.")
        sys.exit(1)

    # Gerar relatório
    print("\n[4/5] Gerando relatório...")
    df = gerar_tabela(todos_resultados)
    grafico_acuracia(df)
    grafico_custo(df)
    recomendacoes = recomendar(df)
    imprimir_relatorio_terminal(df, recomendacoes)

    # Salvar recomendações em JSON
    rec_path = os.path.join(BASE_DIR, "output", "recomendacoes.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        # Serializar sem os sub-dicts complexos para JSON limpo
        rec_simples = {
            k: {
                "tecnica": v["tecnica"],
                "nome_amigavel": v["nome_amigavel"],
                "acuracia_media": v["acuracia_media"],
                "tokens_medio": v["tokens_medio"],
                "justificativa": v["justificativa"],
            }
            for k, v in recomendacoes.items()
        }
        json.dump(rec_simples, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Recomendações salvas em: {rec_path}")

    # Teste de temperatura (opcional)
    if args.temp_test:
        print("\n[5/5] Executando teste de temperatura...")
        # Usar a primeira tarefa e o primeiro input para o teste
        tarefa_teste = tarefas_exec[0]
        caso_teste = inputs.get(tarefa_teste["nome"], [{}])[0]
        input_teste = caso_teste.get("input", "Produto excelente!")

        # Usar zero_shot para o teste de temperatura
        prompt_teste = zero_shot(tarefa_teste, input_teste)

        print(f"  Tarefa: {tarefa_teste['nome']}")
        print(f"  Input: {input_teste[:60]}...")
        print(f"  Temperaturas: 0.1, 0.5, 1.0 (3 execuções cada)")

        if args.dry_run:
            print("  [DRY-RUN] Simulando resultados de temperatura...")
            resultados_temp = [
                {"temperatura": 0.1, "consistencia": 1.0, "tokens_medio": 45, "tempo_medio_ms": 0},
                {"temperatura": 0.5, "consistencia": 0.67, "tokens_medio": 52, "tempo_medio_ms": 0},
                {"temperatura": 1.0, "consistencia": 0.33, "tokens_medio": 61, "tempo_medio_ms": 0},
            ]
        else:
            resultados_temp = testar_temperatura(llm, prompt_teste, temps=[0.1, 0.5, 1.0])

        grafico_temperatura(resultados_temp)

        print("\n  Resultados por temperatura:")
        print(f"  {'Temp':>6} {'Consistência':>14} {'Tokens Méd':>12}")
        print(f"  {'─'*6} {'─'*14} {'─'*12}")
        for r in resultados_temp:
            print(
                f"  {r['temperatura']:>6.1f} {r['consistencia']:>13.0%} "
                f"{r.get('tokens_medio', 0):>12.0f}"
            )
    else:
        print("\n[5/5] Teste de temperatura pulado. Use --temp-test para ativar.")

    print("\n" + "=" * 70)
    print("  EXECUÇÃO CONCLUÍDA")
    print(f"  Resultados em: output/resultados.csv")
    print(f"  Gráficos em:   output/graficos/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

