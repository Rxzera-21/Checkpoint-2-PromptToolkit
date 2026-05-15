"""
report.py — Aula 09
Geração de tabelas, gráficos comparativos e recomendações.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sem GUI para salvar PNGs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
GRAFICOS_DIR = os.path.join(OUTPUT_DIR, "graficos")

# Paleta de cores por técnica
CORES_TECNICAS = {
    "zero_shot": "#4C72B0",
    "few_shot": "#DD8452",
    "chain_of_thought": "#55A868",
    "role_prompting": "#C44E52",
}

NOMES_AMIGAVEIS = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "chain_of_thought": "Chain-of-Thought",
    "role_prompting": "Role Prompting",
}


def gerar_tabela(resultados: list[dict]) -> pd.DataFrame:
    """
    Gera DataFrame comparativo e salva CSV.

    Args:
        resultados: Lista de dicts com resultados de cada execução.

    Returns:
        DataFrame pandas com os resultados.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame(resultados)

    # Reordenar colunas para legibilidade
    colunas_ordem = [
        "tarefa", "tecnica", "input_idx",
        "acuracia", "tokens_total", "tempo_ms",
        "tokens_prompt", "tokens_resposta", "resposta"
    ]
    colunas_presentes = [c for c in colunas_ordem if c in df.columns]
    df = df[colunas_presentes + [c for c in df.columns if c not in colunas_presentes]]

    csv_path = os.path.join(OUTPUT_DIR, "resultados.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] CSV salvo em: {csv_path}")

    return df


def _preparar_dados_por_tecnica(df: pd.DataFrame, coluna: str) -> tuple:
    """Prepara dados agrupados por tarefa e técnica para plotagem."""
    tarefas = df["tarefa"].unique()
    tecnicas = df["tecnica"].unique()

    medias = {}
    for tecnica in tecnicas:
        medias[tecnica] = []
        for tarefa in tarefas:
            subset = df[(df["tarefa"] == tarefa) & (df["tecnica"] == tecnica)]
            if len(subset) > 0:
                medias[tecnica].append(subset[coluna].mean())
            else:
                medias[tecnica].append(0)

    return tarefas, tecnicas, medias


def grafico_acuracia(df: pd.DataFrame) -> str:
    """
    Gráfico de barras agrupadas: acurácia por técnica e tarefa.

    Returns:
        Caminho do arquivo PNG salvo.
    """
    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    tarefas, tecnicas, medias = _preparar_dados_por_tecnica(df, "acuracia")
    n_tarefas = len(tarefas)
    n_tecnicas = len(tecnicas)

    x = np.arange(n_tarefas)
    largura = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for i, tecnica in enumerate(tecnicas):
        offset = (i - n_tecnicas / 2 + 0.5) * largura
        cor = CORES_TECNICAS.get(tecnica, "#888888")
        nome = NOMES_AMIGAVEIS.get(tecnica, tecnica)
        bars = ax.bar(
            x + offset,
            medias[tecnica],
            largura,
            label=nome,
            color=cor,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )
        # Rótulos nas barras
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.02,
                    f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    ax.set_xlabel("Tarefa", fontsize=11)
    ax.set_ylabel("Acurácia Média", fontsize=11)
    ax.set_title("Acurácia por Técnica e Tarefa\nPrompt Toolkit — FIAP CP02", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    tarefas_labels = [t.replace("_", "\n") for t in tarefas]
    ax.set_xticklabels(tarefas_labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, "acuracia_por_tecnica.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Gráfico de acurácia salvo em: {path}")
    return path


def grafico_custo(df: pd.DataFrame) -> str:
    """
    Gráfico de barras: tokens médios por técnica (custo estimado).

    Returns:
        Caminho do arquivo PNG salvo.
    """
    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    resumo = (
        df.groupby("tecnica")["tokens_total"]
        .mean()
        .reset_index()
        .rename(columns={"tokens_total": "tokens_medio"})
    )
    resumo["nome_amigavel"] = resumo["tecnica"].map(
        lambda t: NOMES_AMIGAVEIS.get(t, t)
    )
    resumo = resumo.sort_values("tokens_medio", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    cores = [CORES_TECNICAS.get(t, "#888888") for t in resumo["tecnica"]]
    bars = ax.barh(
        resumo["nome_amigavel"],
        resumo["tokens_medio"],
        color=cores,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 2, bar.get_y() + bar.get_height() / 2,
            f"{w:.0f} tokens",
            ha="left", va="center", fontsize=9,
        )

    ax.set_xlabel("Tokens Médios por Execução", fontsize=11)
    ax.set_title("Custo (Tokens) por Técnica\nPrompt Toolkit — FIAP CP02", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, "custo_tokens_por_tecnica.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Gráfico de custo salvo em: {path}")
    return path


def grafico_temperatura(resultados_temp: list[dict]) -> str:
    """
    Gráfico de linha: consistência por temperatura.

    Args:
        resultados_temp: Lista retornada por evaluator.testar_temperatura().

    Returns:
        Caminho do arquivo PNG salvo.
    """
    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    if not resultados_temp:
        print("  [AVISO] Sem dados de temperatura para plotar.")
        return ""

    temps = [r["temperatura"] for r in resultados_temp]
    consistencias = [r["consistencia"] * 100 for r in resultados_temp]
    tokens = [r.get("tokens_medio", 0) for r in resultados_temp]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax1.set_facecolor("#F8F9FA")

    # Linha de consistência
    ax1.plot(temps, consistencias, "o-", color="#4C72B0", linewidth=2.5,
             markersize=8, label="Consistência (%)", zorder=3)
    ax1.fill_between(temps, consistencias, alpha=0.15, color="#4C72B0")
    ax1.set_xlabel("Temperatura", fontsize=11)
    ax1.set_ylabel("Consistência (%)", fontsize=11, color="#4C72B0")
    ax1.set_ylim(0, 110)
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    for x, y in zip(temps, consistencias):
        ax1.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="#4C72B0",
                     fontweight="bold")

    # Linha de tokens no eixo secundário
    ax2 = ax1.twinx()
    ax2.plot(temps, tokens, "s--", color="#DD8452", linewidth=2,
             markersize=7, label="Tokens Médios", alpha=0.8)
    ax2.set_ylabel("Tokens Médios", fontsize=11, color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    # Legenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title(
        "Consistência vs. Temperatura\nPrompt Toolkit — FIAP CP02",
        fontsize=13, fontweight="bold"
    )
    ax1.set_xticks(temps)
    ax1.grid(axis="y", alpha=0.3)
    fig.patch.set_facecolor("#F8F9FA")

    plt.tight_layout()
    path = os.path.join(GRAFICOS_DIR, "consistencia_por_temperatura.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Gráfico de temperatura salvo em: {path}")
    return path


def recomendar(df: pd.DataFrame) -> dict:
    """
    Recomenda a melhor técnica por tarefa com base nos resultados.

    Args:
        df: DataFrame com todos os resultados.

    Returns:
        Dict tarefa -> dict com técnica recomendada e justificativa.
    """
    recomendacoes = {}

    for tarefa in df["tarefa"].unique():
        df_tarefa = df[df["tarefa"] == tarefa]

        resumo = (
            df_tarefa.groupby("tecnica")
            .agg(
                acuracia_media=("acuracia", "mean"),
                tokens_medio=("tokens_total", "mean"),
                tempo_medio=("tempo_ms", "mean"),
            )
            .reset_index()
        )

        # Score composto: 70% acurácia, 30% eficiência (menor token = melhor)
        max_tokens = resumo["tokens_medio"].max()
        if max_tokens > 0:
            resumo["eficiencia"] = 1 - (resumo["tokens_medio"] / max_tokens)
        else:
            resumo["eficiencia"] = 1.0

        resumo["score"] = 0.7 * resumo["acuracia_media"] + 0.3 * resumo["eficiencia"]

        melhor = resumo.loc[resumo["score"].idxmax()]
        tecnica_melhor = melhor["tecnica"]
        nome_amigavel = NOMES_AMIGAVEIS.get(tecnica_melhor, tecnica_melhor)

        # Gera justificativa automática
        acc = melhor["acuracia_media"]
        tokens = melhor["tokens_medio"]
        justificativa = (
            f"{nome_amigavel} obteve a melhor performance na tarefa '{tarefa}' "
            f"com acurácia média de {acc:.0%} e consumo médio de {tokens:.0f} tokens."
        )

        recomendacoes[tarefa] = {
            "tecnica": tecnica_melhor,
            "nome_amigavel": nome_amigavel,
            "acuracia_media": round(acc, 3),
            "tokens_medio": round(tokens, 1),
            "score_composto": round(float(melhor["score"]), 3),
            "justificativa": justificativa,
            "resumo_por_tecnica": resumo.to_dict(orient="records"),
        }

    return recomendacoes


def imprimir_relatorio_terminal(df: pd.DataFrame, recomendacoes: dict) -> None:
    """Imprime relatório formatado no terminal."""
    print("\n" + "=" * 70)
    print("  RELATÓRIO COMPARATIVO — PROMPT TOOLKIT FIAP CP02")
    print("=" * 70)

    for tarefa in df["tarefa"].unique():
        print(f"\n{'─'*70}")
        print(f"  TAREFA: {tarefa.upper()}")
        print(f"{'─'*70}")

        df_tarefa = df[df["tarefa"] == tarefa]
        resumo = (
            df_tarefa.groupby("tecnica")
            .agg(
                acuracia_media=("acuracia", "mean"),
                tokens_medio=("tokens_total", "mean"),
                tempo_medio_ms=("tempo_ms", "mean"),
            )
            .reset_index()
        )

        print(f"\n  {'Técnica':<20} {'Acurácia':>10} {'Tokens Méd':>12} {'Tempo(ms)':>12}")
        print(f"  {'─'*20} {'─'*10} {'─'*12} {'─'*12}")
        for _, row in resumo.iterrows():
            nome = NOMES_AMIGAVEIS.get(row["tecnica"], row["tecnica"])
            print(
                f"  {nome:<20} {row['acuracia_media']:>9.0%} "
                f"{row['tokens_medio']:>12.0f} {row['tempo_medio_ms']:>12.0f}"
            )

        if tarefa in recomendacoes:
            rec = recomendacoes[tarefa]
            print(f"\n  ★ RECOMENDAÇÃO: {rec['nome_amigavel']}")
            print(f"  {rec['justificativa']}")

    print("\n" + "=" * 70)
    print("  GUIA DE BOLSO — Quando usar cada técnica neste domínio")
    print("=" * 70)
    guia = [
        ("Zero-Shot",        "Classificação simples de sentimento; quando velocidade importa"),
        ("Few-Shot",         "Extração de dados estruturados; quando formato é crítico"),
        ("Chain-of-Thought", "Sumarização complexa; quando raciocínio explícito melhora qualidade"),
        ("Role Prompting",   "Quando o tom/perspectiva profissional específica agrega valor"),
    ]
    for tecnica, quando in guia:
        print(f"  • {tecnica:<20}: {quando}")
    print()

