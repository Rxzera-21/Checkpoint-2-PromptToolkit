"""
evaluator.py — Aula 09
Métricas: acurácia, consistência, tokens, temperatura.
"""

import json
import re
import time
import tiktoken


# Encoder tiktoken (cl100k_base é compatível com GPT-4 e modelos modernos)
try:
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None


def contar_tokens(texto: str) -> int:
    """
    Conta tokens de um texto usando tiktoken.

    Args:
        texto: String a ser tokenizada.

    Returns:
        Número de tokens.
    """
    if _ENCODER is None:
        # Fallback: estimativa por palavras
        return len(texto.split())
    return len(_ENCODER.encode(texto))


def medir_acuracia(resposta: str, esperado) -> float:
    """
    Mede acurácia comparando resposta com valor esperado.
    Suporta comparação de strings simples e dicts JSON.

    Args:
        resposta: String retornada pelo LLM.
        esperado: String ou dict com o valor esperado.

    Returns:
        Float entre 0.0 e 1.0 (0 = errado, 1 = correto).
    """
    if not resposta or resposta.startswith("[ERRO"):
        return 0.0

    resposta_limpa = resposta.strip().upper()

    # Comparação com dict (extração de dados)
    if isinstance(esperado, dict):
        return _acuracia_json(resposta, esperado)

    # Comparação de string simples (classificação)
    esperado_upper = str(esperado).strip().upper()
    if esperado_upper in resposta_limpa:
        return 1.0

    return 0.0


def _acuracia_json(resposta: str, esperado: dict) -> float:
    """Mede acurácia para respostas JSON extraindo e comparando campos."""
    try:
        # Tentar extrair JSON do texto (pode ter texto ao redor)
        match = re.search(r"\{.*?\}", resposta, re.DOTALL)
        if not match:
            return 0.0

        resp_dict = json.loads(match.group())

        acertos = 0
        total = len(esperado)

        for campo, valor_esperado in esperado.items():
            valor_resp = str(resp_dict.get(campo, "")).strip().lower()
            valor_esp = str(valor_esperado).strip().lower()

            # Match parcial: se valor esperado está contido na resposta ou vice-versa
            if valor_esp in valor_resp or valor_resp in valor_esp:
                acertos += 1
            elif _similaridade_simples(valor_resp, valor_esp) > 0.6:
                acertos += 0.5

        return round(acertos / total, 2)

    except (json.JSONDecodeError, Exception):
        return 0.0


def _similaridade_simples(a: str, b: str) -> float:
    """Similaridade simples por palavras em comum."""
    palavras_a = set(a.lower().split())
    palavras_b = set(b.lower().split())
    if not palavras_a or not palavras_b:
        return 0.0
    intersecao = palavras_a & palavras_b
    return len(intersecao) / max(len(palavras_a), len(palavras_b))


def medir_consistencia(respostas: list[str]) -> float:
    """
    Mede consistência como porcentagem de respostas iguais à moda.

    Args:
        respostas: Lista de respostas do mesmo prompt executado N vezes.

    Returns:
        Float entre 0.0 e 1.0 (proporção de respostas iguais à mais comum).
    """
    if not respostas:
        return 0.0
    if len(respostas) == 1:
        return 1.0

    # Normaliza respostas para comparação
    normalizadas = [r.strip().lower()[:100] for r in respostas]

    # Conta frequência de cada resposta
    contagem = {}
    for r in normalizadas:
        contagem[r] = contagem.get(r, 0) + 1

    mais_comum = max(contagem.values())
    return round(mais_comum / len(respostas), 2)


def testar_temperatura(llm_client, prompt: str, system: str = "", temps: list = None) -> list[dict]:
    """
    Roda o mesmo prompt com temperaturas diferentes e mede consistência.

    Args:
        llm_client: Instância de LLMClient.
        prompt: Prompt a ser testado.
        system: System prompt (opcional).
        temps: Lista de temperaturas a testar (padrão: [0.1, 0.5, 1.0]).

    Returns:
        Lista de dicts com resultados por temperatura.
    """
    if temps is None:
        temps = [0.1, 0.5, 1.0]

    resultados = []
    N_RUNS = 3  # Número de execuções por temperatura

    for temp in temps:
        respostas = []
        tokens_totais = []
        tempos = []

        for _ in range(N_RUNS):
            resultado = llm_client.chat(
                prompt=prompt,
                system=system,
                temp=temp,
                max_tokens=256,
            )
            respostas.append(resultado["resposta"])
            tokens_totais.append(resultado["tokens_total"])
            tempos.append(resultado["tempo_ms"])
            time.sleep(0.3)

        consistencia = medir_consistencia(respostas)

        resultados.append({
            "temperatura": temp,
            "respostas": respostas,
            "consistencia": consistencia,
            "tokens_medio": round(sum(tokens_totais) / len(tokens_totais), 1) if tokens_totais else 0,
            "tempo_medio_ms": round(sum(tempos) / len(tempos), 1) if tempos else 0,
        })

    return resultados


def calcular_metricas_run(resultado_llm: dict, esperado, prompt_texto: str) -> dict:
    """
    Calcula todas as métricas para uma execução.

    Args:
        resultado_llm: Dict retornado pelo LLMClient.chat().
        esperado: Valor esperado para cálculo de acurácia.
        prompt_texto: Texto do prompt para contar tokens via tiktoken.

    Returns:
        Dict com todas as métricas calculadas.
    """
    resposta = resultado_llm.get("resposta", "")
    tokens_prompt_tiktoken = contar_tokens(prompt_texto)
    tokens_resposta_tiktoken = contar_tokens(resposta)
    acuracia = medir_acuracia(resposta, esperado)

    return {
        "resposta": resposta,
        "acuracia": acuracia,
        "tokens_prompt": resultado_llm.get("tokens_prompt") or tokens_prompt_tiktoken,
        "tokens_resposta": resultado_llm.get("tokens_resposta") or tokens_resposta_tiktoken,
        "tokens_total": resultado_llm.get("tokens_total") or (tokens_prompt_tiktoken + tokens_resposta_tiktoken),
        "tempo_ms": resultado_llm.get("tempo_ms", 0),
    }

