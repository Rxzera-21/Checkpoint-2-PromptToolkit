"""
techniques.py — Aulas 06 + 07
Implementação das 4 técnicas de prompting: Zero-Shot, Few-Shot, CoT e Role Prompting.
"""

import json
import os
from src.prompt_builder import (
    montar_prompt,
    adicionar_exemplos,
    adicionar_cot,
    substituir_input,
)

# Carrega personas e templates uma única vez
_BASE = os.path.dirname(os.path.dirname(__file__))
_SYSTEM_PROMPTS_PATH = os.path.join(_BASE, "prompts", "system_prompts.json")
_TEMPLATES_PATH = os.path.join(_BASE, "prompts", "templates.json")

with open(_SYSTEM_PROMPTS_PATH, encoding="utf-8") as f:
    PERSONAS = json.load(f)

with open(_TEMPLATES_PATH, encoding="utf-8") as f:
    TEMPLATES = json.load(f)


# ──────────────────────────────────────────────
# 1. ZERO-SHOT (Aula 06)
# ──────────────────────────────────────────────

def zero_shot(tarefa: dict, input_texto: str) -> str:
    """
    Monta prompt direto usando prompt_builder. Sem exemplos.
    Instrução clara + formato de output definido.

    Args:
        tarefa: Dict com definição da tarefa (nome, instrucao, formato_output).
        input_texto: Texto de entrada a ser processado.

    Returns:
        Prompt montado como string.
    """
    nome_tarefa = tarefa["nome"]
    template_zs = TEMPLATES.get(nome_tarefa, {}).get("zero_shot", "")

    if template_zs:
        return substituir_input(template_zs, input_texto)

    # Fallback usando montar_prompt genérico
    return montar_prompt(
        instrucao=tarefa["instrucao"],
        contexto=f"Domínio: e-commerce brasileiro.",
        input_dados=input_texto,
        formato_output=tarefa.get("formato_output", "Responda de forma direta e concisa."),
    )


# ──────────────────────────────────────────────
# 2. FEW-SHOT (Aula 06)
# ──────────────────────────────────────────────

def few_shot(tarefa: dict, input_texto: str, exemplos: list[dict]) -> str:
    """
    Monta prompt com 2-3 exemplos do data/examples.json.
    Formato: Input: "..." -> Output: "..."

    Args:
        tarefa: Dict com definição da tarefa.
        input_texto: Texto de entrada a ser processado.
        exemplos: Lista de exemplos input→output para few-shot.

    Returns:
        Prompt montado como string.
    """
    # Prompt base
    prompt_base = montar_prompt(
        instrucao=tarefa["instrucao"],
        contexto="Domínio: e-commerce brasileiro. Siga o padrão dos exemplos abaixo.",
        input_dados=input_texto,
        formato_output=tarefa.get("formato_output", ""),
    )

    # Limitar a 3 exemplos (sweet spot conforme Aula 06)
    exemplos_uso = exemplos[: min(3, len(exemplos))]

    return adicionar_exemplos(prompt_base, exemplos_uso)


# ──────────────────────────────────────────────
# 3. CHAIN-OF-THOUGHT (Aula 06)
# ──────────────────────────────────────────────

def chain_of_thought(tarefa: dict, input_texto: str, passos: list[str] = None) -> str:
    """
    Monta prompt com instrução de raciocínio explícito passo a passo.

    Args:
        tarefa: Dict com definição da tarefa.
        input_texto: Texto de entrada a ser processado.
        passos: Lista de passos de raciocínio. Se None, usa os da tarefa.

    Returns:
        Prompt montado como string.
    """
    passos_uso = passos or tarefa.get("passos_cot", [])

    prompt_base = montar_prompt(
        instrucao=tarefa["instrucao"],
        contexto="Domínio: e-commerce brasileiro. Analise com cuidado antes de responder.",
        input_dados=input_texto,
        formato_output=tarefa.get("formato_output", ""),
    )

    return adicionar_cot(prompt_base, passos_uso)


# ──────────────────────────────────────────────
# 4. ROLE PROMPTING (Aula 07)
# ──────────────────────────────────────────────

def role_prompting(tarefa: dict, input_texto: str, persona: str) -> tuple[str, str]:
    """
    Usa system prompt com persona detalhada do prompts/system_prompts.json.
    Retorna tupla (system_prompt, user_prompt).

    Args:
        tarefa: Dict com definição da tarefa.
        input_texto: Texto de entrada a ser processado.
        persona: Chave da persona em system_prompts.json.

    Returns:
        Tupla (system, user) onde system é o prompt de sistema com a persona
        e user é o prompt do usuário com a tarefa.
    """
    if persona not in PERSONAS:
        raise ValueError(
            f"Persona '{persona}' não encontrada. Disponíveis: {list(PERSONAS.keys())}"
        )

    system_prompt = PERSONAS[persona]["sistema"]

    user_prompt = montar_prompt(
        instrucao=tarefa["instrucao"],
        input_dados=input_texto,
        formato_output=tarefa.get("formato_output", ""),
    )

    return system_prompt, user_prompt

