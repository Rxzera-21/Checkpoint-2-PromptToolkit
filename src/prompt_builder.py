"""
prompt_builder.py — Aula 05
Montagem de prompts por anatomia (instrução, contexto, input, formato).
"""


def montar_prompt(
    instrucao: str,
    contexto: str = "",
    input_dados: str = "",
    formato_output: str = "",
) -> str:
    """
    Monta um prompt estruturado com os 4 componentes da anatomia (Aula 05).

    Args:
        instrucao: O que o modelo deve fazer (obrigatório).
        contexto: Informação de background e domínio.
        input_dados: Dados de entrada sobre os quais a tarefa será executada.
        formato_output: Como a resposta deve ser formatada.

    Returns:
        Prompt montado como string.

    Raises:
        ValueError: Se instrução estiver vazia.
    """
    if not instrucao or not instrucao.strip():
        raise ValueError("A instrução é obrigatória e não pode estar vazia.")

    partes = []

    if contexto and contexto.strip():
        partes.append(f"[Contexto]: {contexto.strip()}")

    partes.append(f"[Instrucao]: {instrucao.strip()}")

    if input_dados and input_dados.strip():
        partes.append(f"---\n[Input]:\n{input_dados.strip()}\n---")

    if formato_output and formato_output.strip():
        partes.append(f"[Formato de resposta]: {formato_output.strip()}")

    return "\n\n".join(partes)


def adicionar_exemplos(prompt: str, exemplos: list[dict]) -> str:
    """
    Adiciona exemplos few-shot ao prompt (Aula 06).

    Args:
        prompt: Prompt base já montado.
        exemplos: Lista de dicts com chaves 'input' e 'output'.

    Returns:
        Prompt com exemplos inseridos antes do input real.
    """
    if not exemplos:
        return prompt

    linhas_exemplos = ["[Exemplos]:"]
    for ex in exemplos:
        inp = ex.get("input", "")
        out = ex.get("output", "")
        linhas_exemplos.append(f'Input: "{inp}" -> Output: "{out}"')

    bloco_exemplos = "\n".join(linhas_exemplos)

    # Inserir exemplos antes do bloco [Input] se existir
    if "[Input]:" in prompt:
        return prompt.replace("[Input]:", bloco_exemplos + "\n\n[Input]:", 1)
    else:
        return prompt + "\n\n" + bloco_exemplos


def adicionar_cot(prompt: str, passos: list[str]) -> str:
    """
    Adiciona instrução de Chain-of-Thought ao prompt (Aula 06).

    Args:
        prompt: Prompt base já montado.
        passos: Lista de passos de raciocínio a seguir.

    Returns:
        Prompt com instrução CoT adicionada.
    """
    if not passos:
        return prompt + "\n\nPense passo a passo antes de responder."

    numerados = "\n".join(f"{i+1}. {p}" for i, p in enumerate(passos))
    instrucao_cot = f"[Raciocínio - Pense passo a passo]:\n{numerados}"

    return prompt + "\n\n" + instrucao_cot


def substituir_input(template: str, valor: str) -> str:
    """Substitui o placeholder {input} em um template pelo valor real."""
    return template.replace("{input}", valor)

