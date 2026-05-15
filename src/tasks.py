"""
tasks.py — Aula 08
Definição das tarefas do domínio de e-commerce.
3 tarefas: classificação de sentimento, extração de dados e sumarização.
"""

# ──────────────────────────────────────────────
# TAREFA 1: Classificação de Sentimento
# ──────────────────────────────────────────────
CLASSIFICACAO_SENTIMENTO = {
    "nome": "classificacao_sentimento",
    "tipo": "classificacao",
    "instrucao": (
        "Classifique o sentimento da avaliacao de cliente de e-commerce "
        "como POSITIVO, NEGATIVO, NEUTRO ou MISTO."
    ),
    "formato_output": (
        "Responda APENAS com uma palavra: POSITIVO, NEGATIVO, NEUTRO ou MISTO."
    ),
    "exemplos_fewshot": [
        {"input": "Adorei o produto, chegou rapido e bem embalado!", "output": "POSITIVO"},
        {"input": "Pessimo! Veio quebrado e o suporte nao resolveu nada.", "output": "NEGATIVO"},
        {"input": "Produto bom, mas a entrega atrasou 10 dias.", "output": "MISTO"},
        {"input": "Produto comum, funciona como esperado.", "output": "NEUTRO"},
        {"input": "Superou todas as expectativas! Qualidade incrivel pelo preco.", "output": "POSITIVO"},
    ],
    "passos_cot": [
        "Identifique palavras e expressoes claramente positivas no texto",
        "Identifique palavras e expressoes claramente negativas no texto",
        "Avalie a proporcao entre aspectos positivos e negativos",
        "Se so ha positivos classifique como POSITIVO",
        "Se so ha negativos classifique como NEGATIVO",
        "Se ha ambos em proporcoes relevantes classifique como MISTO",
        "Se o texto e neutro/informativo classifique como NEUTRO",
        "Responda APENAS com a palavra da classificacao final",
    ],
    "persona": "analista_cx",
}

# ──────────────────────────────────────────────
# TAREFA 2: Extração de Dados do Produto
# ──────────────────────────────────────────────
EXTRACAO_DADOS = {
    "nome": "extracao_dados",
    "tipo": "extracao",
    "instrucao": (
        "Extraia as seguintes informacoes do texto de reclamacao de e-commerce "
        "e retorne em formato JSON valido: "
        "produto (nome completo), preco (valor ou null), defeito (descricao do problema)."
    ),
    "formato_output": (
        'Responda APENAS com JSON valido no formato: '
        '{"produto": "...", "preco": "...", "defeito": "..."}'
        ' Sem texto adicional antes ou depois do JSON.'
    ),
    "exemplos_fewshot": [
        {
            "input": "iPhone 14 Pro R$ 7.999,00 com camera nao funcionando.",
            "output": '{"produto": "iPhone 14 Pro", "preco": "R$ 7.999,00", "defeito": "camera nao funcionando"}',
        },
        {
            "input": "Smart TV Samsung 55 4K R$ 2.799,00 sem imagem no HDMI 2.",
            "output": '{"produto": "Smart TV Samsung 55 4K", "preco": "R$ 2.799,00", "defeito": "sem imagem no HDMI 2"}',
        },
        {
            "input": "Geladeira Brastemp Frost Free 375L R$ 3.199,00 nao gela corretamente.",
            "output": '{"produto": "Geladeira Brastemp Frost Free 375L", "preco": "R$ 3.199,00", "defeito": "nao gela corretamente"}',
        },
    ],
    "passos_cot": [
        "Leia o texto completo identificando todas as informacoes sobre o produto",
        "Extraia o nome completo do produto mencionado incluindo marca e modelo",
        "Procure por valores monetarios com R$ ou indicacao de preco",
        "Identifique a descricao do defeito ou problema relatado pelo cliente",
        "Monte o JSON com exatamente tres campos: produto, preco e defeito",
        "Se preco nao for mencionado use null",
        "Verifique se o JSON e valido (aspas, virgulas, chaves) antes de retornar",
        "Retorne APENAS o JSON sem texto adicional",
    ],
    "persona": "gerente_qualidade",
}

# ──────────────────────────────────────────────
# TAREFA 3: Sumarização de Reclamação
# ──────────────────────────────────────────────
SUMARIZACAO_RECLAMACAO = {
    "nome": "sumarizacao_reclamacao",
    "tipo": "sumarizacao",
    "instrucao": (
        "Resuma a reclamacao do cliente de e-commerce em no maximo 2 frases objetivas. "
        "Mantenha: o problema principal, acoes tomadas pelo cliente e o que ele solicita."
    ),
    "formato_output": (
        "Maximo 2 frases. Tom neutro e informativo. Sem julgamentos."
    ),
    "exemplos_fewshot": [
        {
            "input": (
                "Comprei um celular ha 2 semanas e a bateria ja nao dura nem 3 horas. "
                "Fui na assistencia e eles disseram que e normal para esse modelo, "
                "mas isso nao e verdade pois meu amigo tem o mesmo e dura o dia todo. "
                "Quero troca ou reembolso."
            ),
            "output": (
                "Bateria do celular dura menos de 3 horas em 2 semanas de uso. "
                "Assistencia alega ser normal; cliente solicita troca ou reembolso."
            ),
        },
        {
            "input": (
                "Fiz um pedido de urgencia para presente de casamento e escolhi entrega expressa "
                "pagando R$ 45 a mais. O produto chegou 5 dias depois do casamento. "
                "A loja nao quer devolver o valor do frete expresso."
            ),
            "output": (
                "Entrega expressa nao cumprida para presente de casamento. "
                "Loja recusa reembolso dos R$ 45 do frete expresso."
            ),
        },
    ],
    "passos_cot": [
        "Identifique o problema principal relatado pelo cliente",
        "Verifique quais acoes o cliente ja tomou (contato com suporte, assistencia, etc)",
        "Identifique o que o cliente esta solicitando (reembolso, troca, explicacao)",
        "Condense em no maximo 2 frases mantendo apenas as informacoes essenciais",
        "Use tom neutro e objetivo sem julgamentos sobre quem tem razao",
        "Elimine detalhes emocionais mantendo apenas os fatos relevantes",
    ],
    "persona": "analista_cx",
}

# Lista de todas as tarefas disponíveis
TAREFAS = [
    CLASSIFICACAO_SENTIMENTO,
    EXTRACAO_DADOS,
    SUMARIZACAO_RECLAMACAO,
]


def get_tarefa(nome: str) -> dict:
    """Retorna uma tarefa pelo nome."""
    for t in TAREFAS:
        if t["nome"] == nome:
            return t
    raise ValueError(f"Tarefa '{nome}' não encontrada. Disponíveis: {[t['nome'] for t in TAREFAS]}")

