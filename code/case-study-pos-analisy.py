DESCONTO_ESPECIAL = 0.15
DESCONTO_PADRAO = 0.05
IDADE_SENIOR = 60
VALOR_ALTO = 1000
VALOR_PADRAO = 500
MIN_COMPRAS_RECENTES = 5


def processar_pedido(usuario: dict, preco_produto: float) -> float:
    exibir_inicio_processamento(usuario["nome"])

    desconto = calcular_desconto(usuario, preco_produto)
    preco_final = aplicar_desconto(preco_produto, desconto)

    exibir_preco_final(preco_final)

    return preco_final


def calcular_desconto(usuario: dict, preco_produto: float) -> float:
    if eh_elegivel_desconto_especial(usuario, preco_produto):
        print("Aplicando desconto especial.")
        return preco_produto * DESCONTO_ESPECIAL

    if eh_elegivel_desconto_padrao(preco_produto):
        print("Aplicando desconto padrão.")
        return preco_produto * DESCONTO_PADRAO

    return 0.0


def eh_elegivel_desconto_especial(usuario: dict, preco_produto: float) -> bool:
    return eh_senior_fidelidade(usuario) or eh_compra_alto_valor(usuario, preco_produto)


def eh_senior_fidelidade(usuario: dict) -> bool:
    return (
        usuario["idade"] > IDADE_SENIOR and
        usuario["status"] == "fidelidade"
    )


def eh_compra_alto_valor(usuario: dict, preco_produto: float) -> bool:
    return (
        preco_produto > VALOR_ALTO and
        usuario["compras_recentes"] > MIN_COMPRAS_RECENTES
    )


def eh_elegivel_desconto_padrao(preco_produto: float) -> bool:
    return preco_produto > VALOR_PADRAO


def aplicar_desconto(preco_produto: float, desconto: float) -> float:
    return preco_produto - desconto


def exibir_inicio_processamento(nome_usuario: str) -> None:
    print(f"Processando pedido para {nome_usuario}...")


def exibir_preco_final(preco_final: float) -> None:
    print(f"Preço final: R${preco_final:.2f}")
