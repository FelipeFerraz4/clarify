def f(x, y):
    print("Processando pedido para " + x['nome'] + "...")

    d = 0

    if (x['idade'] > 60 and x['status'] == 'fidelidade') or (y > 1000 and x['compras_recentes'] > 5):
        print("Aplicando desconto.")
        d = y * 0.15
    else:
        if y > 500:
            print("Aplicando desconto.")
            d = y * 0.05

    if (x['idade'] > 60 and x['status'] == 'fidelidade') or (y > 1000 and x['compras_recentes'] > 5):
        print("Desconto especial aplicado novamente.")
        d = y * 0.15

    z = y - (y * 0.15 if ((x['idade'] > 60 and x['status'] == 'fidelidade') or (y > 1000 and x['compras_recentes'] > 5)) else (y * 0.05 if y > 500 else 0))

    print("Preço final: R$" + str(y - d))

    return z
