# Servidor LSP Clarify

Um servidor simples do Language Server Protocol (LSP) em Python construído com pygls que responde a notificações `textDocument/didOpen`.

## Funcionalidades

- Escuta notificações `textDocument/didOpen`
- Registra "Hello, world!" quando um documento é aberto
- Construído com pygls para facilitar o manuseio do protocolo LSP

## Configuração

1. Crie um ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

Execute o servidor LSP:
```bash
python clarify_lsp.py
```

Ou instale como pacote e execute:
```bash
pip install -e .
clarify-lsp
```

O servidor vai começar a ouvir por mensagens LSP no stdin/stdout.

## Estrutura do Projeto

```
clarify/
├── src/
│   └── clarify_lsp/
│       ├── __init__.py
│       └── server.py
├── clarify_lsp.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Desenvolvimento

Para testar o servidor com um cliente, você pode usar qualquer editor ou ferramenta compatível com LSP baseado em stdio.

## Referências

- https://pygls.readthedocs.io/en/stable/