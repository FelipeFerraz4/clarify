# Python Refactor VSCode Extension

Este plugin fornece uma ferramenta de **renomeação de variáveis** em arquivos Python diretamente no VSCode, utilizando um servidor LSP (Language Server Protocol) implementado em Python.

---

## Funcionalidades

- Renomeia todas as ocorrências de uma variável em um arquivo Python.
- Baseado em análise de **AST** (Abstract Syntax Tree), garantindo que apenas variáveis reais sejam alteradas (não strings ou comentários).
- Funciona em arquivos Python abertos no VSCode.
- Comando customizado no VSCode: `Python Refactor: Rename Variable`.

---

## Pré-requisitos

- VSCode instalado.
- Python 3.11+ instalado e disponível no PATH.
- Node.js instalado (para rodar a extensão).
- Dependências Python:
  ```bash
  pip install pygls lsprotocol
