from pygls.server import LanguageServer
from lsprotocol.types import (
    Position,
    Range,
    TextEdit,
    WorkspaceEdit,
    TEXT_DOCUMENT_RENAME,
)
import ast

# Classe que substitui nomes no AST
class Renamer(ast.NodeTransformer):
    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name(self, node):
        # Substitui todas as ocorrências
        if node.id == self.old_name:
            node.id = self.new_name
        return self.generic_visit(node)  # garante visitar filhos

# Servidor LSP
class RefactorServer(LanguageServer):
    pass

server = RefactorServer("refactor-server", "v0.2")

# Função de rename
@server.feature(TEXT_DOCUMENT_RENAME)
def rename_variable(ls: RefactorServer, params):
    doc = ls.workspace.get_document(params.text_document.uri)
    code = doc.source
    lines = code.splitlines()

    line = params.position.line
    char = params.position.character
    old_name = None

    # Busca a palavra sob o cursor
    for word in lines[line].split():
        start_char = lines[line].find(word)
        end_char = start_char + len(word)
        if start_char <= char < end_char:
            old_name = word
            break

    if not old_name:
        ls.show_message("No variable found at cursor")
        return None

    try:
        tree = ast.parse(code)
        renamer = Renamer(old_name, params.new_name)
        tree = renamer.visit(tree)
        ast.fix_missing_locations(tree)
        new_code = ast.unparse(tree)

        edit = TextEdit(
            range=Range(
                start=Position(line=0, character=0),
                end=Position(line=len(lines)-1, character=len(lines[-1]))
            ),
            new_text=new_code
        )

        return WorkspaceEdit(changes={params.text_document.uri: [edit]})

    except Exception as e:
        ls.show_message_log(f"Rename failed: {e}")
        return None

if __name__ == "__main__":
    server.start_io()