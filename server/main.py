from pygls.server import LanguageServer
from lsprotocol.types import (
    Position,
    Range,
    TextEdit,
    WorkspaceEdit,
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    TEXT_DOCUMENT_RENAME,
    TEXT_DOCUMENT_CODE_ACTION,
)
import ast
import traceback

class Renamer(ast.NodeTransformer):
    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name(self, node):
        if node.id == self.old_name:
            node.id = self.new_name
        return self.generic_visit(node)

class RefactorServer(LanguageServer):
    pass

server = RefactorServer("refactor-server", "v0.6")

@server.feature(TEXT_DOCUMENT_RENAME)
def rename_variable(ls: RefactorServer, params):
    try:
        doc = ls.workspace.get_document(params.text_document.uri)
        code = doc.source
        lines = code.splitlines()

        line = params.position.line
        char = params.position.character
        old_name = None

        for word in lines[line].split():
            start_char = lines[line].find(word)
            end_char = start_char + len(word)
            if start_char <= char < end_char:
                old_name = word
                break

        if not old_name:
            ls.show_message("No variable found at cursor")
            return None

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

        #ls.show_message(f"[INFO] Variable '{old_name}' renamed to '{params.new_name}'")
        return WorkspaceEdit(changes={params.text_document.uri: [edit]})

    except Exception as e:
        ls.show_message_log(f"[ERROR] Rename failed: {e}\n{traceback.format_exc()}")
        return None


def analyse_variables(code_snippet, full_code, start_line, end_line):
    try:
        snippet_tree = ast.parse(code_snippet)

        assigned_in_snippet = {
            node.targets[0].id
            for node in ast.walk(snippet_tree)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }

        used_in_snippet = {
            node.id for node in ast.walk(snippet_tree) if isinstance(node, ast.Name)
        }

        inputs = used_in_snippet - assigned_in_snippet

        lines_after = "\n".join(full_code.splitlines()[end_line + 1:])
        tree_after = ast.parse(lines_after) if lines_after.strip() else ast.parse("")
        used_after = {node.id for node in ast.walk(tree_after) if isinstance(node, ast.Name)}

        outputs = assigned_in_snippet & used_after

        return list(inputs), list(outputs)

    except Exception as e:
        print(f"[ERROR] analyse_variables failed: {e}")
        print(traceback.format_exc())
        return [], []


@server.feature(TEXT_DOCUMENT_CODE_ACTION)
def extract_function(ls: RefactorServer, params: CodeActionParams):
    try:
        doc = ls.workspace.get_document(params.text_document.uri)
        code = doc.source
        lines = code.splitlines()

        sel_range: Range = params.range
        start_line, end_line = sel_range.start.line, sel_range.end.line

        extracted_code = "\n".join(lines[start_line:end_line + 1]).strip()
        if not extracted_code:
            print("[INFO] Selected range too small or empty for extraction")
            return None

        inputs, outputs = analyse_variables(extracted_code, code, start_line, end_line)
        input_str = ", ".join(inputs)
        func_name = "extracted_function"

        if outputs:
            out_str = ", ".join(outputs)
            call_line = f"{out_str} = {func_name}({input_str})" if input_str else f"{out_str} = {func_name}()"
        else:
            call_line = f"{func_name}({input_str})" if input_str else f"{func_name}()"

        selection_edit = TextEdit(
            range=Range(
                start=Position(line=start_line, character=0),
                end=Position(line=end_line, character=len(lines[end_line]))
            ),
            new_text=call_line
        )

        function_text = f"\n\ndef {func_name}({input_str}):\n"
        for l in extracted_code.splitlines():
            function_text += "    " + l + "\n"
        if outputs:
            out_str = ", ".join(outputs)
            function_text += f"    return {out_str}\n"

        function_edit = TextEdit(
            range=Range(
                start=Position(line=len(lines), character=0),
                end=Position(line=len(lines), character=0)
            ),
            new_text=function_text
        )

        return [
            CodeAction(
                title="Extract Function",
                kind=CodeActionKind.RefactorExtract,
                edit=WorkspaceEdit(changes={params.text_document.uri: [selection_edit, function_edit]}),
                data={"ask_for_name": True}
            )
        ]

    except Exception as e:
        ls.show_message_log(f"[ERROR] Extract Function failed: {e}\n{traceback.format_exc()}")
        return None


if __name__ == "__main__":
    import sys
    print("[INFO] Starting Python Refactor Server...")
    try:
        server.start_io(sys.stdin.buffer, sys.stdout.buffer)
    except Exception as e:
        print(f"[ERROR] Server failed to start: {e}")
        import traceback
        print(traceback.format_exc())
    print("[INFO] Server stopped.")
