import ast
import traceback

from lsprotocol.types import (
    TEXT_DOCUMENT_CODE_ACTION,
    TEXT_DOCUMENT_RENAME,
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Position,
    Range,
    TextEdit,
    WorkspaceEdit,
)
from pygls.server import LanguageServer


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


class NodeFinder(ast.NodeVisitor):
    """Finds AST nodes at a specific cursor position."""

    def __init__(self, target_line, target_col):
        # AST uses 1-indexed lines
        self.target_line = target_line + 1
        self.target_col = target_col
        self.candidates = []

    def visit(self, node):
        if self._contains_position(node):
            self.candidates.append(node)
        self.generic_visit(node)

    def _contains_position(self, node):
        if not hasattr(node, "lineno") or not hasattr(node, "col_offset"):
            return False

        # Python 3.8+
        if not hasattr(node, "end_lineno") or not hasattr(node, "end_col_offset"):
            return False

        if not (node.lineno <= self.target_line <= node.end_lineno):
            return False

        if node.lineno == node.end_lineno == self.target_line:
            return node.col_offset <= self.target_col < node.end_col_offset

        # Multi-line node - check start line
        if self.target_line == node.lineno:
            return self.target_col >= node.col_offset

        # Multi-line node - check end line
        if self.target_line == node.end_lineno:
            return self.target_col < node.end_col_offset

        # Middle of multi-line node
        return True

    def get_most_specific_node(self):
        """Return the smallest (most specific) node containing the cursor."""
        if not self.candidates:
            return None

        # Sort by size (smaller = more specific)
        def node_size(node):
            line_span = node.end_lineno - node.lineno
            if line_span == 0:
                # Single line - use column span
                return node.end_col_offset - node.col_offset
            else:
                # Multi-line - prioritize by line count, then column
                return line_span * 10000 + (node.end_col_offset - node.col_offset)

        return min(self.candidates, key=node_size)


def find_identifier_at_position(code, line, character):
    """
    Find the identifier at the given cursor position.

    Returns:
        tuple: (identifier_name, node) or (None, None) if not found
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None

    finder = NodeFinder(line, character)
    finder.visit(tree)
    node = finder.get_most_specific_node()

    if node is None:
        return None, None

    identifier = extract_identifier_from_node(node, line, character)

    return identifier, node


def extract_identifier_from_node(node, line, character):
    """
    Extract the identifier name from a node based on its type.

    Args:
        node: AST node at cursor position
        line: Cursor line (0-indexed, LSP format)
        character: Cursor character (0-indexed)

    Returns:
        str: Identifier name or None
    """
    if isinstance(node, ast.Name):
        # Simple variable: x, y, foo
        return node.id

    elif isinstance(node, ast.Attribute):
        # Attribute access: obj.method, a.b.c
        # We need to determine if cursor is on the attribute or the value

        # Convert line to AST format (1-indexed)
        ast_line = line + 1

        # If cursor is on the same line as the attribute
        if ast_line == node.end_lineno:
            # The attribute name starts after the dot
            # Approximate: if cursor is near the end, it's on the attribute
            attr_start_col = node.end_col_offset - len(node.attr)

            if character >= attr_start_col:
                # Cursor is on the attribute name
                return node.attr

        # Otherwise, user might be on the value part (handled by finding a more specific node)
        return None

    elif isinstance(node, ast.FunctionDef):
        # Function definition: def foo():
        # Only rename if cursor is on the name part (first line)
        if line + 1 == node.lineno:
            return node.name
        return None

    elif isinstance(node, ast.ClassDef):
        # Class definition: class MyClass:
        if line + 1 == node.lineno:
            return node.name
        return None

    elif isinstance(node, ast.arg):
        # Function parameter: def func(param):
        return node.arg

    elif isinstance(node, ast.keyword):
        # Keyword argument: func(param=value)
        # Check if cursor is on the keyword name (before the =)
        if node.arg:  # keyword arguments have an 'arg' attribute
            return node.arg
        return None

    elif isinstance(node, ast.ExceptHandler):
        # Exception handler: except Exception as e:
        if node.name and line + 1 == node.lineno:
            return node.name
        return None

    # Unsupported node type
    return None


# Replace the variable detection part in your rename_variable function:
def get_identifier_to_rename(ls, params):
    """
    Get the identifier at the cursor position for renaming.

    Returns:
        str: Identifier name or None if not found/invalid
    """
    doc = ls.workspace.get_document(params.text_document.uri)
    code = doc.source

    line = params.position.line
    char = params.position.character

    identifier, node = find_identifier_at_position(code, line, char)

    if not identifier:
        ls.show_message("No renameable identifier found at cursor position")
        return None

    # Additional validation: check if it's a valid Python identifier
    if not identifier.isidentifier():
        ls.show_message(f"'{identifier}' is not a valid identifier")
        return None

    # Don't allow renaming of Python keywords
    import keyword

    if keyword.iskeyword(identifier):
        ls.show_message(f"Cannot rename keyword '{identifier}'")
        return None

    return identifier


# Updated rename_variable function (replace the old one):
@server.feature(TEXT_DOCUMENT_RENAME)
def rename_variable(ls: RefactorServer, params):
    try:
        # Use the new AST-based detection
        old_name = get_identifier_to_rename(ls, params)
        if not old_name:
            return None

        # Get document and code
        doc = ls.workspace.get_document(params.text_document.uri)
        code = doc.source
        lines = code.splitlines()

        # Parse and transform AST
        tree = ast.parse(code)
        renamer = Renamer(old_name, params.new_name)
        tree = renamer.visit(tree)
        ast.fix_missing_locations(tree)
        new_code = ast.unparse(tree)

        # Create edit (still using full replacement for now)
        edit = TextEdit(
            range=Range(
                start=Position(line=0, character=0),
                end=Position(line=len(lines) - 1, character=len(lines[-1])),
            ),
            new_text=new_code,
        )

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

        lines_after = "\n".join(full_code.splitlines()[end_line + 1 :])
        tree_after = ast.parse(lines_after) if lines_after.strip() else ast.parse("")
        used_after = {
            node.id for node in ast.walk(tree_after) if isinstance(node, ast.Name)
        }

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

        extracted_code = "\n".join(lines[start_line : end_line + 1]).strip()
        if not extracted_code:
            print("[INFO] Selected range too small or empty for extraction")
            return None

        inputs, outputs = analyse_variables(extracted_code, code, start_line, end_line)
        input_str = ", ".join(inputs)
        func_name = "extracted_function"

        if outputs:
            out_str = ", ".join(outputs)
            call_line = (
                f"{out_str} = {func_name}({input_str})"
                if input_str
                else f"{out_str} = {func_name}()"
            )
        else:
            call_line = f"{func_name}({input_str})" if input_str else f"{func_name}()"

        selection_edit = TextEdit(
            range=Range(
                start=Position(line=start_line, character=0),
                end=Position(line=end_line, character=len(lines[end_line])),
            ),
            new_text=call_line,
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
                end=Position(line=len(lines), character=0),
            ),
            new_text=function_text,
        )

        return [
            CodeAction(
                title="Extract Function",
                kind=CodeActionKind.RefactorExtract,
                edit=WorkspaceEdit(
                    changes={params.text_document.uri: [selection_edit, function_edit]}
                ),
                data={"ask_for_name": True},
            )
        ]

    except Exception as e:
        ls.show_message_log(
            f"[ERROR] Extract Function failed: {e}\n{traceback.format_exc()}"
        )
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
