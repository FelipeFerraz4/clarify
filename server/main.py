import ast
import traceback
import logging
import os

from lsprotocol.types import (TEXT_DOCUMENT_CODE_ACTION, TEXT_DOCUMENT_RENAME,
                              CodeAction, CodeActionKind, CodeActionParams,
                              Position, Range, TextEdit, WorkspaceEdit)
from pygls.server import LanguageServer


# Configure file-based logging
_LOG_DIR = os.path.dirname(__file__)
_LOG_FILE = os.path.join(_LOG_DIR, "refactor.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_FILE, encoding="utf-8")],
)

logger = logging.getLogger("refactor-server")

class Renamer(ast.NodeTransformer):
    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name

    def visit_Name(self, node):
        if node.id == self.old_name:
            node.id = self.new_name
        return self.generic_visit(node)
    
    def visit_Attribute(self, node):
        if node.attr == self.old_name:
            node.attr = self.new_name
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
        logger.exception("analyse_variables failed")
        return [], []


def find_statement_line(tree, target_line):
    """
    Find the line number where the statement containing target_line starts.
    
    Args:
        tree: AST tree
        target_line: Line number (1-indexed, AST format)
    
    Returns:
        int: Line number (1-indexed) where the containing statement starts
    """
    containing_stmt = None
    
    for node in ast.walk(tree):
        # Check for statement nodes
        if isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
                            ast.Return, ast.If, ast.While, ast.For, ast.With,
                            ast.FunctionDef, ast.ClassDef, ast.Try)):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                if node.lineno <= target_line <= node.end_lineno:
                    # Found a containing statement - keep the smallest one
                    if containing_stmt is None or node.lineno > containing_stmt.lineno:
                        containing_stmt = node
    
    return containing_stmt.lineno if containing_stmt else target_line


def get_line_indentation(line_text):
    """Get the indentation (spaces) at the start of a line."""
    return len(line_text) - len(line_text.lstrip())


def find_function_def(tree, func_name):
    """
    Find a function definition by name in the AST.
    
    Args:
        tree: AST tree
        func_name: Name of the function to find
    
    Returns:
        ast.FunctionDef node or None
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None


def inline_function_body(func_def, call_node, code_lines):
    """
    Inline a function body by substituting parameters with arguments.
    
    Args:
        func_def: ast.FunctionDef node
        call_node: ast.Call node
        code_lines: List of source code lines
    
    Returns:
        str: Inlined code or None if cannot inline
    """
    try:
        # Build parameter to argument mapping
        param_map = {}
        
        # Handle positional arguments
        for i, arg in enumerate(call_node.args):
            if i < len(func_def.args.args):
                param_name = func_def.args.args[i].arg
                # Get the source code of the argument
                arg_code = ast.unparse(arg)
                param_map[param_name] = arg_code
        
        # Handle keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg:
                param_map[keyword.arg] = ast.unparse(keyword.value)
        
        # Get the function body
        if not func_def.body:
            return None
        
        # Handle simple single-return functions
        if len(func_def.body) == 1 and isinstance(func_def.body[0], ast.Return):
            # Simple case: just a return statement
            return_node = func_def.body[0]
            if return_node.value:
                # Clone and transform the return expression
                transformed = _substitute_names(return_node.value, param_map)
                return ast.unparse(transformed)
            return None
        
        # Handle multi-statement functions (more complex)
        # For now, we'll try to inline simple cases
        inlined_statements = []
        
        for stmt in func_def.body:
            transformed_stmt = _substitute_names(stmt, param_map)
            inlined_statements.append(transformed_stmt)
        
        # Check if the last statement is a return
        if inlined_statements and isinstance(inlined_statements[-1], ast.Return):
            # If it's just a return statement at the end
            if len(inlined_statements) == 1:
                return_node = inlined_statements[-1]
                if return_node.value:
                    return ast.unparse(return_node.value)
            else:
                # Multi-statement function - harder to inline safely
                # We'll skip this for now
                logger.info("Cannot inline multi-statement function")
                return None
        
        return None
    
    except Exception as e:
        logger.exception(f"Failed to inline function body: {e}")
        return None


def _substitute_names(node, param_map):
    """
    Recursively substitute parameter names with argument expressions.
    
    Args:
        node: AST node
        param_map: Dict mapping parameter names to argument expressions
    
    Returns:
        Transformed AST node
    """
    if isinstance(node, ast.Name) and node.id in param_map:
        # Parse the replacement expression and return its AST
        try:
            replacement = ast.parse(param_map[node.id], mode='eval').body
            return ast.copy_location(replacement, node)
        except:
            return node
    
    # Recursively process child nodes
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, ast.AST):
                    new_list.append(_substitute_names(item, param_map))
                else:
                    new_list.append(item)
            setattr(node, field, new_list)
        elif isinstance(value, ast.AST):
            setattr(node, field, _substitute_names(value, param_map))
    
    return node


def create_inline_function_action(doc, params: CodeActionParams):
    """
    Create a code action for inlining a function call.
    
    Args:
        doc: Document object
        params: CodeActionParams from LSP
    
    Returns:
        CodeAction or None
    """
    try:
        code = doc.source
        lines = code.splitlines()
        
        # Get cursor position
        cursor_line = params.range.start.line
        cursor_char = params.range.start.character
        
        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        
        # Find the Call node at cursor position
        finder = NodeFinder(cursor_line, cursor_char)
        finder.visit(tree)
        
        # Look for a Call node in candidates
        call_node = None
        for node in finder.candidates:
            if isinstance(node, ast.Call):
                call_node = node
                break
        
        if not call_node:
            return None
        
        # Get the function name being called
        func_name = None
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            # Method call - skip for now
            return None
        else:
            return None
        
        # Find the function definition
        func_def = find_function_def(tree, func_name)
        if not func_def:
            logger.info(f"Function definition for '{func_name}' not found")
            return None
        
        # Inline the function
        inlined_code = inline_function_body(func_def, call_node, lines)
        if not inlined_code:
            logger.info(f"Cannot inline function '{func_name}'")
            return None
        
        # Create the edit to replace the function call
        call_start_line = call_node.lineno - 1  # Convert to 0-indexed
        call_end_line = call_node.end_lineno - 1
        call_start_char = call_node.col_offset
        call_end_char = call_node.end_col_offset
        
        replace_edit = TextEdit(
            range=Range(
                start=Position(line=call_start_line, character=call_start_char),
                end=Position(line=call_end_line, character=call_end_char),
            ),
            new_text=inlined_code,
        )
        
        return CodeAction(
            title=f"Inline function '{func_name}'",
            kind=CodeActionKind.RefactorInline,
            edit=WorkspaceEdit(
                changes={params.text_document.uri: [replace_edit]}
            ),
        )
    
    except Exception as e:
        logger.exception(f"Inline Function failed: {e}")
        return None


def create_extract_variable_action(doc, params: CodeActionParams):
    """
    Create a code action for extracting a variable from selected code.
    
    Args:
        doc: Document object
        params: CodeActionParams from LSP
    
    Returns:
        CodeAction or None
    """
    try:
        code = doc.source
        lines = code.splitlines()
        
        sel_range: Range = params.range
        start_line = sel_range.start.line
        end_line = sel_range.end.line
        start_char = sel_range.start.character
        end_char = sel_range.end.character
        
        # Extract the selected text
        if start_line == end_line:
            # Single line selection
            selected_text = lines[start_line][start_char:end_char].strip()
        else:
            # Multi-line selection
            first_line = lines[start_line][start_char:]
            last_line = lines[end_line][:end_char]
            middle_lines = lines[start_line + 1:end_line]
            selected_text = "\n".join([first_line] + middle_lines + [last_line]).strip()
        
        if not selected_text:
            return None
        
        # Validate it's an expression (try to parse it)
        try:
            ast.parse(selected_text, mode='eval')
        except SyntaxError:
            # Not a valid expression
            logger.info(f"Selected text is not a valid expression: {selected_text}")
            return None
        
        # Parse the full code to find the containing statement
        try:
            tree = ast.parse(code)
        except SyntaxError:
            logger.info("Could not parse document for extract variable")
            return None
        
        # Find the statement containing this expression
        # Convert to 1-indexed for AST
        statement_line = find_statement_line(tree, start_line + 1)
        # Convert back to 0-indexed for LSP
        statement_line_idx = statement_line - 1
        
        # Get indentation of the containing statement
        statement_text = lines[statement_line_idx]
        indent = " " * get_line_indentation(statement_text)
        
        # Generate variable name
        var_name = "extracted_var"
        
        # Create the variable assignment
        assignment_line = f"{indent}{var_name} = {selected_text}\n"
        
        # Create edits:
        # 1. Insert variable assignment before the statement
        insert_edit = TextEdit(
            range=Range(
                start=Position(line=statement_line_idx, character=0),
                end=Position(line=statement_line_idx, character=0),
            ),
            new_text=assignment_line,
        )
        
        # 2. Replace selected expression with variable name
        replace_edit = TextEdit(
            range=Range(
                start=Position(line=start_line, character=start_char),
                end=Position(line=end_line, character=end_char),
            ),
            new_text=var_name,
        )
        
        return CodeAction(
            title="Extract Variable",
            kind=CodeActionKind.RefactorExtract,
            edit=WorkspaceEdit(
                changes={params.text_document.uri: [insert_edit, replace_edit]}
            ),
            data={"ask_for_name": True},
        )
    
    except Exception as e:
        logger.exception(f"Extract Variable failed: {e}")
        return None


@server.feature(TEXT_DOCUMENT_CODE_ACTION)
def code_actions(ls: RefactorServer, params: CodeActionParams):
    """
    Provide code actions for refactoring.
    Returns Extract Function, Extract Variable, and Inline Function actions.
    """
    actions = []
    
    try:
        doc = ls.workspace.get_document(params.text_document.uri)
        
        # Try to create Extract Variable action
        extract_var_action = create_extract_variable_action(doc, params)
        if extract_var_action:
            actions.append(extract_var_action)
        
        # Try to create Extract Function action
        extract_func_action = create_extract_function_action(doc, params)
        if extract_func_action:
            actions.append(extract_func_action)
        
        # Try to create Inline Function action
        inline_func_action = create_inline_function_action(doc, params)
        if inline_func_action:
            actions.append(inline_func_action)
        
        return actions if actions else None
    
    except Exception as e:
        ls.show_message_log(
            f"[ERROR] Code actions failed: {e}\n{traceback.format_exc()}"
        )
        logger.exception("Code actions failed")
        return None


def create_extract_function_action(doc, params: CodeActionParams):
    """
    Create a code action for extracting a function from selected code.
    
    Args:
        doc: Document object
        params: CodeActionParams from LSP
    
    Returns:
        CodeAction or None
    """
    try:
        code = doc.source
        lines = code.splitlines()

        sel_range: Range = params.range
        start_line, end_line = sel_range.start.line, sel_range.end.line

        extracted_code = "\n".join(lines[start_line : end_line + 1]).strip()
        if not extracted_code:
            logger.info("Selected range too small or empty for extraction")
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

        return CodeAction(
            title="Extract Function",
            kind=CodeActionKind.RefactorExtract,
            edit=WorkspaceEdit(
                changes={params.text_document.uri: [selection_edit, function_edit]}
            ),
            data={"ask_for_name": True},
        )

    except Exception as e:
        logger.exception(f"Extract Function failed: {e}")
        return None


if __name__ == "__main__":
    import sys

    logger.info("Starting Python Refactor Server...")
    try:
        server.start_io(sys.stdin.buffer, sys.stdout.buffer)
    except Exception as e:
        logger.exception("Server failed to start")
    logger.info("Server stopped.")
