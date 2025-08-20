import ast
import sys

def list_functions_with_docstring(filepath):
    """
    Print the name and first line of the docstring for each function in a Python file.

    Args:
        filepath (str): Path to the Python file to analyze.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        # Parse the Python file into an AST (Abstract Syntax Tree)
        tree = ast.parse(f.read(), filename=filepath)
    # Walk through all nodes in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get the docstring for the function, if present
            doc = ast.get_docstring(node)
            # Extract the first line of the docstring, or empty string if none
            first_line = doc.splitlines()[0] if doc else ""
            print(f"{node.name}: {first_line}")

if __name__ == "__main__":
    # Check for correct usage
    if len(sys.argv) != 2:
        print("Usage: python list_functions_with_doc.py <python_file.py>")
    else:
        # Call the function with the provided file path
        list_functions_with_docstring(sys.argv[1])
