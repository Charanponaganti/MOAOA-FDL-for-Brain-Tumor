import ast
import nbformat as nbf

def clean_notebook(py_file, output_file):
    with open(py_file, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    nb = nbf.v4.new_notebook()
    cells = []

    def add_md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def add_code(code):
        cells.append(nbf.v4.new_code_cell(code.strip()))

    # 🔹 Title
    add_md("# LSTM + MOAOA Optimization\nClean structured notebook")

    # 🔹 Imports
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(source, node))
    add_md("## 📦 Imports")
    add_code("\n".join(imports))

    # 🔹 Config
    assigns = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assigns.append(ast.get_source_segment(source, node))
    add_md("## ⚙️ Configuration")
    add_code("\n".join(assigns) + "\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")

    # 🔹 Classes
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            add_md(f"## 🧠 Class: {node.name}")
            add_code(ast.get_source_segment(source, node))

    # 🔹 Functions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            add_md(f"## ⚡ Function: {node.name}")
            add_code(ast.get_source_segment(source, node))

    # 🔹 Main block split
    for node in tree.body:
        if isinstance(node, ast.If):
            main_code = ast.get_source_segment(source, node)

            add_md("## 🚀 Execution Pipeline")
            parts = main_code.split("\n\n")

            for i, part in enumerate(parts):
                if part.strip():
                    add_code(part)

    nb["cells"] = cells

    with open(output_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("✅ Clean notebook created:", output_file)


# Usage
clean_notebook("classifier.py", "cla.ipynb")