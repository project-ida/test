import sys
import nbformat
import os

def add_links_to_notebook(notebook_path, repo_base_url="https://github.com/project-ida/test/blob/main"):
    colab_template = (
        '<a href="https://colab.research.google.com/github/{repo_path}" target="_parent">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )
    nbviewer_template = (
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        '<a href="https://nbviewer.org/github/{repo_path}" target="_parent">'
        '<img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>'
    )

    # Compute the full relative path from the repository root
    repo_path = os.path.relpath(notebook_path).replace("\\", "/")
    full_colab_link = colab_template.format(repo_path=repo_path)
    full_nbviewer_link = nbviewer_template.format(repo_path=repo_path)
    full_links = f"{full_colab_link} {full_nbviewer_link}"

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Check if the first cell exists
    if notebook["cells"]:
        first_cell = notebook["cells"][0]

        if first_cell["cell_type"] == "markdown":
            source = first_cell["source"]

            # If Colab exists but nbviewer is missing, or if links are incorrect
            if "colab.research.google.com" in source:
                has_nbviewer = "nbviewer.org" in source
                if not has_nbviewer or full_colab_link not in source or full_nbviewer_link not in source:
                    # Rewrite the cell with both correct links
                    first_cell["source"] = full_links
                    with open(notebook_path, "w", encoding="utf-8") as f:
                        nbformat.write(notebook, f)
                    print(f"Updated notebook: {notebook_path} (rewrote first cell with correct links)")
                    return True
                else:
                    # Both links are present and correct; no changes needed
                    print(f"No changes needed: {notebook_path}")
                    return False

        # If first cell is markdown but doesn't need changes, add a new cell above
        new_links_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": full_links,
        }
        notebook["cells"].insert(0, new_links_cell)
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        print(f"Updated notebook: {notebook_path} (added links as a new cell)")
        return True

    # If no cells exist or first cell is not markdown, add a new cell at the top
    links_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": full_links,
    }
    notebook["cells"].insert(0, links_cell)

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)
    print(f"Updated notebook: {notebook_path} (added links as a new cell)")
    return True


def main(file_list_path):
    with open(file_list_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]

    for notebook_path in files:
        if notebook_path.endswith(".ipynb"):
            add_links_to_notebook(notebook_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_links_to_notebooks.py <file_list>")
        sys.exit(1)

    main(sys.argv[1])
