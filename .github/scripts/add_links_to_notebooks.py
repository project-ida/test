import os
import sys
import nbformat

def add_links_to_notebook(notebook_path):
    # Read repository details from environment variables
    repo_owner = os.getenv("REPO_OWNER", "default-owner")
    repo_name = os.getenv("REPO_NAME", "default-repo")
    branch_name = os.getenv("BRANCH_NAME", "main")

    # Templates for Colab and nbviewer links
    colab_template = (
        '<a href="https://colab.research.google.com/github/{repo_owner}/{repo_name}/blob/{branch}/{file_path}" target="_parent">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )
    nbviewer_template = (
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        '<a href="https://nbviewer.org/github/{repo_owner}/{repo_name}/blob/{branch}/{file_path}" target="_parent">'
        '<img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>'
    )

    # Compute the relative file path from the repository root
    file_path = os.path.relpath(notebook_path).replace("\\", "/")
    colab_link = colab_template.format(
        repo_owner=repo_owner, repo_name=repo_name, branch=branch_name, file_path=file_path
    )
    nbviewer_link = nbviewer_template.format(
        repo_owner=repo_owner, repo_name=repo_name, branch=branch_name, file_path=file_path
    )
    full_links = f"{colab_link} {nbviewer_link}"

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Step 1: Check if the first cell is a Colab auto-added cell
    if notebook["cells"] and notebook["cells"][0]["cell_type"] == "markdown":
        first_cell = notebook["cells"][0]
        if (
            first_cell.get("metadata", {}).get("colab_type") == "text"
            and first_cell.get("metadata", {}).get("id") == "view-in-github"
        ):
            # Remove the Colab auto-added cell
            notebook["cells"].pop(0)
            print(f"Removed Colab auto-added cell from: {notebook_path}")

    # Step 2: Check the next cell
    if notebook["cells"]:
        second_cell = notebook["cells"][0]
        if second_cell["cell_type"] == "markdown":
            source = second_cell["source"]

            # Check if the cell contains both Colab and nbviewer links
            has_colab_link = "colab.research.google.com" in source
            has_nbviewer_link = "nbviewer.org" in source

            if has_colab_link and has_nbviewer_link:
                # Validate the links
                if colab_link not in source or nbviewer_link not in source:
                    # Rewrite the cell with correct links
                    second_cell["source"] = full_links
                    print(f"Updated links in: {notebook_path}")
                    with open(notebook_path, "w", encoding="utf-8") as f:
                        nbformat.write(notebook, f)
                    return True
                else:
                    # Links are correct; no changes needed
                    print(f"No changes needed: {notebook_path}")
                    return False

    # Step 3: Add a new markdown cell with the correct links
    links_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": full_links,
    }
    notebook["cells"].insert(0, links_cell)
    print(f"Added links to: {notebook_path}")

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)
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
