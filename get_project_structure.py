import re
from datetime import datetime
from pathlib import Path


def write_structure_to_file(
    start_path=".",
    output_file="project_structure.txt",
    ignore_patterns=None,
    script_name=None,
    max_depth=None,
    show_size=True,
    show_date=True,
    sort=True,
    use_relative_path=False,
):
    start_path = Path(start_path).resolve()
    ignore_patterns = ignore_patterns or []
    ignore_regexes = [re.compile(pattern) for pattern in ignore_patterns]

    def should_ignore(path):
        if path.name == script_name:
            return True
        rel_path = path.relative_to(start_path)
        for regex in ignore_regexes:
            if regex.search(str(rel_path)) or regex.search(path.name):
                return True
        return False

    def get_size(path):
        if should_ignore(path):
            return 0
        if path.is_file():
            return path.stat().st_size
        return sum(
            get_size(item) for item in path.iterdir() if not should_ignore(item)
        )

    def format_size(size):
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.2f}{unit}"
            size /= 1024
        return f"{size:.2f}TB"

    def write_tree(root, level, f, is_last, parent_has_next, current_depth=0):
        if max_depth is not None and current_depth > max_depth:
            return

        if should_ignore(root):
            return

        if level == 0:
            indent = ""
        else:
            indent = "".join(
                ["│   " if p else "    " for p in parent_has_next[:-1]]
            )
            indent += "└── " if is_last else "├── "

        path = root.relative_to(start_path) if use_relative_path else root.name
        path = str(path).replace(
            "\\", "/"
        )  # Replace backslashes with forward slashes

        size_str = f" ({format_size(get_size(root))})" if show_size else ""
        date_str = (
            f" [Last modified: {datetime.fromtimestamp(root.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}]"
            if show_date
            else ""
        )

        f.write(f"{indent}{path}{size_str}{date_str}\n")

        if root.is_dir():
            items = [item for item in root.iterdir() if not should_ignore(item)]
            if sort:
                items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            for i, item in enumerate(items):
                is_last_item = i == len(items) - 1
                next_parent_has_next = parent_has_next + [not is_last_item]
                write_tree(
                    item,
                    level + 1,
                    f,
                    is_last_item,
                    next_parent_has_next,
                    current_depth + 1,
                )

    def count_files_and_dirs(path):
        files = set()
        dirs = set()
        for item in path.rglob("*"):
            if not should_ignore(item):
                if item.is_file():
                    files.add(item)
                elif item.is_dir():
                    dirs.add(item)
        return len(files), len(dirs)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{start_path.name}\n")
        items = [
            item for item in start_path.iterdir() if not should_ignore(item)
        ]
        if sort:
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            write_tree(item, 1, f, is_last, [not is_last], 1)

        total_size = get_size(start_path)
        total_files, total_dirs = count_files_and_dirs(start_path)

        f.write(f"\nSummary:\n")
        f.write(f"Total size: {format_size(total_size)}\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Total directories: {total_dirs}\n")


if __name__ == "__main__":
    script_name = Path(__file__).name
    write_structure_to_file(
        ignore_patterns=[
            r"^\.|^_",  # Hidden files/folders and those starting with underscore
            r"\.pyc$",
            r"__pycache__",
            r"^\.git$",  # Git directory
            r"^\.gitignore$",  # Git ignore file
            r"^_archive$",  # Archive directory
            r"^node_modules$",  # Node.js modules
            r"^venv$",  # Python virtual environment
            r"^\.vscode$",  # VS Code settings
            r"^\.idea$",  # PyCharm settings
            r"\.txt$",  # txt files
            r"\.xlsx$",  # excel files
            r"\.csv$",  # csv files
            r"\.json$",  # json files
        ],
        script_name=script_name,
        max_depth=None,
        show_size=False,
        show_date=False,
        sort=True,
        use_relative_path=True,
    )
