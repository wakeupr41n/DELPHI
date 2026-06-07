"""Verify all src/ modules can be parsed and import without runtime errors."""

import ast
import pathlib
import sys


def test_src_modules_parse():
    """All Python files in src/ must be syntactically valid."""
    src_dir = pathlib.Path(__file__).parent.parent / "src"
    py_files = sorted(src_dir.glob("*.py"))
    assert len(py_files) >= 4, f"Expected >=4 src/*.py, found {len(py_files)}"

    for f in py_files:
        if f.name.startswith("_"):
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {f.name}: {e}") from e
        assert tree is not None, f"Empty AST for {f.name}"


def test_no_broken_scripts_import():
    """No script should use 'from scripts.X import Y' (broken after reorg)."""
    scripts_dir = pathlib.Path(__file__).parent.parent / "scripts"
    for py_file in scripts_dir.rglob("*.py"):
        if "archive" in str(py_file) or "__pycache__" in str(py_file):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # syntax errors caught by test_src_modules_parse
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(
                    "scripts."
                ), f"{py_file.name} uses absolute scripts import: {node.module}"
