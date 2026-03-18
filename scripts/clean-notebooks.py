#!/usr/bin/env python3

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor

repo_root = Path(__file__).parent.parent
demo_dir = repo_root / "demo"
preprocessor = ClearOutputPreprocessor()

for file in demo_dir.glob("*.ipynb"):
    notebook = nbformat.read(file, as_version=nbformat.NO_CONVERT)
    preprocessor.preprocess(notebook, {})
    nbformat.write(notebook, file)
    print(f"Cleared: {file.relative_to(repo_root)}")
