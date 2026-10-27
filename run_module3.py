"""Выполнить notebook в изолированном kernel и сохранить также при ошибке."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import json
import sys

root = Path(__file__).parent / 'module-3-churn'
kernel_dir = Path(__file__).parent / '.venv-module3' / 'share' / 'jupyter' / 'kernels'
spec = kernel_dir / 'module3'
spec.mkdir(parents=True, exist_ok=True)
(spec / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python (Module 3)', 'language': 'python'}), encoding='utf-8')
manager = KernelManager(kernel_name='module3', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]))
notebook = nbformat.read(root / 'notebook.ipynb', as_version=4)


def progress(cell, cell_index, **kwargs):
    print(f'Executing cell {cell_index + 1}/{len(notebook.cells)}', flush=True)


client = NotebookClient(notebook, km=manager, timeout=1800, resources={'metadata': {'path': str(root)}},
                        on_cell_start=progress)
try:
    client.execute(cwd=str(root))
finally:
    # Даже при ошибке можно изучить уже выполненные ячейки и traceback.
    notebook.metadata.kernelspec = {'display_name': 'Python (Module 3)', 'language': 'python', 'name': 'module3'}
    nbformat.write(notebook, root / 'notebook.ipynb')
