"""Полностью выполнить notebook текущим Python и сохранить выводы."""
from pathlib import Path
import json
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent


def main():
    # Спецификация локальна для окружения; в argv используется текущий интерпретатор.
    kernel_dir = Path(sys.prefix) / 'share/jupyter/kernels'
    spec = kernel_dir / 'module3'
    spec.mkdir(parents=True, exist_ok=True)
    (spec / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Python (Module 3)', 'language': 'python'}), encoding='utf-8')
    manager = KernelManager(kernel_name='module3', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir)]))
    notebook = nbformat.read(ROOT / 'notebook.ipynb', as_version=4)

    def progress(cell, cell_index, **kwargs):
        print(f'Cell {cell_index + 1}/{len(notebook.cells)}', flush=True)

    client = NotebookClient(notebook, km=manager, timeout=1800, on_cell_start=progress)
    try:
        client.execute(cwd=str(ROOT))
    finally:
        # Выполненные выводы сохраняются даже при исключении, чтобы изучить ошибку.
        nbformat.write(notebook, ROOT / 'notebook.ipynb')


if __name__ == '__main__':
    main()
