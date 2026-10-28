from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
base = Path(__file__).resolve().parent
book = nbformat.read(base / 'notebook.ipynb', as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
try:
    def checkpoint(cell, cell_index, **kwargs):
        print(f'Completed cell {cell_index}', flush=True)
        nbformat.write(book, base / 'notebook.ipynb')
    NotebookClient(book, km=manager, timeout=7200, on_cell_executed=checkpoint,
                   resources={'metadata': {'path': str(base)}}).execute()
finally:
    manager.shutdown_kernel(now=True)
    nbformat.write(book, base / 'notebook.ipynb')
