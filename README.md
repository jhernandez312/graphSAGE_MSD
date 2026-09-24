# Room Classification on Floor Plan Graphs

Room classification on floor plan graphs using graph neural networks.

- [Paper](https://arxiv.org/abs/2108.05947)
- [Dataset](https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=0&preview=housegan_clean_data.npy)

## Environment setup

The repository uses [`uv`](https://docs.astral.sh/uv/) to reproduce its Python environment. The lockfile targets 64-bit Windows with Python 3.9.13, PyTorch 2.8.0, and CUDA 12.8. An installed system copy of Python 3.9.13 and an NVIDIA driver compatible with CUDA 12.8 are required.

On this Windows setup, create `.venv` with the installed Visual Studio Python 3.9.13 interpreter, then synchronize it from the committed lockfile:

```powershell
uv venv --python "C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python39_64\python.exe"
uv sync --frozen
```

On another machine, replace the interpreter path with its installed CPython 3.9.13 executable. Once `.venv` exists, only `uv sync --frozen` is needed to restore the locked packages.

Activation is optional because `uv run` automatically uses the project environment. To activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify the GPU runtime:

```powershell
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## MSD training

The graph directories are produced separately and are not included in this repository. Run training from the repository root, replacing the paths as needed:

```powershell
uv run python msd_train.py `
  --feature_mode structural `
  --semantic_features room_shape `
  --semantic_dropout 0.5 `
  --val_ratio 0.1 `
  --train_graph_in_dir "..\swiss_dwellings_test\actual_data\train\graph_in" `
  --train_graph_out_dir "..\swiss_dwellings_test\actual_data\train\graph_out" `
  --test_graph_in_dir "..\swiss_dwellings_test\actual_data\test\graph_in"
```

Use `uv run python msd_train.py --help` for all training options.

## Inference visualization

```powershell
uv run python visualize_inference.py `
  --model-path ".\results\best_model.pt" `
  --graph-pt ".\ablation_GraphRNN\graph-rnn\generated_graph.pt"
```

For a given graph, predictions remain deterministic while the model weights are unchanged.

## GraphRNN ablation

The bundled GraphRNN tools use the same root environment. Run them from their directory so relative checkpoint and log paths continue to resolve there:

```powershell
Push-Location .\ablation_GraphRNN\graph-rnn
uv run python train.py configs\config_swiss.yaml
uv run python generate_for_graphSAGE.py --help
Pop-Location
```

Before GraphRNN training, update `configs\config_swiss.yaml` if its dataset path does not match your checkout. See [`ablation_GraphRNN/graph-rnn/README.md`](ablation_GraphRNN/graph-rnn/README.md) for the model requirements and background.
