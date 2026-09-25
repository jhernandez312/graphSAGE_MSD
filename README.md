# Room Classification on Floor Plan Graphs

Room classification on floor plan graphs using graph neural networks.

- [Graph2Plan paper](https://arxiv.org/abs/2108.05947)
- [Modified Swiss Dwellings dataset](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings)
- [Legacy HouseGAN data](https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=0&preview=housegan_clean_data.npy)

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

## Consolidated pipeline

[`run_pipeline.py`](run_pipeline.py) is the main workflow for optional training, graph generation, conversion, GraphSAGE inference, and visualization. Edit the typed `CONFIG` block near the top of that file, then run:

```powershell
uv run python run_pipeline.py
```

GraphRNN and GraphSAGE training are separate switches and both default to `False`, so the default run uses the configured existing checkpoints. Set the corresponding `enabled` field to `True` and fill in that stage's dataset paths when retraining is needed. A successful run creates a new, non-overwriting directory below `runs/` with a manifest and one directory per generated graph.

Each graph directory contains:

- `raw_graph.png`, showing the generated topology.
- `graph.pt`, containing the PyG `Data`, adjacency matrix, feature mode, predictions, and provenance.
- `classified_graph.png`, showing the GraphSAGE room predictions.
- `predictions.json`, listing each node's class ID and room name.
- `metadata.json`, recording seeds, checkpoints, feature schema, layout, and graph sizes.

The graph is sampled once. Conversion, both images, predictions, and metadata all use that same adjacency matrix. Generated graphs provide six structural features; when a structural GraphSAGE checkpoint declares additional semantic columns, the pipeline records and zero-fills those columns. Zoning checkpoints and directed GraphRNN checkpoints are rejected because the generator does not provide the required attributes.


### Expected dataset layout

Set the root for the current shell after downloading. To keep it for future PowerShell sessions, also save it as a user environment variable:

```powershell
$env:MSD_DATA_ROOT = $datasetRoot
[Environment]::SetEnvironmentVariable("MSD_DATA_ROOT", $datasetRoot, "User")
```

For `--graphs-only` and existing legacy copies, `MSD_DATA_ROOT` should contain this layout:

```text
MSD_DATA_ROOT/
|-- actual_data/
    |-- train/
    |   |-- graph_in/
    |   |   `-- <id>.pickle
    |   `-- graph_out/
    |       `-- <id>.pickle
    `-- test/
        `-- graph_in/
            `-- <id>.pickle
```

The full Kaggle download keeps its archive-native outer directory. This layout is also accepted:

```text
MSD_DATA_ROOT/
|-- dataset_receipt.json
`-- modified-swiss-dwellings-v2/
    |-- train/
    |   |-- graph_in/
    |   |-- graph_out/
    |   |-- struct_in/
    |   `-- full_out/
    `-- test/
        |-- graph_in/
        `-- struct_in/
```

Additional files from the Kaggle release may be present. Training requires the three graph directories shown in the first layout; `struct_in` and `full_out` are not used by the current GraphRNN or GraphSAGE code.

## MSD training

GraphRNN, GraphSAGE, and enabled training stages in `run_pipeline.py` resolve their default dataset paths from `MSD_DATA_ROOT`. Generation with both training switches disabled does not require the dataset.

Run GraphSAGE training from the repository root:

```powershell
uv run python msd_train.py `
  --feature_mode structural `
  --semantic_features room_shape `
  --semantic_dropout 0.5 `
  --val_ratio 0.1
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

The Swiss GraphRNN configuration also resolves `train/graph_in` from `MSD_DATA_ROOT`; `--graph-dir` remains available as an explicit override. See [`ablation_GraphRNN/graph-rnn/README.md`](ablation_GraphRNN/graph-rnn/README.md) for the model requirements and background.
