## Room Classification on Floor Plan Graphs using Graph Neural Networks

[Paper](https://arxiv.org/abs/2108.05947) |
[Dataset](https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=0&preview=housegan_clean_data.npy) |
[Install PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)

### Usage
## How to Run

msd_train.py
```
python msd_train.py `
  --feature_mode structural `
  --semantic_features room_shape `
  --semantic_dropout 0.5 `
  --val_ratio 0.1 `
  --train_graph_in_dir "..\swiss_dwellings_test\actual_data\train\graph_in" `
  --train_graph_out_dir "..\swiss_dwellings_test\actual_data\train\graph_out" `
  --test_graph_in_dir "..\swiss_dwellings_test\actual_data\test\graph_in"

```


visualize_inference.py
```
python visualize_inference.py `
  --model-path ".\results\best_model.pt" `
  --graph-pt "..\ablation_GraphRNN\graph-rnn\generated_graph.pt"

```

Notes about inference
- For a given graph, the predictions will be the same because the weights remain unchanged
