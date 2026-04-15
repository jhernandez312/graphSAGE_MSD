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
  --graph-pt "..\ablation_GraphRNN\graph-rnn\generated_graph.pt" `
  --save-labeled-graph ".\results\labeled_generated_graph.pt"

```

RNN to GraphSAGE labeled graph
```
python rnn_to_graphsage.py `
  --graph-pt ".\ablation_GraphRNN\graph-rnn\generated_graph.pt" `
  --graphsage-model-path ".\results\best_model.pt" `
  --labeled-graph-out ".\results\labeled_generated_graph.pt"

```

Generate from GraphRNN checkpoint and feed into GraphSAGE
```
python .\ablation_GraphRNN\graph-rnn\generate.py `
  .\ablation_GraphRNN\graph-rnn\configs\checkpoints\checkpoint-96000.pth `
  -n 40 `
  --graphsage-out ".\results\generated_graph_40.pt"

python rnn_to_graphsage.py `
  --graph-pt ".\results\generated_graph_40.pt" `
  --graphsage-model-path ".\results\best_model.pt" `
  --labeled-graph-out ".\results\labeled_generated_graph_40.pt"

```

Or run the full bridge directly from the GraphRNN checkpoint
```
python rnn_to_graphsage.py `
  --rnn-model-path ".\ablation_GraphRNN\graph-rnn\configs\checkpoints\checkpoint-96000.pth" `
  --nodes 40 `
  --generated-graph-out ".\results\generated_graph_40.pt" `
  --graphsage-model-path ".\results\best_model.pt" `
  --labeled-graph-out ".\results\labeled_generated_graph_40.pt"

```

Notes about inference
- For a given graph, the predictions will be the same because the weights remain unchanged
- `labeled_generated_graph.pt` stores the predicted node labels in both a PyG `Data` object (`data.y`) and a NetworkX graph with `predicted_room_type` / `predicted_room_name` node attributes
