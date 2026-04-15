import os
import pickle
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset


ROOM_NAMES = [
    "Bedroom", "Livingroom", "Kitchen", "Dining", "Corridor",
    "Stairs", "Storeroom", "Bathroom", "Balcony",
    "Structure", "Door", "Entrance Door", "Window",
]

VALID_ROOM_IDS = set(range(9))
DOOR_ROOM_IDS = {10, 11}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _numeric_pickle_ids(graph_dir):
    ids = []
    for filename in os.listdir(graph_dir):
        stem, ext = os.path.splitext(filename)
        if ext == ".pickle":
            ids.append(int(stem))
    return sorted(ids)


def _polygon_area(points):
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)


def _bbox_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _bbox_intersection_area(box_a, box_b):
    xmin = max(float(box_a[0]), float(box_b[0]))
    ymin = max(float(box_a[1]), float(box_b[1]))
    xmax = min(float(box_a[2]), float(box_b[2]))
    ymax = min(float(box_a[3]), float(box_b[3]))
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


class MSDFloorplanGraphDataset(Dataset):
    def __init__(
        self,
        graph_in_dir,
        graph_out_dir=None,
        keep_non_rooms=False,
        num_zoning_types=None,
        feature_mode="structural",
        semantic_features="none",
    ):
        super().__init__()
        self.graph_in_dir = graph_in_dir
        self.graph_out_dir = graph_out_dir
        self.keep_non_rooms = keep_non_rooms
        self.has_labels = graph_out_dir is not None
        self.feature_mode = feature_mode
        self.semantic_features = semantic_features

        input_ids = set(_numeric_pickle_ids(graph_in_dir))
        if self.has_labels:
            output_ids = set(_numeric_pickle_ids(graph_out_dir))
            common_ids = sorted(input_ids & output_ids)
            dropped_input = len(input_ids - output_ids)
            dropped_output = len(output_ids - input_ids)
            if dropped_input or dropped_output:
                warnings.warn(
                    f"Using {len(common_ids)} paired graphs; "
                    f"dropped {dropped_input} graph_in-only ids and "
                    f"{dropped_output} graph_out-only ids."
                )
            self.graph_ids = common_ids
        else:
            self.graph_ids = sorted(input_ids)

        if len(self.graph_ids) == 0:
            raise ValueError("No graph files found for the requested dataset.")

        if self.feature_mode == "zoning":
            if num_zoning_types is None:
                self.num_zoning_types = self._infer_num_zoning_types()
            else:
                self.num_zoning_types = num_zoning_types
        else:
            self.num_zoning_types = num_zoning_types

        self.base_feature_dim = self._get_base_feature_dim()
        self.semantic_feature_dim = self._get_semantic_feature_dim()

        print(
            f"Number of graphs: {len(self.graph_ids)} "
            f"(labels={'yes' if self.has_labels else 'no'}, "
            f"feature_mode={self.feature_mode}, semantic_features={self.semantic_features})"
        )

    def _get_base_feature_dim(self):
        if self.feature_mode == "structural":
            return 7
        if self.feature_mode == "zoning":
            if self.num_zoning_types is None:
                raise ValueError("num_zoning_types must be known for zoning features.")
            return self.num_zoning_types + 1
        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def _get_semantic_feature_dim(self):
        if self.semantic_features == "none":
            return 0
        if self.semantic_features == "room_shape":
            return 7
        raise ValueError(f"Unsupported semantic_features mode: {self.semantic_features}")

    def _infer_num_zoning_types(self):
        max_zoning_type = -1
        for graph_id in self.graph_ids:
            graph_in = load_pickle(os.path.join(self.graph_in_dir, f"{graph_id}.pickle"))
            for _, attrs in graph_in.nodes(data=True):
                zoning_type = attrs.get("zoning_type")
                if zoning_type is not None:
                    max_zoning_type = max(max_zoning_type, int(zoning_type))

        if max_zoning_type < 0:
            raise ValueError("Could not infer zoning_type values from graph_in dataset.")

        return max_zoning_type + 1

    def len(self):
        return len(self.graph_ids)

    def _build_edge_index(self, graph):
        edge_list = []
        for u, v in graph.edges():
            edge_list.append([u, v])
            edge_list.append([v, u])

        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def _build_structural_features(self, graph):
        n = graph.number_of_nodes()
        degrees = np.array([d for _, d in graph.degree()], dtype=np.float32)

        if n > 1:
            norm_degree = degrees / (n - 1)
        else:
            norm_degree = np.zeros_like(degrees)
        
        clustering_dict = nx.clustering(graph)
        clustering = np.array([clustering_dict[i] for i in range(n)], dtype=np.float32)

        betweenness_dict = nx.betweenness_centrality(graph, normalized=True)
        betweenness = np.array([betweenness_dict[i] for i in range(n)], dtype=np.float32)

        pagerank_dict = nx.pagerank(graph) if n > 0 else {}
        pagerank = np.array([pagerank_dict.get(i, 0.0) for i in range(n)], dtype=np.float32)

        closeness_dict = nx.closeness_centrality(graph)
        closeness = np.array([closeness_dict[i] for i in range(n)], dtype=np.float32)


        avg_neighbor_deg = np.array(
        [np.mean([degrees[nb] for nb in graph.neighbors(i)]) if degrees[i] > 0 else 0.0
        for i in range(n)],
        dtype=np.float32,
)
        avg_neighbor_deg_norm = avg_neighbor_deg / (n - 1) if n > 1 else np.zeros_like(avg_neighbor_deg)

        triangles_dict = nx.triangles(graph)
        triangles = np.array([triangles_dict[i] for i in range(n)], dtype=np.float32)

        triangles_norm = triangles / (triangles.max() + 1e-6)

        is_leaf = (degrees == 1).astype(np.float32)

        x = np.stack(
            [norm_degree, clustering, betweenness, pagerank, is_leaf, closeness, avg_neighbor_deg_norm],
            axis=1,
        )
        return torch.tensor(x, dtype=torch.float)

    def _build_zoning_features(self, graph):
        degrees = dict(graph.degree())
        features = []
        for node_idx, attrs in graph.nodes(data=True):
            zoning_type = int(attrs["zoning_type"])
            zoning_one_hot = F.one_hot(
                torch.tensor(zoning_type), num_classes=self.num_zoning_types
            ).to(torch.float)
            degree_feature = torch.tensor([float(degrees[node_idx])], dtype=torch.float)
            features.append(torch.cat([zoning_one_hot, degree_feature]))
        return torch.stack(features)

    def _build_room_shape_semantics(self, graph_in_full, graph_out_full, keep_nodes):
        semantic_dim = self._get_semantic_feature_dim()
        zeros = torch.zeros((len(keep_nodes), semantic_dim), dtype=torch.float)

        source_graph = None
        if graph_in_full is not None and all(
            node in graph_in_full.nodes and "geometry" in graph_in_full.nodes[node]
            for node in keep_nodes
        ):
            source_graph = graph_in_full
        elif graph_out_full is not None and all(
            node in graph_out_full.nodes and "geometry" in graph_out_full.nodes[node]
            for node in keep_nodes
        ):
            source_graph = graph_out_full
        else:
            return zeros

        boxes = {}
        areas = {}
        lengths = {}
        widths = {}
        global_xmins = []
        global_ymins = []
        global_xmaxs = []
        global_ymaxs = []

        for node in keep_nodes:
            points = source_graph.nodes[node]["geometry"]
            box = _bbox_from_points(points)
            boxes[node] = box
            areas[node] = _polygon_area(points)
            side_x = float(box[2] - box[0])
            side_y = float(box[3] - box[1])
            lengths[node] = max(side_x, side_y)
            widths[node] = min(side_x, side_y)
            global_xmins.append(float(box[0]))
            global_ymins.append(float(box[1]))
            global_xmaxs.append(float(box[2]))
            global_ymaxs.append(float(box[3]))

        total_area = max(sum(areas.values()), 1e-6)
        graph_span = max(
            max(global_xmaxs) - min(global_xmins),
            max(global_ymaxs) - min(global_ymins),
            1e-6,
        )

        door_counts = defaultdict(int)
        if graph_out_full is not None:
            for node in keep_nodes:
                if node not in graph_out_full:
                    continue
                for neighbor in graph_out_full.neighbors(node):
                    neighbor_type = graph_out_full.nodes[neighbor].get("room_type")
                    if neighbor_type in DOOR_ROOM_IDS:
                        door_counts[node] += 1

        is_parent = defaultdict(float)
        is_child = defaultdict(float)
        for i, node_i in enumerate(keep_nodes):
            area_i = _bbox_area(boxes[node_i])
            for node_j in keep_nodes[i + 1:]:
                area_j = _bbox_area(boxes[node_j])
                intersection = _bbox_intersection_area(boxes[node_i], boxes[node_j])

                if area_j > 0 and intersection > 0.7 * area_j:
                    if area_i > area_j:
                        is_parent[node_i] = 1.0
                        is_child[node_j] = 1.0
                    else:
                        is_child[node_i] = 1.0
                        is_parent[node_j] = 1.0

                if area_i > 0 and intersection > 0.7 * area_i:
                    if area_j > area_i:
                        is_parent[node_j] = 1.0
                        is_child[node_i] = 1.0
                    else:
                        is_child[node_j] = 1.0
                        is_parent[node_i] = 1.0

        semantic_features = []
        for node in keep_nodes:
            semantic_features.append(
                [
                    float(areas[node] / total_area),
                    float(lengths[node] / graph_span),
                    float(widths[node] / graph_span),
                    float(door_counts[node]),
                    float(is_parent[node]),
                    float(is_child[node]),
                    1.0,
                ]
            )

        return torch.tensor(semantic_features, dtype=torch.float)

    def get(self, index):
        graph_id = self.graph_ids[index]
        graph_in_full = load_pickle(os.path.join(self.graph_in_dir, f"{graph_id}.pickle"))
        graph_out_full = None
        if self.has_labels:
            graph_out_full = load_pickle(os.path.join(self.graph_out_dir, f"{graph_id}.pickle"))

        if self.feature_mode == "zoning":
            keep_nodes = [n for n, attrs in graph_in_full.nodes(data=True) if "zoning_type" in attrs]
        else:
            keep_nodes = list(graph_in_full.nodes())

        if graph_out_full is not None:
            keep_nodes = [
                n for n in keep_nodes
                if n in graph_out_full.nodes and "room_type" in graph_out_full.nodes[n]
            ]
            if not self.keep_non_rooms:
                keep_nodes = [
                    n for n in keep_nodes
                    if int(graph_out_full.nodes[n]["room_type"]) in VALID_ROOM_IDS
                ]

        if len(keep_nodes) == 0:
            raise ValueError(f"Graph {graph_id} has no usable nodes after filtering.")

        graph_in = graph_in_full.subgraph(keep_nodes).copy()
        graph_out = None
        if graph_out_full is not None:
            graph_out = graph_out_full.subgraph(keep_nodes).copy()

        relabel = {node_id: idx for idx, node_id in enumerate(keep_nodes)}
        graph_in = nx.relabel_nodes(graph_in, relabel, copy=True)
        if graph_out is not None:
            graph_out = nx.relabel_nodes(graph_out, relabel, copy=True)

        labels = []
        for node_idx in graph_in.nodes():
            if graph_out is not None:
                labels.append(int(graph_out.nodes[node_idx]["room_type"]))

        if self.feature_mode == "zoning":
            x = self._build_zoning_features(graph_in)
        elif self.feature_mode == "structural":
            x = self._build_structural_features(graph_in)
        else:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

        if self.semantic_features == "room_shape":
            semantic_x = self._build_room_shape_semantics(graph_in_full, graph_out_full, keep_nodes)
            x = torch.cat([x, semantic_x], dim=1)
        elif self.semantic_features != "none":
            raise ValueError(f"Unsupported semantic_features mode: {self.semantic_features}")

        edge_index = self._build_edge_index(graph_in)

        data = Data(
            x=x,
            edge_index=edge_index,
            graph_id=torch.tensor([graph_id], dtype=torch.long),
            original_node_ids=torch.tensor(keep_nodes, dtype=torch.long),
            base_feature_dim=torch.tensor([self.base_feature_dim], dtype=torch.long),
            semantic_feature_dim=torch.tensor([self.semantic_feature_dim], dtype=torch.long),
        )

        if graph_out is not None:
            data.y = torch.tensor(labels, dtype=torch.long)

        return data
