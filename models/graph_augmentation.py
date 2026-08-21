import copy
import random
import torch


class GraphAugmentation:

    def __init__(

        self,

        edge_drop_prob=0.2,

        feature_mask_prob=0.2,

        feature_noise_std=0.05,

        node_drop_prob=0.0

    ):

        self.edge_drop_prob = edge_drop_prob

        self.feature_mask_prob = feature_mask_prob

        self.feature_noise_std = feature_noise_std

        self.node_drop_prob = node_drop_prob

    # =====================================================
    # Edge Dropout
    # =====================================================

    def edge_dropout(

        self,

        edge_index

    ):

        num_edges = edge_index.size(1)

        keep_mask = (

            torch.rand(

                num_edges,

                device=edge_index.device

            )

            > self.edge_drop_prob

        )

        if keep_mask.sum().item() == 0:

            keep_mask[

                random.randint(

                    0,

                    num_edges - 1

                )

            ] = True

        return edge_index[:, keep_mask]

    # =====================================================
    # Feature Masking
    # =====================================================

    def feature_masking(

        self,

        x

    ):

        x = x.clone()

        mask = (

            torch.rand_like(x)

            < self.feature_mask_prob

        )

        x[mask] = 0

        return x

    # =====================================================
    # Feature Noise
    # =====================================================

    def feature_noise(

        self,

        x

    ):

        noise = (

            torch.randn_like(x)

            * self.feature_noise_std

        )

        return x + noise

    # =====================================================
    # Node Dropout
    # =====================================================

    def node_dropout(

        self,

        graph

    ):

        if self.node_drop_prob == 0:

            return graph

        graph = copy.deepcopy(graph)

        num_nodes = graph.num_nodes

        keep_nodes = (

            torch.rand(

                num_nodes,

                device=graph.x.device

            )

            > self.node_drop_prob

        )

        if keep_nodes.sum().item() < 2:

            return graph

        mapping = -torch.ones(

            num_nodes,

            dtype=torch.long,

            device=graph.x.device

        )

        mapping[keep_nodes] = torch.arange(

            keep_nodes.sum(),

            device=graph.x.device

        )

        graph.x = graph.x[keep_nodes]

        graph.y = graph.y[keep_nodes]

        src = graph.edge_index[0]

        dst = graph.edge_index[1]

        edge_mask = (

            keep_nodes[src]

            &

            keep_nodes[dst]

        )

        src = mapping[src[edge_mask]]

        dst = mapping[dst[edge_mask]]

        graph.edge_index = torch.stack(

            [

                src,

                dst

            ],

            dim=0

        )

        return graph

    # =====================================================
    # Augment One Graph
    # =====================================================

    def augment_graph(

        self,

        graph

    ):

        graph = copy.deepcopy(graph)

        graph = self.node_dropout(graph)

        graph.edge_index = self.edge_dropout(

            graph.edge_index

        )

        graph.x = self.feature_masking(

            graph.x

        )

        graph.x = self.feature_noise(

            graph.x

        )

        return graph

    # =====================================================
    # Create Two Views
    # =====================================================

    def __call__(

        self,

        graph_sequence

    ):

        view1 = []

        view2 = []

        for graph in graph_sequence:

            view1.append(

                self.augment_graph(graph)

            )

            view2.append(

                self.augment_graph(graph)

            )

        return view1, view2