import pickle
from typing import Dict

import numpy as np
import torch
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator


BATCH_SIZE = 200

# Evaluating only on naive negative sampling is faster
# Set this to False to compute the top N popular negative sampling

# TODO: make this both inductive and generalizable to bipartite graphs
def update_popularity(popularity: np.ndarray, nodes: torch.Tensor, decay: float) -> np.ndarray:
    """
    Update the popularity array based on destination nodes and decay factor.

    Args:
        popularity: Current popularity array.
        dst_nodes: Destination nodes to update popularity for.
        decay: Decay factor for popularity.

    Returns:
        Updated popularity array.
    """
    for node in nodes:
        popularity[node[0].item()] += 0.01
        popularity[node[1].item()] += 1
    popularity *= decay
    return popularity


def sort_tensors_by_time(
    timestamps: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Sorts the given tensors based on the values in train_data_t in ascending order.

    Parameters:
        timestamps: The tensor containing time data.
        src: The tensor containing source data.
        dst: The tensor containing destination data.

    Returns:
        tuple: Sorted timestamps, src, dst tensors.
    """
    sorted_indices = torch.argsort(timestamps)
    sorted_t = timestamps[sorted_indices]
    sorted_src = src[sorted_indices]
    sorted_dst = dst[sorted_indices]

    return sorted_t, sorted_src, sorted_dst


def train(train_loader: TemporalDataLoader, num_nodes: int, decay: float) -> np.ndarray:
    """
    Train the popularity model.

    Args:
        train_loader: The data loader for the training set.
        num_nodes: The total number of nodes in the graph.
        decay: The decay factor for popularity.

    Returns:
        The popularity scores for each node.
    """
    popularity = np.zeros(num_nodes)
    for batch in tqdm(train_loader):
        # create nodes variable which consists of src, dst pairs for each edge in the batch as a 2tuple
        nodes = torch.stack([batch.src, batch.dst], dim=1)
        popularity = update_popularity(popularity, nodes, decay)
    return popularity


def test(
    loader,
    neg_sampler,
    split_mode: str,
    popularity: torch.Tensor,
    decay: float,
    evaluator: Evaluator,
):
    """
    Test the popularity model and compute the Mean Reciprocal Rank (MRR).

    Args:
        loader: The data loader.
        neg_sampler: The negative sampler.
        split_mode: The mode for splitting the data ('val' or 'test').
        popularity: The popularity scores for each node.
        decay: The decay factor for popularity.
        evaluator: The evaluator object for computing metrics.

    Returns:
        The Mean Reciprocal Rank (MRR) for the given data.
    """
    naive_mrr = []

    for pos_batch in tqdm(loader):
        # we call it pos_batch because the test set only has positive samples
        pos_src, pos_dst, pos_t = pos_batch.src, pos_batch.dst, pos_batch.t
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, negative_candidates in enumerate(neg_batch_list):
            ground_truth = pos_dst[idx].item()
            candidates = np.concatenate([negative_candidates, [ground_truth]])
            scores = popularity[candidates]
            input_dict = {
                "y_pred_pos": np.array([scores[-1]]),
                "y_pred_neg": scores[:-1],
                "eval_metric": ["mrr"],
            }

            naive_mrr.append(evaluator.eval(input_dict)["mrr"])

        nodes = torch.stack([pos_batch.src, pos_batch.dst], dim=1)
        popularity = update_popularity(popularity, nodes, decay)

    naive_mrr = float(torch.tensor(naive_mrr).mean())
    print(f"Naive MRR: {naive_mrr}")

    return naive_mrr


def run(dataset_name: str, initial_decay: float):
    """
    Run the popularity baseline model for a given dataset.

    Args:
        dataset_name: The name of the dataset to use.
        initial_decay: The initial decay factor for popularity.
    """

    print(f"Running popularity baseline for {dataset_name} with decay {initial_decay}")
    dataset = PyGLinkPropPredDataset(name=dataset_name, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    data = dataset.get_TemporalData()
    assert dataset.eval_metric == "mrr"

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    train_data.t, train_data.src, train_data.dst = sort_tensors_by_time(train_data.t, train_data.src, train_data.dst)
    val_data.t, val_data.src, val_data.dst = sort_tensors_by_time(val_data.t, val_data.src, val_data.dst)
    test_data.t, test_data.src, test_data.dst = sort_tensors_by_time(test_data.t, test_data.src, test_data.dst)

    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

    neg_sampler = dataset.negative_sampler
    dataset.load_val_ns()

    best_mrr = 0.0
    mrr_per_decay = {}

    evaluator = Evaluator(name=dataset_name)
    # Grid search over decay hyperparameter
    decay = initial_decay
    best_decay = decay

    while True:
        if decay > 1.0:
            break
        popularity = train(train_loader, num_nodes=data.num_nodes, decay=initial_decay)
        mrr = test(
            val_loader,
            neg_sampler,
            split_mode="val",
            popularity=popularity,
            decay=decay,
            evaluator=evaluator,
        )
        print(f"MRR: {mrr} for decay {decay}")
        mrr_per_decay[decay] = mrr
        if mrr > best_mrr:
            best_mrr = mrr
            best_decay = decay
        else:
            break
        if decay >= 0.99:
            decay += 0.001
        else:
            decay += 0.01
    print(f"Best MRR: {best_mrr} for decay {best_decay}")

    dataset.load_test_ns()
    # Test set
    train_val_data = data[train_mask | val_mask]
    train_val_loader = TemporalDataLoader(train_val_data, batch_size=BATCH_SIZE)
    popularity = train(train_val_loader, num_nodes=data.num_nodes, decay=initial_decay)
    mrr = test(
        test_loader,
        neg_sampler,
        split_mode="test",
        popularity=popularity,
        decay=best_decay,
        evaluator=evaluator,
    )
    print(f"MRR on test set: {mrr}")


if __name__ == "__main__":
    # run("tgbl-comment", 0.94)
    run("tgbl-review", initial_decay=0.997)
    # run("tgbl-coin", initial_decay=0.93)
    # run("tgbl-wiki", initial_decay=0.36)
