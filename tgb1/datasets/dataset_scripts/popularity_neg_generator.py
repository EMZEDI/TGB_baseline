import os
import pickle

import numpy as np
import torch
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


def train(num_nodes: int, targets: torch.Tensor, decay: float):
    popularity = torch.zeros(num_nodes)
    for ix, dst in tqdm(enumerate(targets), desc="Training popularity scores", total=len(targets)):
        popularity[dst.item()] += 1
        if ix > 0 and ix % BATCH_SIZE == 0:
            popularity *= decay
    return popularity


BATCH_SIZE = 200
TOP_N_POPULAR = 500


def create_negatives(
    loader, popularity: torch.Tensor, decay: float, num_examples: int, file_suffix: str, output_dir: str
):
    # empty buffer for negative samples
    popular_negatives = torch.zeros(num_examples, TOP_N_POPULAR, requires_grad=False, dtype=torch.int32)

    # map (src, dst, t) to index in popular_negatives
    src_dst_t_to_index = {}

    counter = 0
    for pos_batch in tqdm(loader, desc=f"Generating negatives {file_suffix}", total=len(loader)):
        pos_src, pos_dst, pos_t = pos_batch.src, pos_batch.dst, pos_batch.t
        negative_candidates = torch.topk(popularity, k=TOP_N_POPULAR + 1).indices.cpu().numpy()

        for idx in range(pos_dst.shape[0]):
            # filter out positive dst from negative candidates
            popular_negatives[counter] = torch.tensor(
                negative_candidates[negative_candidates != pos_dst[idx].item()][:TOP_N_POPULAR]
            )
            src_dst_t_to_index[(pos_src[idx].item(), pos_dst[idx].item(), pos_t[idx].item())] = counter
            counter += 1

        popularity *= decay
        for dst in pos_dst:
            popularity[dst.item()] += 1

    with open(f"{output_dir}/src_dst_t_to_index_{file_suffix}.pickle", "wb") as f:
        pickle.dump(src_dst_t_to_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(f"{output_dir}/popular_negatives_{file_suffix}.npy", popular_negatives.numpy())


def generate_negative_samples(dataset_name: str, popularity_decay: int):
    print(f"Generating negative samples for {dataset_name} with decay {popularity_decay}")
    output_dir = f"output/popular_neg_samples/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    dataset = PyGLinkPropPredDataset(name=dataset_name, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    data = dataset.get_TemporalData()

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    # create popularity scores from training data
    popularity = train(num_nodes=data.num_nodes, targets=train_data.dst, decay=popularity_decay)

    val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
    create_negatives(val_loader, popularity, popularity_decay, val_mask.sum().item(), "val", output_dir)

    test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)
    create_negatives(test_loader, popularity, popularity_decay, test_mask.sum().item(), "test", output_dir)


if __name__ == "__main__":
    generate_negative_samples(dataset_name="tgbl-comment", popularity_decay=0.95)
    generate_negative_samples(dataset_name="tgbl-coin", popularity_decay=0.94)
    generate_negative_samples(dataset_name="tgbl-review", popularity_decay=0.999)
