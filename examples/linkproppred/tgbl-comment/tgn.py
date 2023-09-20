"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python examples/linkproppred/tgbl-comment/tgn.py --data "tgbl-comment" --num_run 1 --seed 1
"""

import math
import timeit
import pickle

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from torch_geometric.nn import TransformerConv

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tqdm import tqdm
from typing import Dict


# ==========
# ========== Define helper function...
# ==========

def train():
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0

    popularity = torch.zeros(data.num_nodes)

    for batch in tqdm(train_loader, desc="Training", total=len(train_loader)):
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        if SAMPLING_STRATEGY == "naive":
            # Sample negative destination nodes.
            neg_dst = torch.randint(
                min_dst_idx,
                max_dst_idx + 1,
                (src.size(0),),
                dtype=torch.long,
                device=device,
            )
        elif SAMPLING_STRATEGY == "popularity":
            neg_dst = torch.multinomial(popularity + 1e-2, src.size(0), replacement=False).to(device)
            popularity *= POPULARITY_DECAY
            for entity in pos_dst:
                popularity[entity.item()] += 1
        else:
            raise NotImplementedError("Sampling strategy not implemented")

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](n_id)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


def predict(positive_src, positive_dst, negatives):
    dst = torch.tensor(
                np.concatenate(
                    ([np.array([positive_dst]), np.array(negatives)]),
                    axis=0,
                ),
                device=device,
            )
    src = torch.full((dst.shape[0],), positive_src, device=device)
    n_id = torch.cat([src, dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)

    # Get updated memory of all nodes involved in the computation.
    z, last_update = model["memory"](n_id)
    z = model["gnn"](
        z,
        last_update,
        edge_index,
        data.t[e_id].to(device),
        data.msg[e_id].to(device),
    )

    y_pred = model["link_pred"](z[assoc[src]], z[assoc[dst]])
    count_oversaturated_predictions = torch.sum(y_pred[1:,:].flatten() == 1.0).item()

    input_dict = {
        "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
        "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),# + noise,
        "eval_metric": [metric],
    }
    mrr = evaluator.eval(input_dict)[metric]

    return mrr, count_oversaturated_predictions

@torch.no_grad()
def test(loader, neg_sampler, split_mode, popular_negatives: np.ndarray, src_dst_t_to_index: Dict):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    naive_mrr = []
    popular_top20_mrr = []

    oversaturated_naive_sampling = []
    oversaturated_popular_top20_sampling = []

    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            mrr, naive_candidates_oversaturated_predictions = predict(
                positive_src=pos_src[idx],
                positive_dst=pos_dst.cpu().numpy()[idx],
                negatives=neg_batch,
            )
            naive_mrr.append(mrr)
            oversaturated_naive_sampling.append(naive_candidates_oversaturated_predictions)

            neg_popular_index = src_dst_t_to_index[(pos_src[idx].item(), pos_dst[idx].item(), pos_t[idx].item())]
            neg_popular = popular_negatives[neg_popular_index][:20]
            mrr, popular_candidates_oversaturated_predictions = predict(
                positive_src=pos_src[idx],
                positive_dst=pos_dst.cpu().numpy()[idx],
                negatives=neg_popular,
            )
            popular_top20_mrr.append(mrr)
            oversaturated_popular_top20_sampling.append(popular_candidates_oversaturated_predictions)

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(naive_mrr).mean())
    print(f"MRR on top 20 popular negatives: {float(torch.tensor(popular_top20_mrr).mean())}")

    print(f"Oversaturated predictions on naive sampling: {np.mean(oversaturated_naive_sampling)}")
    print(f"Oversaturated predictions on popular top 20 sampling: {np.mean(oversaturated_popular_top20_sampling)}")

    return perf_metrics

# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = "tgbl-comment"
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10

POPULARITY_DECAY = 0.95
SAMPLING_STRATEGY = args.sampling_strategy

NUM_TRAINING_EXAMPLES = 100_000
NUM_VALIDATION_EXAMPLES = 10_000

validation_popular_negatives = np.load(f"output/popular_neg_samples/{DATA}/popular_negatives_val.npy", mmap_mode="r")
with open(f"output/popular_neg_samples/{DATA}/src_dst_t_to_index_val.pickle", "rb") as handle:
    # mapping from (src, dst, t) to the index of the corresponding negative edge
    validation_src_dst_t_to_index = pickle.load(handle)

test_popular_negatives = np.load(f"output/popular_neg_samples/{DATA}/popular_negatives_test.npy", mmap_mode="r")
with open(f"output/popular_neg_samples/{DATA}/src_dst_t_to_index_test.pickle", "rb") as handle:
    # mapping from (src, dst, t) to the index of the corresponding negative edge
    test_src_dst_t_to_index = pickle.load(handle)

MODEL_NAME = 'TGN'
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading

dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

train_data = data[train_mask][-NUM_TRAINING_EXAMPLES:]
val_data = data[val_mask][:NUM_VALIDATION_EXAMPLES]
test_data = data[test_mask]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighhorhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

model = {'memory': memory,
         'gnn': gnn,
         'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train & Validation
    # loading the validation negative samples
    dataset.load_val_ns()

    val_perf_list = []
    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        # training
        start_epoch_train = timeit.default_timer()
        loss = train()
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
        )

        # validation
        start_val = timeit.default_timer()
        perf_metric_val = test(val_loader, neg_sampler, split_mode="val", popular_negatives=validation_popular_negatives, src_dst_t_to_index=validation_src_dst_t_to_index)
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        val_perf_list.append(perf_metric_val)

        # check for early stopping
        if early_stopper.step_check(perf_metric_val, model):
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

#     # ==================================================== Test
#     # first, load the best model
#     early_stopper.load_checkpoint(model)

#     # loading the test negative samples
#     dataset.load_test_ns()

#     # final testing
#     start_test = timeit.default_timer()
#     perf_metric_test = test(test_loader, neg_sampler, split_mode="test", popular_negatives=test_popular_negatives, src_dst_t_to_index=test_src_dst_t_to_index)

#     print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
#     print(f"\tTest: {metric}: {perf_metric_test: .4f}")
#     test_time = timeit.default_timer() - start_test
#     print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

#     save_results({'model': MODEL_NAME,
#                   'data': DATA,
#                   'run': run_idx,
#                   'seed': SEED,
#                   f'val {metric}': val_perf_list,
#                   f'test {metric}': perf_metric_test,
#                   'test_time': test_time,
#                   'tot_train_val_time': train_val_time
#                   },
#     results_filename)

#     print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
#     print('-------------------------------------------------------------------------------')

# print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
# print("==============================================================")
