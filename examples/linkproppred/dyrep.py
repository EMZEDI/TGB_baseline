"""
DyRep
    This has been implemented with intuitions from the following sources:
    - https://github.com/twitter-research/tgn
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

    Spec.:
        - Memory Updater: RNN
        - Embedding Module: ID
        - Message Function: ATTN
"""
import timeit
import os
import os.path as osp
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import TemporalDataLoader

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import DyRepMemory
from modules.early_stopping import EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
import pickle
import random
from tqdm import tqdm
from tgb.utils.sliding_window_counter import SlidingWindowCounter


# ==========
# ========== Define helper function...
# ==========


def train():
    model["memory"].train()
    model["gnn"].train()
    model["link_pred"].train()

    model["memory"].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    popularity = torch.zeros(data.num_nodes) + 1e-15
    for batch in tqdm(train_loader, desc="Training", total=len(train_loader)):
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        if SAMPLING_STRATEGY == "naive":
            # Sample negative destination nodes.
            neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0),), dtype=torch.long, device=device,)
        elif SAMPLING_STRATEGY == "popularity":
            if random.random() < RANDOM_RATIO:
                neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0),), dtype=torch.long, device=device,)
            else:
                popularity_raised = torch.pow(popularity, POPULARITY_POWER)
                denominator = torch.sum(popularity_raised)
                neg_dst = torch.multinomial(popularity_raised / denominator, src.size(0), replacement=False).to(device)

            popularity *= POPULARITY_DECAY
            for entity in pos_dst:
                popularity[entity.item()] += 1
        else:
            raise ValueError("Unknown sampling strategy")

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model["memory"](n_id)

        pos_out = model["link_pred"](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model["link_pred"](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # update the memory with ground-truth
        z = model["gnn"](z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device),)
        model["memory"].update_state(src, pos_dst, t, msg, z, assoc)

        # update neighbor loader
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model["memory"].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


def predict(negatives, pos_src, pos_dst):
    src = torch.full((1 + len(negatives),), pos_src, device=device)
    dst = torch.tensor(np.concatenate(([np.array([pos_dst]), np.array(negatives)]), axis=0,), device=device,)

    n_id = torch.cat([src, dst]).unique()
    n_id, _, _ = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)

    # Get updated memory of all nodes involved in the computation.
    z, _ = model["memory"](n_id)

    y_pred = model["link_pred"](z[assoc[src]], z[assoc[dst]])
    return y_pred


@torch.no_grad()
def test_one_vs_many(loader, neg_sampler, split_mode, popular_negatives, src_dst_t_to_index):
    """
    Evaluated the dynamic link prediction
    """
    model["memory"].eval()
    model["gnn"].eval()
    model["link_pred"].eval()

    perf_list = []
    popular_top20_mrr = []
    popular_top100_mrr = []
    popular_top500_mrr = []

    oversaturated_naive_sampling = []
    oversaturated_popular_top20_sampling = []
    oversaturated_popular_top100_sampling = []
    oversaturated_popular_top500_sampling = []

    oversaturated_K50_N5000 = []
    oversaturated_K100_N5000 = []
    oversaturated_K1000_N5000 = []

    oversaturated_K50_N100000 = []
    oversaturated_K100_N100000 = []
    oversaturated_K1000_N100000 = []

    oversaturated_K50_N20000 = []
    oversaturated_K100_N20000 = []
    oversaturated_K1000_N20000 = []

    for batch_ix, pos_batch in tqdm(enumerate(loader), desc="Testing", total=len(loader)):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode)

        top_K1000_N5000 = torch.tensor([i[0] for i in counter_last_5_000.counter.most_common(1_000)])
        top_K1000_N20000 = torch.tensor([i[0] for i in counter_last_20_000.counter.most_common(1_000)])
        top_K1000_N100000 = torch.tensor([i[0] for i in counter_last_100_000.counter.most_common(1_000)])

        for idx, neg_batch in enumerate(neg_batch_list):
            naive_predictions = predict(neg_batch, pos_src[idx], pos_dst.cpu().numpy()[idx])
            oversaturated_naive_sampling.append(
                torch.sum(naive_predictions[1:, :].flatten() == 1.0).item() / len(neg_batch)
            )
            mrr_naive_input_dict = {
                "y_pred_pos": np.array([naive_predictions[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(naive_predictions[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(mrr_naive_input_dict)[metric])

            # compute MRR for popular negatives
            neg_popular_index = src_dst_t_to_index[(pos_src[idx].item(), pos_dst[idx].item(), pos_t[idx].item())]
            neg_popular = popular_negatives[neg_popular_index][:500]
            popular_predictions = predict(neg_popular, pos_src[idx], pos_dst.cpu().numpy()[idx])

            oversaturated_popular_top20_sampling.append(
                torch.sum(popular_predictions[1:21, :].flatten() == 1.0).item() / 20
            )
            oversaturated_popular_top100_sampling.append(
                torch.sum(popular_predictions[1:101, :].flatten() == 1.0).item() / 100
            )
            oversaturated_popular_top500_sampling.append(
                torch.sum(popular_predictions[1:501, :].flatten() == 1.0).item() / 500
            )

            mrr_popular_input_dict_top20 = {
                "y_pred_pos": np.array([popular_predictions[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(popular_predictions[1:21, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            popular_top20_mrr.append(evaluator.eval(mrr_popular_input_dict_top20)[metric])

            mrr_popular_input_dict_top100 = {
                "y_pred_pos": np.array([popular_predictions[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(popular_predictions[1:101, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            popular_top100_mrr.append(evaluator.eval(mrr_popular_input_dict_top100)[metric])

            mrr_popular_input_dict_top500 = {
                "y_pred_pos": np.array([popular_predictions[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(popular_predictions[1:501, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            popular_top500_mrr.append(evaluator.eval(mrr_popular_input_dict_top500)[metric])

            if split_mode == "val":
                # calcualate oversaturation
                predictions_N5000 = predict(top_K1000_N5000, pos_src[idx], pos_dst.cpu().numpy()[idx])
                # indexing from 1: to exclude the positive example
                oversaturated_K1000_N5000.append(
                    torch.sum(predictions_N5000[1:1001, :].flatten() == 1.0).item() / 1000
                )
                oversaturated_K100_N5000.append(torch.sum(predictions_N5000[1:101, :].flatten() == 1.0).item() / 100)
                oversaturated_K50_N5000.append(torch.sum(predictions_N5000[1:51, :].flatten() == 1.0).item() / 50)

                predictions_N20000 = predict(top_K1000_N20000, pos_src[idx], pos_dst.cpu().numpy()[idx])
                oversaturated_K1000_N20000.append(
                    torch.sum(predictions_N20000[1:1001, :].flatten() == 1.0).item() / 1000
                )
                oversaturated_K100_N20000.append(torch.sum(predictions_N20000[1:101, :].flatten() == 1.0).item() / 100)
                oversaturated_K50_N20000.append(torch.sum(predictions_N20000[1:51, :].flatten() == 1.0).item() / 50)

                predictions_N100000 = predict(top_K1000_N100000, pos_src[idx], pos_dst.cpu().numpy()[idx])
                oversaturated_K1000_N100000.append(
                    torch.sum(predictions_N100000[1:1001, :].flatten() == 1.0).item() / 1000
                )
                oversaturated_K100_N100000.append(
                    torch.sum(predictions_N100000[1:101, :].flatten() == 1.0).item() / 100
                )
                oversaturated_K50_N100000.append(torch.sum(predictions_N100000[1:51, :].flatten() == 1.0).item() / 50)

        if batch_ix > 0 and batch_ix % 1000 == 0:
            tqdm.write(str(batch_ix))
            tqdm.write(f"Naive MRR: {float(torch.tensor(perf_list).mean())}")
            tqdm.write(f"Popular MRR top 20: {float(torch.tensor(popular_top20_mrr).mean())}")
            tqdm.write(f"Popular MRR top 100: {float(torch.tensor(popular_top100_mrr).mean())}")
            tqdm.write(f"Popular MRR top 500: {float(torch.tensor(popular_top500_mrr).mean())}")
            tqdm.write(f"Oversaturated Naive Sampling: {float(torch.tensor(oversaturated_naive_sampling).mean())}")
            tqdm.write(
                f"Oversaturated Popular Top 20 Sampling: {float(torch.tensor(oversaturated_popular_top20_sampling).mean())}"
            )
            tqdm.write(
                f"Oversaturated Popular Top 100 Sampling: {float(torch.tensor(oversaturated_popular_top100_sampling).mean())}"
            )
            tqdm.write(
                f"Oversaturated Popular Top 500 Sampling: {float(torch.tensor(oversaturated_popular_top500_sampling).mean())}"
            )
            if split_mode == "val":
                tqdm.write(f"Oversaturated K50 N5000: {float(torch.tensor(oversaturated_K50_N5000).mean())}")
                tqdm.write(f"Oversaturated K100 N5000: {float(torch.tensor(oversaturated_K100_N5000).mean())}")
                tqdm.write(f"Oversaturated K1000 N5000: {float(torch.tensor(oversaturated_K1000_N5000).mean())}")
                tqdm.write(f"Oversaturated K50 N20000: {float(torch.tensor(oversaturated_K50_N20000).mean())}")
                tqdm.write(f"Oversaturated K100 N20000: {float(torch.tensor(oversaturated_K100_N20000).mean())}")
                tqdm.write(f"Oversaturated K1000 N20000: {float(torch.tensor(oversaturated_K1000_N20000).mean())}")
                tqdm.write(f"Oversaturated K50 N100000: {float(torch.tensor(oversaturated_K50_N100000).mean())}")
                tqdm.write(f"Oversaturated K100 N100000: {float(torch.tensor(oversaturated_K100_N100000).mean())}")
                tqdm.write(f"Oversaturated K1000 N100000: {float(torch.tensor(oversaturated_K1000_N100000).mean())}")

        # update the memory with positive edges
        n_id = torch.cat([pos_src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = model["memory"](n_id)
        z = model["gnn"](z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device),)
        model["memory"].update_state(pos_src, pos_dst, pos_t, pos_msg, z, assoc)

        # update the neighbor loader
        neighbor_loader.insert(pos_src, pos_dst)
        for dst in pos_dst:
            counter_last_100_000.add(dst.item())
            counter_last_20_000.add(dst.item())
            counter_last_5_000.add(dst.item())

    perf_metric = float(torch.tensor(perf_list).mean())
    print(f"Naive MRR: {perf_metric}")
    print(f"Popular MRR top 20: {float(torch.tensor(popular_top20_mrr).mean())}")
    print(f"Popular MRR top 100: {float(torch.tensor(popular_top100_mrr).mean())}")
    print(f"Popular MRR top 500: {float(torch.tensor(popular_top500_mrr).mean())}")
    print(f"Oversaturated Naive Sampling: {float(torch.tensor(oversaturated_naive_sampling).mean())}")
    print(f"Oversaturated Popular Top 20 Sampling: {float(torch.tensor(oversaturated_popular_top20_sampling).mean())}")
    print(
        f"Oversaturated Popular Top 100 Sampling: {float(torch.tensor(oversaturated_popular_top100_sampling).mean())}"
    )
    print(
        f"Oversaturated Popular Top 500 Sampling: {float(torch.tensor(oversaturated_popular_top500_sampling).mean())}"
    )
    if split_mode == "val":
        print(f"Oversaturated K50 N5000: {float(torch.tensor(oversaturated_K50_N5000).mean())}")
        print(f"Oversaturated K100 N5000: {float(torch.tensor(oversaturated_K100_N5000).mean())}")
        print(f"Oversaturated K1000 N5000: {float(torch.tensor(oversaturated_K1000_N5000).mean())}")
        print(f"Oversaturated K50 N20000: {float(torch.tensor(oversaturated_K50_N20000).mean())}")
        print(f"Oversaturated K100 N20000: {float(torch.tensor(oversaturated_K100_N20000).mean())}")
        print(f"Oversaturated K1000 N20000: {float(torch.tensor(oversaturated_K1000_N20000).mean())}")
        print(f"Oversaturated K50 N100000: {float(torch.tensor(oversaturated_K50_N100000).mean())}")
        print(f"Oversaturated K100 N100000: {float(torch.tensor(oversaturated_K100_N100000).mean())}")
        print(f"Oversaturated K1000 N100000: {float(torch.tensor(oversaturated_K1000_N100000).mean())}")

    result = {
        "naive_mrr": perf_metric,
        "popular_mrr_top20": float(torch.tensor(popular_top20_mrr).mean()),
        "popular_mrr_top100": float(torch.tensor(popular_top100_mrr).mean()),
        "popular_mrr_top500": float(torch.tensor(popular_top500_mrr).mean()),
    }
    if split_mode == "val":
        result["oversaturated_K50_N5000"] = float(torch.tensor(oversaturated_K50_N5000).mean())
        result["oversaturated_K100_N5000"] = float(torch.tensor(oversaturated_K100_N5000).mean())
        result["oversaturated_K1000_N5000"] = float(torch.tensor(oversaturated_K1000_N5000).mean())

        result["oversaturated_K50_N20000"] = float(torch.tensor(oversaturated_K50_N20000).mean())
        result["oversaturated_K100_N20000"] = float(torch.tensor(oversaturated_K100_N20000).mean())
        result["oversaturated_K1000_N20000"] = float(torch.tensor(oversaturated_K1000_N20000).mean())

        result["oversaturated_K50_N100000"] = float(torch.tensor(oversaturated_K50_N100000).mean())
        result["oversaturated_K100_N100000"] = float(torch.tensor(oversaturated_K100_N100000).mean())
        result["oversaturated_K1000_N100000"] = float(torch.tensor(oversaturated_K1000_N100000).mean())

    return result


# ==========
# ==========
# ==========

# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = args.data
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

POPULARITY_POWER = args.popularity_power
POPULARITY_DECAY = args.popularity_decay
RANDOM_RATIO = args.random_ratio
print(f"RANDOM_RATIO: {RANDOM_RATIO}")
print(f"POPULARITY_DECAY: {POPULARITY_DECAY}")
print(f"POPULARITY_POWER: {POPULARITY_POWER}")
SAMPLING_STRATEGY = args.sampling_strategy
print(f"SAMPLING_STRATEGY: {SAMPLING_STRATEGY}")

MODEL_NAME = "DyRep"
USE_SRC_EMB_IN_MSG = False
USE_DST_EMB_IN_MSG = True
# ==========


validation_popular_negatives = np.load(f"output/popular_neg_samples/{DATA}/popular_negatives_val.npy", mmap_mode="r")
with open(f"output/popular_neg_samples/{DATA}/src_dst_t_to_index_val.pickle", "rb") as handle:
    # mapping from (src, dst, t) to the index of the corresponding negative edge
    validation_src_dst_t_to_index = pickle.load(handle)

test_popular_negatives = np.load(f"output/popular_neg_samples/{DATA}/popular_negatives_test.npy", mmap_mode="r")
with open(f"output/popular_neg_samples/{DATA}/src_dst_t_to_index_test.pickle", "rb") as handle:
    # mapping from (src, dst, t) to the index of the corresponding negative edge
    test_src_dst_t_to_index = pickle.load(handle)

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

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighhorhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = DyRepMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
    memory_updater_type="rnn",
    use_src_emb_in_msg=USE_SRC_EMB_IN_MSG,
    use_dst_emb_in_msg=USE_DST_EMB_IN_MSG,
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM, out_channels=EMB_DIM, msg_dim=data.msg.size(-1), time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}

optimizer = torch.optim.Adam(
    set(model["memory"].parameters()) | set(model["gnn"].parameters()) | set(model["link_pred"].parameters()), lr=LR,
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
results_path = f"{osp.dirname(osp.abspath(__file__))}/saved_results"
if not osp.exists(results_path):
    os.mkdir(results_path)
    print("INFO: Create directory {}".format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f"{results_path}/{SAMPLING_STRATEGY}_{MODEL_NAME}_{DATA}_results.json"

for run_idx in range(NUM_RUNS):
    print("-------------------------------------------------------------------------------")
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f"{osp.dirname(osp.abspath(__file__))}/saved_models/"
    save_model_id = f"{SAMPLING_STRATEGY}_{MODEL_NAME}_{DATA}_{SEED}_{run_idx}"
    early_stopper = EarlyStopMonitor(
        save_model_dir=save_model_dir, save_model_id=save_model_id, tolerance=TOLERANCE, patience=PATIENCE
    )

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

        counter_last_100_000 = SlidingWindowCounter(100_000)
        counter_last_20_000 = SlidingWindowCounter(20_000)
        counter_last_5_000 = SlidingWindowCounter(5_000)

        for dst in train_data.dst[-100_000:]:
            counter_last_100_000.add(dst.item())
            counter_last_20_000.add(dst.item())
            counter_last_5_000.add(dst.item())

        # validation
        start_val = timeit.default_timer()
        val_result = test_one_vs_many(
            val_loader,
            neg_sampler,
            split_mode="val",
            popular_negatives=validation_popular_negatives,
            src_dst_t_to_index=validation_src_dst_t_to_index,
        )
        val_naive_mrr = val_result["naive_mrr"]
        print(f"\tValidation {metric}: {val_naive_mrr: .4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        val_perf_list.append(val_naive_mrr)

        # check for early stopping
        if early_stopper.step_check(val_naive_mrr, model):
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint(model)

    # loading the test negative samples
    dataset.load_test_ns()

    # final testing
    start_test = timeit.default_timer()
    test_result = test_one_vs_many(
        test_loader,
        neg_sampler,
        split_mode="test",
        popular_negatives=test_popular_negatives,
        src_dst_t_to_index=test_src_dst_t_to_index,
    )
    test_naive_mrr = test_result["naive_mrr"]
    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {test_naive_mrr: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    save_results(
        {
            "model": MODEL_NAME,
            "data": DATA,
            "run": run_idx,
            "seed": SEED,
            f"val {metric}": val_perf_list,
            f"test {metric}": test_naive_mrr,
            "val_popular_mrr_top20": val_result["popular_mrr_top20"],
            "val_popular_mrr_top100": val_result["popular_mrr_top100"],
            "val_popular_mrr_top500": val_result["popular_mrr_top500"],
            "test_popular_mrr_top20": test_result["popular_mrr_top20"],
            "test_popular_mrr_top100": test_result["popular_mrr_top100"],
            "test_popular_mrr_top500": test_result["popular_mrr_top500"],
            "oversaturated_K50_N5000": val_result["oversaturated_K50_N5000"],
            "oversaturated_K100_N5000": val_result["oversaturated_K100_N5000"],
            "oversaturated_K1000_N5000": val_result["oversaturated_K1000_N5000"],
            "oversaturated_K50_N20000": val_result["oversaturated_K50_N20000"],
            "oversaturated_K100_N20000": val_result["oversaturated_K100_N20000"],
            "oversaturated_K1000_N20000": val_result["oversaturated_K1000_N20000"],
            "oversaturated_K50_N100000": val_result["oversaturated_K50_N100000"],
            "oversaturated_K100_N100000": val_result["oversaturated_K100_N100000"],
            "oversaturated_K1000_N100000": val_result["oversaturated_K1000_N100000"],
            "test_time": test_time,
            "tot_train_val_time": train_val_time,
        },
        results_filename,
    )

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print("-------------------------------------------------------------------------------")

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
