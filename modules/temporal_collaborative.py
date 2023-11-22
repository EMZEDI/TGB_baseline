import torch
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
import numpy as np
from time import time

class TCF(object):
    r"""
    The temporal collaborative filtering model.
    """

    DEVICE: torch.DeviceObjType = torch.device("cpu")
    # TODO: Make the implementation compatible with CUDA?
    BATCH_SIZE: int = 200

    TRAIN_DATA = None
    VAL_DATA = None
    TEST_DATA = None

    def __init__(
        self,
        dataset_name: str,
        decay: float = 0.955,
        device: torch.DeviceObjType = DEVICE,
        batch_size: int = BATCH_SIZE,
    ):
        self.dataset_name = dataset_name
        self.device = device
        self.DECAY: float = decay
        self.batch_size = batch_size

        # load the dataset
        dataset = PyGLinkPropPredDataset(name=dataset_name, root="data")
        self.negative_sampler = dataset.negative_sampler
        self.dataset = dataset

        self.TRAIN_DATA, self.VAL_DATA, self.TEST_DATA = self._train_test_split(dataset)

        self.bank = {}

    def _train_test_split(
        self, dataset: PyGLinkPropPredDataset
    ) -> (TemporalDataLoader, TemporalDataLoader, TemporalDataLoader):
        """Split the dataset into train, validation, and test sets.

        Args:
            self (TCF): self Object
            dataset (PyGLinkPropPredDataset): the dataset to split
        """
        # get masks
        train_mask = dataset.train_mask
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask

        data = dataset.get_TemporalData()

        assert dataset.eval_metric == "mrr"

        train_data = data[train_mask]
        val_data = data[val_mask]
        test_data = data[test_mask]

        train_data.t, train_data.src, train_data.dst = self._sort_graph_by_time(
            train_data.t, train_data.src, train_data.dst
        )
        val_data.t, val_data.src, val_data.dst = self._sort_graph_by_time(
            val_data.t, val_data.src, val_data.dst
        )
        test_data.t, test_data.src, test_data.dst = self._sort_graph_by_time(
            test_data.t, test_data.src, test_data.dst
        )

        train_loader = TemporalDataLoader(train_data, batch_size=self.BATCH_SIZE)
        val_loader = TemporalDataLoader(val_data, batch_size=self.BATCH_SIZE)
        test_loader = TemporalDataLoader(test_data, batch_size=self.BATCH_SIZE)

        # find the union of nodes in the train, val, and test sets
        nodes = set()
        nodes.update([a.item() for a in train_data.src])
        nodes.update([a.item() for a in train_data.dst])
        nodes.update([a.item() for a in val_data.src])
        nodes.update([a.item() for a in val_data.dst])
        nodes.update([a.item() for a in test_data.src])
        nodes.update([a.item() for a in test_data.dst])
        self.NUM_NODES = len(nodes)
        # print("done here")

        return train_loader, val_loader, test_loader

    def _sort_graph_by_time(
        self, src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor
    ) -> None:
        """Sort the given graph by time.

        Args:
            src (torch.Tensor): tensor containing the sources
            dst (torch.Tensor): tensor containing the destinations
            ts (torch.Tensor): tensor containing the timestamps
        """

        # sort the graph by time
        sorted_indices = torch.argsort(ts)
        sorted_src = src[sorted_indices]
        sorted_dst = dst[sorted_indices]
        sorted_ts = ts[sorted_indices]

        return sorted_src, sorted_dst, sorted_ts

    def train(self) -> None:
        """
        Train the temporal collaborative filtering model.
        """
        print(
            f"Training the temporal collaborative filtering model for O({self.NUM_NODES}^2) edges."
        )

        self.batch_counter = 1
        # iterate over each batch
        for batch in tqdm(self.TRAIN_DATA, desc="Training"):
            for src, dst in zip(batch.src, batch.dst):
                src_item = src.item()
                dst_item = dst.item()
                if (src_item, dst_item) in self.bank:
                    # TODO: the following might result in a blow up -> normalize
                    # to avoid the time complexity of the bank, we reverse decay it after each positive edge
                    # the logic here might be wrong
                    self.bank[(src_item, dst_item)] += 1
                    self.bank[(src_item, dst_item)] *= (1 / self.DECAY)
                else:
                    self.bank[(src_item, dst_item)] = 1
                # print("Train: updated the bank")

            self.batch_counter += 1
        
        return

    def _get_most_similar_to(self, node: int, is_source: bool = True) -> list[int]:
        """Get the 20 most similar nodes to the given node.

        Args:
            node (int): the item associated with the candid node
            is_source (bool): whether the given node is a source node

        Returns:
            list[int]: list of the 20 most similar nodes
        """

        # Initialize a dictionary to store the similarities for each candidate node
        similarities = {}

        if is_source:
            # Get the scores for the destinations seen by the given node
            scores = {dst: value for (src, dst), value in self.bank.items() if src == node}

            # Get the unique source nodes in self.bank
            unique_nodes = set(src for (src, _), _ in self.bank.items())

            # Iterate over unique source nodes
            for candid_node in unique_nodes:
                # Skip if the candid node is the given node
                if candid_node == node:
                    continue

                # Calculate the dot product of the scores for the given node and the candid node
                dot_product = sum(scores[dst] * self.bank.get((candid_node, dst), 0) for dst in scores if (candid_node, dst) in self.bank)

                # Store the dot product in the dictionary
                similarities[candid_node] = dot_product

        # Add the node itself to the dictionary with a high similarity
        similarities[node] = float('inf')

        # Get the 20 most similar nodes
        most_similar = sorted(similarities, key=similarities.get, reverse=True)[:5000]

        return most_similar
    

    def predict(self, src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability of the given edges.

        Args:
            src (torch.Tensor): tensor containing the sources
            dst (torch.Tensor): tensor containing the destinations
            ts (torch.Tensor): tensor containing the timestamps

        Returns:
            torch.Tensor: tensor containing the probabilities
        """
        # sort the graph by time
        sorted_src, sorted_dst, sorted_ts = self._sort_graph_by_time(src, dst, ts)

        # initialize the probabilities
        probs = torch.zeros(len(src))

        # iterate over each edge
        for i, (src_item, dst_item) in enumerate(zip(sorted_src, sorted_dst)):
            # find the 20 most similar nodes to the given source
            most_similar_nodes = self._get_most_similar_to(src_item.item())

            # count the number of most similar nodes that have had a link to the destination
            count = sum(1 for node in most_similar_nodes if (node, dst_item.item()) in self.bank)

            # if more than 1/3 of the most similar nodes have had a link to the destination, predict as 1, otherwise predict as 0
            # TODO: the threshold might be low - adjust based on surprise rate
            if count > 100:
                probs[i] = 1

        return probs

    def val_test(self, split_mode: str="test") -> None:
        """
        Test the temporal collaborative filtering model.

        Args:
            negative_sampler (NegativeSampler): the negative sampler
            split_mode (str): the split mode. Defaults to "test"
        """
        if split_mode == "test":
            print(f"Testing the temporal collaborative filtering model.")
            data = self.TEST_DATA
            self.dataset.load_test_ns()
        else:
            print(f"Validating the temporal collaborative filtering model.")
            data = self.VAL_DATA
            self.dataset.load_val_ns()


        evaluator = Evaluator(name=self.dataset_name)
        naive_mrr = []

        
        # iterate over each batch
        for pos_batch in tqdm(data, desc="Testing"):
            tmp_counter = 1
            pos_src, pos_dst, pos_t = pos_batch.src, pos_batch.dst, pos_batch.t
            neg_batch_list = self.negative_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

            for idx, negative_candidates in tqdm(enumerate(neg_batch_list)):
                tmp_counter += 1
                if tmp_counter > 2:
                    break
                # query_src = torch.Tensor([int(pos_src[idx]) for _ in range(len(negative_candidates) + 1)])
                # query_dst = torch.Tensor(np.concatenate([np.array([int(pos_dst[idx])]), negative_candidates]))
                query_src = torch.Tensor([int(pos_src[idx]) for _ in range(min(3, len(negative_candidates)) + 1)])
                query_dst = torch.Tensor(np.concatenate([np.array([int(pos_dst[idx])]), negative_candidates[:3]]))
                query_ts = torch.Tensor([int(pos_t[idx]) for _ in range(min(3, len(negative_candidates)) + 1)])
                # print(query_dst)
                # query_ts = torch.Tensor([int(pos_t[idx]) for _ in range(len(negative_candidates) + 1)])

                # print(len(query_src), len(query_dst), len(query_ts))
                scores = self.predict(query_src, query_dst, query_ts)
                print("computed the scores for a batch")
                print(f"prediction score is {scores[0]}, and the wrong prediction score is {scores[1], scores[2]}")
                input_dict = {
                        "y_pred_pos": np.array([scores[0]]),
                        "y_pred_neg": np.array(scores[1:]),
                        # TODO: change the following to use the evaluator's metric
                        "eval_metric": ["mrr"],
                    }
                naive_mrr.append(evaluator.eval(input_dict)["mrr"])

            # update the bank after each positive batch has been seen
            for src, dst in zip(pos_batch.src, pos_batch.dst):
                src_item = src.item()
                dst_item = dst.item()
                if (src_item, dst_item) in self.bank:
                    self.bank[(src_item, dst_item)] += 1
                    # TODO: the following might result in a blow up -> normalize
                    # to avoid the time complexity of the bank, we reverse decay it after each positive edge
                    self.bank[(src_item, dst_item)] *= (1 / self.DECAY)
                else:
                    self.bank[(src_item, dst_item)] = 1
            
            # self.batch_counter += 1

        naive_mrr = float(torch.tensor(naive_mrr).mean())
        print(f"Naive MRR: {naive_mrr}")
        return naive_mrr

    # def run