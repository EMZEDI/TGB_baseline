import torch
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
import numpy as np

class TCF(object):
    r"""
    The temporal collaborative filtering model.
    """

    DEVICE: torch.DeviceObjType = torch.device("cpu")
    # TODO: Make the implementation compatible with CUDA?
    BATCH_SIZE: int = 5000

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

    def _dict_to_sparse(self, dict_obj):
        indices = list(zip(*dict_obj.keys()))
        values = list(dict_obj.values())
        return torch.sparse_coo_tensor(indices=indices, values=values, size=(self.NUM_NODES, self.NUM_NODES))

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
                    self.bank[(src_item, dst_item)] += 1
                    # TODO: the following might result in a blow up -> normalize
                    # to avoid the time complexity of the bank, we reverse decay it after each positive edge
                    self.bank[(src_item, dst_item)] *= (1 / self.DECAY)**self.batch_counter
                else:
                    self.bank[(src_item, dst_item)] = 1
                # print("Train: updated the bank")

            self.batch_counter += 1
        
        return

    def _get_most_similar_to(self, node: int, is_source: bool = True) -> int:
        """Get the most similar node to the given node.

        Args:
            node (int): the item associated with the candid node
            is_source (bool, optional): whether the given node is a source or a destination. Defaults to True.
        """

        # Initialize the maximum similarity and the most similar node
        max_similarity = -1
        most_similar = -1

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
                # print("computed the dot product")
                # If the dot product is higher than the current maximum similarity
                if dot_product > max_similarity:
                    # Update the maximum similarity and the most similar node
                    max_similarity = dot_product
                    most_similar = candid_node

        else:
            # Get the scores for the sources seen by the given node
            scores = {src: value for (src, dst), value in self.bank.items() if dst == node}

            # Get the unique destination nodes in self.bank
            unique_nodes = set(dst for (_, dst), _ in self.bank.items())

            # Iterate over unique destination nodes
            for candid_node in unique_nodes:
                # Skip if the candid node is the given node
                if candid_node == node:
                    continue

                # Calculate the dot product of the scores for the given node and the candid node
                dot_product = sum(scores[src] * self.bank.get((src, candid_node), 0) for src in scores if (src, candid_node) in self.bank)
                # print("computed the dot product")
                # If the dot product is higher than the current maximum similarity
                if dot_product > max_similarity:
                    # Update the maximum similarity and the most similar node
                    max_similarity = dot_product
                    most_similar = candid_node

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
            # find the most similar source to the given source and the most similar destination to the given destination
            most_similar_src = self._get_most_similar_to(src_item.item())
            # most_similar_dst = self._get_most_similar_to(dst_item.item(), is_source=False)

            # if the bank has seen an edge between the most similar source and the destination and the most similar destination and the source, predict as 1, otherwise if one of them, 0.5, otherwise 0
            # if (
            #     (most_similar_src, dst_item.item()) in self.bank
            #     and (src_item.item(), most_similar_dst) in self.bank
            # ):
            #     probs[i] = 1
            # elif (
            #     (most_similar_src, dst_item.item()) in self.bank
            #     or (src_item.item(), most_similar_dst) in self.bank
            # ):
            #     probs[i] = 0.5
            # else:
            #     probs[i] = 0

            probs[i] = int((most_similar_src, dst_item.item()) in self.bank)

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
            pos_src, pos_dst, pos_t = pos_batch.src, pos_batch.dst, pos_batch.t
            neg_batch_list = self.negative_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

            for idx, negative_candidates in tqdm(enumerate(neg_batch_list)):
                print("entered the loop for a batch")
                query_src = torch.Tensor([int(pos_src[idx]) for _ in range(len(negative_candidates) + 1)])
                query_dst = torch.Tensor(np.concatenate([np.array([int(pos_dst[idx])]), negative_candidates]))
                query_ts = torch.Tensor([int(pos_t[idx]) for _ in range(len(negative_candidates) + 1)])
                print(len(query_src), len(query_dst), len(query_ts))
                scores = self.predict(query_src, query_dst, query_ts)
                print("computed the scores for a batch")
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
                    self.bank[(src_item, dst_item)] *= (1 / self.DECAY)**self.batch_counter
                else:
                    self.bank[(src_item, dst_item)] = 1
            
            self.batch_counter += 1

        naive_mrr = float(torch.tensor(naive_mrr).mean())
        print(f"Naive MRR: {naive_mrr}")
        return naive_mrr

    # def run