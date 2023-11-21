import torch
import numpy as np
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import TemporalData
from tqdm import tqdm
from collections import defaultdict
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


class TCF(object):
    r"""
    The temporal collaborative filtering model.
    """

    DEVICE: torch.DeviceObjType = torch.device("cpu")
    # TODO: Make the implementation compatible with CUDA.
    BATCH_SIZE: int = 200

    TRAIN_DATA = None
    VAL_DATA = None
    TEST_DATA = None

    def __init__(
        self,
        dataset_name: str,
        decay: float = 0.95,
        device: torch.DeviceObjType = DEVICE,
        batch_size: int = BATCH_SIZE,
    ):
        self.device = device
        self.DECAY: float = decay
        self.batch_size = batch_size

        # load the dataset
        dataset = PyGLinkPropPredDataset(name=dataset_name, root="data")

        self.TRAIN_DATA, self.VAL_DATA, self.TEST_DATA = self._train_test_split(dataset)

        # Initialize sim_track and bank as 2D dictionaries
        # TODO: this is not memory efficient, use sparse tensors instead

        # Create sparse tensors
        self.sim_track = torch.sparse_coo_tensor(indices=torch.empty((2, 0)), values=torch.empty(0), size=(self.NUM_NODES, self.NUM_NODES))
        self.bank = torch.sparse_coo_tensor(indices=torch.empty((2, 0)), values=torch.empty(0), size=(self.NUM_NODES, self.NUM_NODES))

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
        print("done here")

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

        # iterate over each batch
        for batch in tqdm(self.TRAIN_DATA, desc="Training"):
            for src, dst in zip(batch.src, batch.dst):
                self.bank[src.item()][dst.item()] += 1
                self._update_similarities()

            # decay the bank
            self._decay_bank()

        return

    def _decay_bank(self) -> None:
        """
        Decay the bank by multiplying with the decay factor.
        """
        self.bank *= self.DECAY
        return

    def _update_similarities(self) -> None:
        """
        Update the similarity matrix based on the bank.
        """
        # # TODO: make this efficient
        # for item in range(len(self.sim_track)):
        #     for candid in range(len(self.sim_track[item])):
        #         common_destinations = torch.nonzero((self.bank[item] > 0) & (self.bank[candid] > 0))

        #         # find the similarity between item and candid based on the common destinations.
        #         # multiply the self.bank[item][destination] by self.bank[candid][destination] for each destination in common_destinations and sum them up.
        #         similarity = torch.dot(self.bank[item][common_destinations], self.bank[candid][common_destinations])
        #         self.sim_track[item][candid] = similarity

        # Convert the bank to a sparse tensor
        sparse_bank = self.bank.to_sparse()

        # Compute the dot product of the bank with its transpose
        self.sim_track = torch.sparse.mm(sparse_bank, sparse_bank.t()).to_dense()

        return

    def predict(
        self, src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor
    ) -> torch.Tensor:
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
        for i, (src, dst, ts) in enumerate(zip(sorted_src, sorted_dst, sorted_ts)):
            # get the most similar node to source, max of the similarity matrix over the row corresponding to source
            most_similar_src = torch.argmax(self.sim_track[src.item()])
            # get the most similar node to destination, max of the similarity matrix over the row corresponding to destination
            most_similar_dst = torch.argmax(self.sim_track[dst.item()])

            # if the bank has seen an edge between the most similar source and the destination and the most similar destination and the source, predict as 1, otherwise if one of them, 0.5, otherwise 0
            if (
                self.bank[most_similar_src][dst.item()] > 0
                and self.bank[most_similar_dst][src.item()] > 0
            ):
                probs[i] = 1
            elif (
                self.bank[most_similar_src][dst.item()] > 0
                or self.bank[most_similar_dst][src.item()] > 0
            ):
                probs[i] = 0.5
            else:
                probs[i] = 0

        return probs

    # def evaluate

    # def test

    # def run