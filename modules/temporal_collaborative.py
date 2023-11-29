from time import time
import torch
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import logging

class TCF(object):
    r"""
    The temporal collaborative filtering model.
    """

    # TODO: Make the implementation compatible with CUDA?
    DEVICE: torch.DeviceObjType = torch.device("cpu")
    
    TRAIN_DATA = None
    VAL_DATA = None
    TEST_DATA = None

    def __init__(
        self,
        dataset_name: str,
        decay: float = 0.9, 
        device: torch.DeviceObjType = DEVICE,
        batch_size: int = 200,
        k: int = 5,
        factor: int = 0.5,
    ):
        self.logger = logging.getLogger(__name__)
        # setup logging to file 
        logging.basicConfig(filename=f"logs/{dataset_name}_tcf.log",
                            filemode='w',
                            format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.DEBUG)
        self.decision_factor = factor
        self.K = k
        self.dataset_name = dataset_name
        self.device = device
            
        self.DECAY: float = decay
        self.batch_size = batch_size        

        # load the dataset
        dataset = PyGLinkPropPredDataset(name=dataset_name, root="data")
        self.negative_sampler = dataset.negative_sampler
        self.dataset = dataset

        self.TRAIN_DATA, self.VAL_DATA, self.TEST_DATA = self._train_test_split(dataset)

        # create scipy lil matrix for the similarities between nodes - use coo matrix 
        self.bank_row = []
        self.bank_col = []
        self.bank_data = []

        self.bank = {}
        self.sparse_bank = None
        self.similarities = None


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

        train_data.src, train_data.dst, train_data.t = self._sort_graph_by_time(
            train_data.src, train_data.dst, train_data.t
        )
        # assert train data is sorted by time
        assert (train_data.t == torch.sort(train_data.t).values).all(), "The train data is not sorted by time"

        val_data.src, val_data.dst, val_data.t = self._sort_graph_by_time(
            val_data.src, val_data.dst, val_data.t
        )

        test_data.src, test_data.dst, test_data.t = self._sort_graph_by_time(
            test_data.src, test_data.dst, test_data.t
        )

        train_loader = TemporalDataLoader(train_data, batch_size=self.batch_size)
        val_loader = TemporalDataLoader(val_data, batch_size=self.batch_size)
        test_loader = TemporalDataLoader(test_data, batch_size=self.batch_size)

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
        torch_sorted = torch.sort(ts).values
        # print(f"real sorted is {torch_sorted}")

        # assert that all elements in sorted_ts are equal to the corresponding elements in torch_sorted
        assert (sorted_ts == torch_sorted).all(), "The tensors are not equal"

        return sorted_src, sorted_dst, sorted_ts

    def train(self) -> None:
        """
        Train the temporal collaborative filtering model.
        """

        # iterate over each batch
        for batch in tqdm(self.TRAIN_DATA, desc="Training"):

            for src, dst in zip(batch.src, batch.dst):
                src_item = src.item()
                dst_item = dst.item()

                if (src_item, dst_item) not in self.bank:
                    self.bank_row.append(src_item)
                    self.bank_col.append(dst_item)
                    self.bank_data.append(1)
                    self.bank[(src_item, dst_item)] = len(self.bank_data) - 1
                else:
                    self.bank_data[self.bank[(src_item, dst_item)]] += 1
                    
            self.bank_data = [self.DECAY * x for x in self.bank_data]

        self.sparse_bank = coo_matrix((self.bank_data, (self.bank_row, self.bank_col)))
        mat = self.sparse_bank.tocsr()
        mat_transpose = mat.transpose()
        self.similarities = (mat @ mat_transpose).tolil()
        return
        
    def get_similars(self, source: int) -> np.ndarray:
        """
        Get the most similar nodes to the given source.

        Args:
            source (int): the source node

        Returns:
            np.ndarray: the most similar nodes to the given source
        """
        try:

            # get the column indices of the nodes with non-zero similarity
            non_zero_indices = self.similarities[source].nonzero()[1]

            # get the non-zero similarity scores
            non_zero_values = self.similarities[source].data

            # print(f"non_zero_indices: {non_zero_indices}")
            # print(f"non_zero_values: {non_zero_values}")

            # if non_zero_values is a scalar (only 1 neighbour), convert it back to a 1D array
            if np.isscalar(non_zero_values):
                non_zero_values = np.array([non_zero_values])

            # sort the non-zero indices by their similarity scores in descending order
            sorted_indices = non_zero_indices[np.argsort(non_zero_values[0])[::-1]]

            # get the first K indices, or all of them if there are less than K
            closest_to_source = sorted_indices[:min(self.K, len(sorted_indices))]

        except IndexError:
            # the given node is totally new - TODO: infer from the destination's popularity
            closest_to_source = np.array([])

        return closest_to_source

    def predict_given_sim(self, src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor, most_similars: np.ndarray) -> np.ndarray:
        """
        Predict the probability of the given edges.

        Args:
            src (torch.Tensor): tensor containing the sources
            dst (torch.Tensor): tensor containing the destinations
            ts (torch.Tensor): tensor containing the timestamps
            most_similars (np.ndarray): the most similar nodes to the given source

        Returns:
            np.ndarray: tensor containing the probabilities
        """
        bank = self.sparse_bank.tolil()
        # convert the destination tensor to a numpy array
        destination_indices = dst.numpy()

        # count the number of most similar nodes that have had a link to each destination
        counts = np.zeros(len(destination_indices))
        if len(most_similars) == 0:
            # give 0.5 probability to all edges
            return np.array([0.5] * len(destination_indices))
        
        # Iterate over the most similar nodes
        for similar_node in most_similars:
            # Get the destinations that the similar node has had a link with
            _, linked_destinations = bank[similar_node, :].nonzero()
            
            # Check which destinations in destination_indices are in linked_destinations
            is_destination_linked = np.in1d(destination_indices, linked_destinations)
            
            # Increment the count for the destinations that the similar node has had a link with
            counts[is_destination_linked] += 1

        # Calculate the threshold for deciding whether to predict 1 or 0
        threshold = len(most_similars) * self.decision_factor

        # Predict as 1 if the count for a destination is greater than or equal to the threshold, otherwise predict as 0
        probs = (counts >= threshold).astype(np.float32)

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
                source_id = int(pos_src[idx])
                # get the k highest similarity scores for the given source - source is a row
                start = time()
                # TODO: the most time consuming part of the code
                closest_to_source = self.get_similars(source_id)
                # log to file the time
                # self.logger.debug(f"Time to get similars: {time() - start}")

                query_src = torch.Tensor([int(pos_src[idx]) for _ in range(len(negative_candidates) + 1)])
                query_dst = torch.Tensor(np.concatenate([np.array([int(pos_dst[idx])]), negative_candidates]))
                query_ts = torch.Tensor([int(pos_t[idx]) for _ in range(len(negative_candidates) + 1)])

                # compute the scores
                start = time()
                scores = self.predict_given_sim(query_src, query_dst, query_ts, closest_to_source)
                # log to file the time
                # self.logger.debug(f"Time to predict given sim: {time() - start}")

                input_dict = {
                        "y_pred_pos": np.array([scores[0]]),
                        "y_pred_neg": np.array(scores[1:]),
                        # TODO: change the following to use the evaluator's metric
                        "eval_metric": ["mrr"],
                    }
                naive_mrr.append(evaluator.eval(input_dict)["mrr"])

            start = time()
            for src, dst in zip(pos_batch.src, pos_batch.dst):
                src_item = src.item()
                dst_item = dst.item()

                if (src_item, dst_item) not in self.bank:
                    self.bank_row.append(src_item)
                    self.bank_col.append(dst_item)
                    self.bank_data.append(1)
                    self.bank[(src_item, dst_item)] = len(self.bank_data) - 1
                else:
                    self.bank_data[self.bank[(src_item, dst_item)]] += 1
            # log to file the time
            # self.logger.debug(f"Time to update bank: {time() - start}")
            
            start = time()
            self.bank_data = [self.DECAY * x for x in self.bank_data]
            
            # multiply the sparse bank by the self.DECAY factor
            self.sparse_bank = coo_matrix((self.bank_data, (self.bank_row, self.bank_col)))
            mat = self.sparse_bank.tocsr()
            mat_transpose = mat.transpose()
            self.similarities = (mat @ mat_transpose).tolil()
            # log to file the time
            # self.logger.debug(f"Time to update similarities: {time() - start}")

        naive_mrr = float(torch.tensor(naive_mrr).mean())
        print(f"Naive MRR: {naive_mrr}")
        return naive_mrr

    # def run