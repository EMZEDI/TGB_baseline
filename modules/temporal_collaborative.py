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
    K: int = 4
    decision_factor: int = 50

    TRAIN_DATA = None
    VAL_DATA = None
    TEST_DATA = None

    def __init__(
        self,
        dataset_name: str,
        decay: float = 0.5,   # TODO: call it growth - will depend on the surprise rate of the dataset or the derived metric \sigma - technically, the lower the decay factor, the higher the surprise rate
        device: torch.DeviceObjType = DEVICE,
        batch_size: int = BATCH_SIZE,
        k: int = K,
        factor: int = decision_factor,
    ):
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

        train_data.src, train_data.dst, train_data.t = self._sort_graph_by_time(
            train_data.src, train_data.dst, train_data.t
        )
        # assert train data is sorted by time
        assert (train_data.t == torch.sort(train_data.t).values).all(), "The train data is not sorted by time"

        # save the min train time and the train time range
        self.min_train_time: int = min(train_data.t)
        self.train_time_range: int = max(train_data.t) - self.min_train_time

        val_data.src, val_data.dst, val_data.t = self._sort_graph_by_time(
            val_data.src, val_data.dst, val_data.t
        )

        self.min_val_time = min(val_data.t)
        self.val_time_range = max(val_data.t) - self.min_val_time

        test_data.src, test_data.dst, test_data.t = self._sort_graph_by_time(
            test_data.src, test_data.dst, test_data.t
        )

        self.min_test_time = min(test_data.t)
        self.test_time_range = max(test_data.t) - self.min_test_time
        # sum up the time ranges
        self.time_range = self.train_time_range + self.val_time_range + self.test_time_range

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
        print(
            f"Training the temporal collaborative filtering model for O({self.NUM_NODES}^2) edges."
        )

        # iterate over each batch
        for batch in tqdm(self.TRAIN_DATA, desc="Training"):
            # TODO: change the following to normalize from time 0 to the whole time range till the end of test
            normalized_ts = (batch.t - self.min_train_time) / self.time_range
            # calculate the recency factor for the current batch - between 1 and 2 (not inclusive)
            recency_factor = np.finfo(float).eps + normalized_ts[0].item()

            for src, dst in zip(batch.src, batch.dst):
                src_item = src.item()
                dst_item = dst.item()

                # update the weight of the edge in the bank
                old_weight = self.bank.get((src_item, dst_item), 0)
                if self.DECAY > 1:
                    new_weight = (1 + old_weight)
                else:
                    new_weight = (1 + old_weight) * (recency_factor / self.DECAY)
                self.bank[(src_item, dst_item)] = new_weight

        return

    def _get_most_similar_to(self, node: int, is_source: bool = True) -> np.ndarray:
        """Get the K most similar nodes to the given node.

        Args:
            node (int): the item associated with the candid node
            is_source (bool): whether the given node is a source node

        Returns:
            np.ndarray: list of the 20 most similar nodes
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

                # Calculate the dot product of the scores for the given node and the candid node
                dot_product = sum(scores[dst] * self.bank.get((candid_node, dst), 0) for dst in scores if (candid_node, dst) in self.bank)

                # Store the dot product in the dictionary
                similarities[candid_node] = dot_product

        # Get the 20 most similar nodes
        most_similar = np.array(sorted(similarities, key=similarities.get, reverse=True)[:self.K])

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

        # initialize the probabilities
        probs = np.zeros(len(src))

        # iterate over each edge
        for (i, dst_item) in enumerate(dst):

            # count the number of most similar nodes that have had a link to the destination
            count = sum(1 for node in most_similars if (node, dst_item.item()) in self.bank)

            # if more than 1/3 of the most similar nodes have had a link to the destination, predict as 1, otherwise predict as 0
            # TODO: the threshold might be low - adjust based on surprise rate
            if count >= (len(most_similars) * (self.decision_factor / 100)):
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
            # compute how much time an iteration takes
            pos_src, pos_dst, pos_t = pos_batch.src, pos_batch.dst, pos_batch.t
            neg_batch_list = self.negative_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
            # counter = 0

            for idx, negative_candidates in tqdm(enumerate(neg_batch_list)):
                # if counter > 50:
                #     break
                # counter += 1
                # start = time()
                source_id = int(pos_src[idx])
                closest_to_source = self._get_most_similar_to(source_id) 
                # print(f"getting the most similar nodes took {time() - start} seconds")
                # start = time()
                query_src = torch.Tensor([int(pos_src[idx]) for _ in range(len(negative_candidates) + 1)])
                query_dst = torch.Tensor(np.concatenate([np.array([int(pos_dst[idx])]), negative_candidates]))
                query_ts = torch.Tensor([int(pos_t[idx]) for _ in range(len(negative_candidates) + 1)])
                # print(f"creating the query took {time() - start} seconds")

                # print(len(query_src), len(query_dst), len(query_ts))
                # compute how much time getting the scores takes
                # start = time()
                scores = self.predict_given_sim(query_src, query_dst, query_ts, closest_to_source)
                # print(f"getting the scores took {time() - start} seconds")
                # print(f"getting the scores took {time() - start} seconds")
                # print("computed the scores for a batch")
                # print(f"prediction score is {scores[0]}, and the wrong prediction score is {scores[1], scores[2]}")
                # start = time()
                input_dict = {
                        "y_pred_pos": np.array([scores[0]]),
                        "y_pred_neg": np.array(scores[1:]),
                        # TODO: change the following to use the evaluator's metric
                        "eval_metric": ["mrr"],
                    }
                naive_mrr.append(evaluator.eval(input_dict)["mrr"])
                # print(f"evaluating the scores took {time() - start} seconds")

            normalized_ts = (pos_batch.t - self.min_train_time) / self.time_range
            # calculate the recency factor for the current batch - between 1 and 2 (not inclusive)
            recency_factor = np.finfo(float).eps + normalized_ts[0].item()

            for src, dst in zip(pos_batch.src, pos_batch.dst):
                src_item = src.item()
                dst_item = dst.item()

                # update the weight of the edge in the bank
                old_weight = self.bank.get((src_item, dst_item), 0)
                if self.DECAY > 1:
                    new_weight = (1 + old_weight)
                else:
                    new_weight = (1 + old_weight) * (recency_factor / self.DECAY)
                self.bank[(src_item, dst_item)] = new_weight


        naive_mrr = float(torch.tensor(naive_mrr).mean())
        print(f"Naive MRR: {naive_mrr}")
        return naive_mrr

    # def run