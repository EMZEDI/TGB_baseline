from time import time
import torch
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
import numpy as np
import scipy as sp

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
        decay: float = 0.9,   # TODO: call it growth - will depend on the surprise rate of the dataset or the derived metric \sigma - technically, the lower the decay factor, the higher the surprise rate
        device: torch.DeviceObjType = DEVICE,
        batch_size: int = 200,
        k: int = 5,
        factor: int = 0.5,
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

        # create scipy lil matrix for the similarities between nodes
        self.similarities = sp.sparse.lil_matrix((self.NUM_SRC, self.NUM_DST))
        self.sparse_bank = sp.sparse.lil_matrix((self.NUM_SRC, self.NUM_DST))

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

        val_data.src, val_data.dst, val_data.t = self._sort_graph_by_time(
            val_data.src, val_data.dst, val_data.t
        )

        test_data.src, test_data.dst, test_data.t = self._sort_graph_by_time(
            test_data.src, test_data.dst, test_data.t
        )

        train_loader = TemporalDataLoader(train_data, batch_size=self.batch_size)
        val_loader = TemporalDataLoader(val_data, batch_size=self.batch_size)
        test_loader = TemporalDataLoader(test_data, batch_size=self.batch_size)

        # find the union of source nodes in train, test, val and assign self.NUM_SRC and then destinations self.NUM_DST in all data
        sources = set(train_data.src.tolist() + val_data.src.tolist() + test_data.src.tolist())
        destinations = set(train_data.dst.tolist() + val_data.dst.tolist() + test_data.dst.tolist())
        nodes = sources.union(destinations)
        self.NUM_SRC = len(sources)
        self.NUM_DST = len(destinations)
        self.NUM_NODES = len(nodes)

        self.source_id_to_index = {source_id: index for index, source_id in enumerate(sorted(list(sources)))}
        self.destination_id_to_index = {destination_id: index for index, destination_id in enumerate(sorted(list(destinations)))}
        self.index_to_source_id = {index: source_id for source_id, index in self.source_id_to_index.items()}
        self.index_to_destination_id = {index: destination_id for destination_id, index in self.destination_id_to_index.items()}

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

            for src, dst in zip(batch.src, batch.dst):
                src_index = self.source_id_to_index[src.item()]
                dst_index = self.destination_id_to_index[dst.item()]

                # update the weight of the edge in the bank
                if self.sparse_bank[(src_index, dst_index)] == 0:
                    self.bank[(src_index, dst_index)] = 1
                    self.sparse_bank[(src_index, dst_index)] = 1
                else:
                    self.bank[(src_index, dst_index)] += 1
                    self.sparse_bank[(src_index, dst_index)] += 1
                    
            # multiply the sparse bank by the self.DECAY factor
            tmp = self.sparse_bank.tocsr()
            self.sparse_bank = (tmp * self.DECAY).tolil()

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
        source_index = self.source_id_to_index[source]

        # get the similarity scores for the given source - source is a row TODO: get non zero ones
        similarity_scores = self.similarities[source_index].toarray()[0]

        # get the indices of the nodes with non-zero similarity
        non_zero_indices = np.nonzero(similarity_scores)[0]

        # sort the non-zero indices by their similarity scores in descending order
        sorted_indices = non_zero_indices[np.argsort(similarity_scores[non_zero_indices])[::-1]]

        # get the first K indices, or all of them if there are less than K - these indices are not node ids, but indices in the sparse matrix
        closest_to_source = sorted_indices[:min(self.K, len(sorted_indices))]

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
        # initialize the probabilities
        probs = np.zeros(len(src))

        # iterate over each edge
        # TODO: vectorize this
        for (i, dst_item) in enumerate(dst):
            # get the links of the most similar nodes to the destination
            destination = int(self.destination_id_to_index[dst_item.item()])

            # TODO: get non zero ones only
            if len(most_similars) == 1:
                similars = most_similars[0]
                links = np.array(self.sparse_bank[similars, destination])

            else:
                similars = most_similars
                links = self.sparse_bank[similars, destination].toarray()

            # count the number of most similar nodes that have had a link to the destination
            count = np.count_nonzero(links)

            # if more than 1/3 of the most similar nodes have had a link to the destination, predict as 1, otherwise predict as 0
            if count >= (len(most_similars) * (self.decision_factor)):
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
            pos_src, pos_dst, pos_t = pos_batch.src, pos_batch.dst, pos_batch.t
            neg_batch_list = self.negative_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)


            for idx, negative_candidates in tqdm(enumerate(neg_batch_list)):
                source_id = int(pos_src[idx])
                # get the k highest similarity scores for the given source - source is a row

                closest_to_source = self.get_similars(source_id)

                query_src = torch.Tensor([int(pos_src[idx]) for _ in range(len(negative_candidates) + 1)])
                query_dst = torch.Tensor(np.concatenate([np.array([int(pos_dst[idx])]), negative_candidates]))
                query_ts = torch.Tensor([int(pos_t[idx]) for _ in range(len(negative_candidates) + 1)])

                # compute the scores
                scores = self.predict_given_sim(query_src, query_dst, query_ts, closest_to_source)

                input_dict = {
                        "y_pred_pos": np.array([scores[0]]),
                        "y_pred_neg": np.array(scores[1:]),
                        # TODO: change the following to use the evaluator's metric
                        "eval_metric": ["mrr"],
                    }
                naive_mrr.append(evaluator.eval(input_dict)["mrr"])

            for src, dst in zip(pos_batch.src, pos_batch.dst):
                src_index = self.source_id_to_index[src.item()]
                dst_index = self.destination_id_to_index[dst.item()]

                # update the weight of the edge in the bank
                if self.sparse_bank[(src_index, dst_index)] == 0:
                    self.bank[(src_index, dst_index)] = 1
                    self.sparse_bank[(src_index, dst_index)] = 1
                else:
                    self.bank[(src_index, dst_index)] += 1
                    self.sparse_bank[(src_index, dst_index)] += 1
            
            start = time()
            # multiply the sparse bank by the self.DECAY factor
            mat = self.sparse_bank.tocsr()
            mat_transpose = mat.transpose()
            self.similarities = (mat @ mat_transpose).tolil()
            self.sparse_bank = (mat * self.DECAY).tolil()
            print(f"Time to update similarities bank: {time() - start}")

        naive_mrr = float(torch.tensor(naive_mrr).mean())
        print(f"Naive MRR: {naive_mrr}")
        return naive_mrr

    # def run