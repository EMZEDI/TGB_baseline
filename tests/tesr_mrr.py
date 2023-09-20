import numpy as np
import torch

from tgb.linkproppred.evaluate import Evaluator


def test_the_same_scores_20_negatives():
    evaluator = Evaluator(name="tgbl-comment")

    num_negatives = 20
    input_dict = {
        "y_pred_pos": np.array([1.0]),
        "y_pred_neg": np.array([1.0] * num_negatives),
        "eval_metric": ["mrr"],
    }
    expected_mrr = 0.1735
    assert np.isclose(evaluator.eval(input_dict)["mrr"], expected_mrr, atol=1e-4)


def test_the_same_scores_100_negatives():
    evaluator = Evaluator(name="tgbl-comment")

    num_negatives = 100
    input_dict = {
        "y_pred_pos": np.array([1.0]),
        "y_pred_neg": np.array([1.0] * num_negatives),
        "eval_metric": ["mrr"],
    }
    expected_mrr = 0.0514
    assert np.isclose(evaluator.eval(input_dict)["mrr"], expected_mrr, atol=1e-4)


def test_two_the_same_scores():
    evaluator = Evaluator(name="tgbl-comment")

    input_dict = {
        "y_pred_pos": np.array([1.0]),
        "y_pred_neg": np.array([0.8, 0.7, 0.5, 1.0]),
        "eval_metric": ["mrr"],
    }

    expected_mrr = 0.75
    assert np.isclose(evaluator.eval(input_dict)["mrr"], expected_mrr, atol=1e-4)


def test_best_score():
    evaluator = Evaluator(name="tgbl-comment")

    input_dict = {
        "y_pred_pos": np.array([1.0]),
        "y_pred_neg": np.array([0.0] * 5),
        "eval_metric": ["mrr"],
    }
    expected_mrr = 1.0
    assert np.isclose(evaluator.eval(input_dict)["mrr"], expected_mrr, atol=1e-4)


def test_second_score():
    evaluator = Evaluator(name="tgbl-comment")

    input_dict = {
        "y_pred_pos": np.array([0.99]),
        "y_pred_neg": np.array([0.9, 0.7, 1.0, 0.1, 0.2]),
        "eval_metric": ["mrr"],
    }
    expected_mrr = 0.5
    assert np.isclose(evaluator.eval(input_dict)["mrr"], expected_mrr, atol=1e-4)


def test_third_score():
    evaluator = Evaluator(name="tgbl-comment")

    input_dict = {
        "y_pred_pos": np.array([0.9]),
        "y_pred_neg": np.array([0.99, 0.7, 1.0, 0.1, 0.2]),
        "eval_metric": ["mrr"],
    }
    expected_mrr = 0.3333
    assert np.isclose(evaluator.eval(input_dict)["mrr"], expected_mrr, atol=1e-4)


def test_tenth_score():
    evaluator = Evaluator(name="tgbl-comment")

    input_dict = {
        "y_pred_pos": np.array([0.1]),
        "y_pred_neg": np.array([0.99, 0.7, 1.0, 0.9, 0.2, 0.3, 0.01, 0.5, 0.6, 0.8]),
        "eval_metric": ["mrr"],
    }
    expected_mrr = 0.1
    assert np.isclose(evaluator.eval(input_dict)["mrr"], expected_mrr, atol=1e-4)
