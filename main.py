import argparse
import pickle

import numpy as np

from coco_dataset import Dataset
from graph_cut_runner import main_graph_cut_segmentation, separate_binary_from_multi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="one / all", default="one")
    parser.add_argument(
        "--algorithm", help="binary or alpha-expansion", default="alpha-expansion"
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--unary-cost",
        default="normal",
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lmbda",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--weight-binary",
        type=int,
        default=1000,
    )
    args = parser.parse_args()
    dataset = Dataset()
    if args.mode == "one":
        results = main_graph_cut_segmentation(
            dataset,
            args.image_index,
            args.algorithm,
            args.unary_cost,
            args.sigma,
            args.lmbda,
            args.weight_binary,
            plot_densities=True,
            plot_result=True,
            load_precomputed=False,
            save_path=f"precomputed_annotation/image_{args.image_index}.pkl",
            auto_select_algo=True,
        )
    elif args.mode == "all":
        max_examples = 25
        examples_index = list(range(len(dataset.dataset)))
        np.random.shuffle(examples_index)
        results = {}
        for i in examples_index[:max_examples]:
            image_index = i
            results[i] = main_graph_cut_segmentation(
                dataset,
                image_index,
                args.algorithm,
                args.unary_cost,
                args.sigma,
                args.lmbda,
                args.weight_binary,
                plot_densities=False,
                plot_result=False,
                load_precomputed=True,
                save_path=f"precomputed_annotation/image_{image_index}.pkl",
                auto_select_algo=True,
            )
        results_binary, results_alpha_expansion = separate_binary_from_multi(
            results, dataset.n_labels_list
        )
        with open("results_2.pkl", "wb") as f:
            pickle.dump(results, f)
