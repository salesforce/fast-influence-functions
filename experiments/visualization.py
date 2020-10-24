import torch
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.axes._subplots import Subplot
# from graph_tool.draw import graph_draw
# from joblib import Parallel, delayed

from typing import List, Dict, Tuple, Optional, Union, Callable
from influence_utils import faiss_utils
from influence_utils import parallel
from influence_utils import nn_influence_utils
from experiments.visualization_utils import (
    get_circle_coordinates,
    get_within_circle_constraint,
    # distance_to_points_on_circle,
    # distance_to_points_within_circle,
    distance_to_points_within_circle_vectorized)
from experiments import constants
from experiments import misc_utils
from experiments import remote_utils
from experiments import influence_helpers
from experiments.hans_utils import HansHelper
from transformers import Trainer, TrainingArguments


try:
    import graph_tool as gt
    gt_Graph_t = gt.Graph
except ModuleNotFoundError:
    # We do not need `graph_tool` unless
    # visualization is to be created
    gt = None
    gt_Graph_t = "gt.Graph"

DEFAULT_KNN_K = 1000
DEFAULT_TRAIN_VERTEX_COLOR = 0
DEFAULT_TRAIN_VERTEX_RADIUS = 2
DEFAULT_EVAL_VERTEX_COLORS_BASE = 2
DEFAULT_EVAL_VERTEX_RADIUS = 3
DEFAULT_HELPFUL_EDGE_COLOR = 0
DEFAULT_HARMFUL_EDGE_COLOR = 1
DEFAULT_TRAIN_VERTEX_SIZE = 3


def main(
    train_task_name: str,
    eval_task_name: str,
    num_eval_to_collect: int,
    use_parallel: bool = True,
    hans_heuristic: Optional[str] = None,
    trained_on_task_name: Optional[str] = None,
) -> List[Dict[int, float]]:

    if train_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    if eval_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    if trained_on_task_name is None:
        # The task the model was trained on
        # can be different from `train_task_name`
        # which is used to determine on which the
        # influence values will be computed.
        trained_on_task_name = train_task_name

    if trained_on_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    # `trained_on_task_name` determines the model to load
    if trained_on_task_name in ["mnli-2"]:
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.MNLI2_MODEL_PATH)

    if trained_on_task_name in ["hans"]:
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.HANS_MODEL_PATH)

    train_dataset, _ = misc_utils.create_datasets(
        task_name=train_task_name,
        tokenizer=tokenizer)

    _, eval_dataset = misc_utils.create_datasets(
        task_name=eval_task_name,
        tokenizer=tokenizer)

    faiss_index = influence_helpers.load_faiss_index(
        trained_on_task_name=trained_on_task_name,
        train_task_name=train_task_name)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if eval_task_name in ["mnli-2"]:
        eval_instance_data_loader = misc_utils.get_dataloader(
            dataset=eval_dataset,
            batch_size=1,
            random=False)

    if eval_task_name in ["hans"]:
        if hans_heuristic is None:
            raise ValueError("`hans_heuristic` cannot be None for now")

        hans_helper = HansHelper(
            hans_train_dataset=None,
            hans_eval_dataset=eval_dataset)

        _, eval_instance_data_loader = hans_helper.get_dataset_and_dataloader_of_heuristic(
            mode="eval",
            heuristic=hans_heuristic,
            batch_size=1,
            random=False)

    # Data-points where the model got wrong
    wrong_input_collections = []
    for i, test_inputs in enumerate(eval_instance_data_loader):
        logits, labels, step_eval_loss = misc_utils.predict(
            trainer=trainer,
            model=model,
            inputs=test_inputs)
        if logits.argmax(axis=-1).item() != labels.item():
            wrong_input_collections.append(test_inputs)

    # Other settings are not supported as of now
    (s_test_damp,
     s_test_scale,
     s_test_num_samples) = influence_helpers.select_s_test_config(
        trained_on_task_name=trained_on_task_name,
        eval_task_name=eval_task_name)

    influences_collections = []
    for index, inputs in enumerate(wrong_input_collections[:num_eval_to_collect]):
        print(f"#{index}")
        influences = influence_helpers.compute_influences_simplified(
            k=DEFAULT_KNN_K,
            faiss_index=faiss_index,
            model=model,
            inputs=inputs,
            train_dataset=train_dataset,
            use_parallel=use_parallel,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            device_ids=[0, 1, 2, 3],
            precomputed_s_test=None)

        influences_collections.append(influences)

    remote_utils.save_and_mirror_scp_to_remote(
        object_to_save=influences_collections,
        file_name=(
            f"visualization.{num_eval_to_collect}"
            f".{train_task_name}-{eval_task_name}"
            f"-{hans_heuristic}-{trained_on_task_name}"
            f".{use_parallel}.pth"))

    return influences_collections


def run_experiments(option: str) -> List[List[Dict[int, float]]]:
    if option == "mnli2_and_hans":
        mnli2_influences = main(
            train_task_name="mnli-2",
            eval_task_name="mnli-2",
            num_eval_to_collect=100)

        hans_influences = main(
            train_task_name="mnli-2",
            eval_task_name="hans",
            num_eval_to_collect=100)

        return [mnli2_influences, hans_influences]

    if option == "mnli_and_hans_heuristic":
        hans_influences_collections = []
        for hans_heuristic in ["lexical_overlap", "subsequence", "constituent"]:
            hans_influences = main(
                train_task_name="mnli-2",
                eval_task_name="hans",
                num_eval_to_collect=100,
                hans_heuristic=hans_heuristic)

            hans_influences_collections.append(hans_influences)

        return hans_influences_collections

    if option == "hans_and_hans_heuristic":
        hans_influences_collections = []
        for hans_heuristic in ["lexical_overlap", "subsequence", "constituent"]:
            hans_influences = main(
                train_task_name="hans",
                eval_task_name="hans",
                num_eval_to_collect=100,
                hans_heuristic=hans_heuristic)

            hans_influences_collections.append(hans_influences)

        return hans_influences_collections

    raise ValueError(f"Unrecognized `option` {option}")


def get_datapoints_map(
    influences_collections: List[Dict[int, float]]
) -> Tuple[List[int], Dict[int, int]]:
    possible_datapoints = []
    for influences in influences_collections:
        possible_datapoints.extend(list(influences.keys()))

    possible_datapoints = sorted(set(possible_datapoints))
    datapoints_map = dict((v, k) for k, v in enumerate(possible_datapoints))
    return possible_datapoints, datapoints_map


def get_graph(
        influences_collections_list: List[List[Dict[int, float]]],
        train_vertex_color_map_fn: Optional[Callable[[int], int]] = None,
        train_vertex_radius_map_fn: Optional[Callable[[int], int]] = None,
        eval_vertex_radius: Optional[int] = None,
        eval_vertex_color_base: Optional[int] = None,
) -> gt_Graph_t:

    if train_vertex_color_map_fn is None:
        def train_vertex_color_map_fn(index: int) -> int:
            return DEFAULT_TRAIN_VERTEX_COLOR

    if train_vertex_radius_map_fn is None:
        def train_vertex_radius_map_fn(index: int) -> int:
            return DEFAULT_TRAIN_VERTEX_RADIUS

    if eval_vertex_radius is None:
        eval_vertex_radius = DEFAULT_EVAL_VERTEX_RADIUS

    if eval_vertex_color_base is None:
        eval_vertex_color_base = DEFAULT_EVAL_VERTEX_COLORS_BASE

    if train_vertex_color_map_fn is None:
        raise ValueError

    if train_vertex_radius_map_fn is None:
        raise ValueError

    NUM_INFLUENCE_COLLECTIONS = len(influences_collections_list)
    influences_collections_list_flatten = []
    for influences_collections in influences_collections_list:
        # Assume they all have the same lengths
        if len(influences_collections_list[0][0]) != len(influences_collections[0]):
            raise ValueError
        influences_collections_list_flatten.extend(influences_collections)

    # Note they share the same training dataset
    possible_datapoints, datapoints_map = get_datapoints_map(
        influences_collections=influences_collections_list_flatten)

    g = gt.Graph(directed=True)
    # Edge properties
    e_colors = g.new_edge_property("int")
    e_weights = g.new_edge_property("double")
    e_signed_influences = g.new_edge_property("double")
    e_unsigned_influences = g.new_edge_property("double")
    # Vertex properties
    v_sizes = g.new_vertex_property("int")
    v_colors = g.new_vertex_property("int")
    v_radius = g.new_vertex_property("int")
    v_data_indices = g.new_vertex_property("string")
    v_positions = g.new_vertex_property("vector<double>")
    v_positive_positions = g.new_vertex_property("vector<double>")
    v_negative_positions = g.new_vertex_property("vector<double>")

    train_vertices = []
    eval_vertices_collections = []

    # Add train vertices
    for datapoint_index in trange(len(possible_datapoints)):
        v = g.add_vertex()
        v_sizes[v] = DEFAULT_TRAIN_VERTEX_SIZE
        v_colors[v] = train_vertex_color_map_fn(
            possible_datapoints[datapoint_index])
        v_radius[v] = train_vertex_radius_map_fn(
            possible_datapoints[datapoint_index])
        v_data_indices[v] = f"train-{possible_datapoints[datapoint_index]}"
        train_vertices.append(v)

    # Add eval vertices
    for i, influences_collections in enumerate(influences_collections_list):

        eval_vertices = []
        for datapoint_index in trange(len(influences_collections)):
            v = g.add_vertex()
            v_sizes[v] = 10
            v_colors[v] = eval_vertex_color_base + i
            v_radius[v] = eval_vertex_radius
            v_data_indices[v] = f"eval-{i}-{datapoint_index}"

            base_degree = (360 / NUM_INFLUENCE_COLLECTIONS) * i
            fine_degree = (360 / NUM_INFLUENCE_COLLECTIONS / len(influences_collections)) * datapoint_index
            x_y_coordinate = get_circle_coordinates(
                r=eval_vertex_radius,
                degree=base_degree + fine_degree)
            position = np.random.normal(x_y_coordinate, 0.1)
            v_positions[v] = position
            v_positive_positions[v] = position
            v_negative_positions[v] = position
            eval_vertices.append(v)

        eval_vertices_collections.append(eval_vertices)

    # Add edges
    def add_edges(influences_collections: List[Dict[int, float]],
                  eval_vertices: List[gt.Vertex]) -> None:
        for eval_index, influences in enumerate(tqdm(influences_collections)):
            for train_index, train_influence in influences.items():
                # Negative influence is helpful (when the prediction is wrong)
                if train_influence < 0.0:
                    train_vertex = train_vertices[datapoints_map[train_index]]
                    eval_vertex = eval_vertices[eval_index]
                    e = g.add_edge(train_vertex, eval_vertex)
                    e_colors[e] = DEFAULT_HELPFUL_EDGE_COLOR
                    e_weights[e] = np.abs(train_influence)
                    e_signed_influences[e] = train_influence
                    e_unsigned_influences[e] = np.abs(train_influence)
                else:
                    train_vertex = train_vertices[datapoints_map[train_index]]
                    eval_vertex = eval_vertices[eval_index]
                    e = g.add_edge(train_vertex, eval_vertex)
                    e_colors[e] = DEFAULT_HARMFUL_EDGE_COLOR
                    e_weights[e] = np.abs(train_influence)
                    e_signed_influences[e] = train_influence
                    e_unsigned_influences[e] = np.abs(train_influence)

    for i, influences_collections in enumerate(influences_collections_list):
        add_edges(influences_collections, eval_vertices_collections[i])

    def _calculate_position(train_vertex: gt.Vertex) -> None:
        """Determine X-axis and Y-axis
        - We use X-axis to determine the divergence
        - We use Y-axis to determine the helpfulness/harmfulness
        """
        # Two types of targets
        # two types of connections
        _positive_points = []
        _negative_points = []
        _positive_influences = []
        _negative_influences = []
        for e in train_vertex.all_edges():
            target = e.target()
            if e_signed_influences[e] > 0:
                _positive_points.append(v_positions[target])
                _positive_influences.append(e_unsigned_influences[e])
            else:
                _negative_points.append(v_positions[target])
                _negative_influences.append(e_unsigned_influences[e])

        # `minimize` might fail using `np.sqrt(2)` for some reasons :\
        bound = 1.4 * v_radius[train_vertex]
        constraints = ({
            "type": "ineq",
            "fun": get_within_circle_constraint(v_radius[train_vertex])
        })

        if len(_positive_influences) == 0:
            _positive_xval = 0.0
            _positive_yval = 0.0
        else:
            _positive_points_stacked = np.stack(_positive_points, axis=0)
            _positive_influences_stacked = np.stack(_positive_influences, axis=0)
            _positive_optimize_result = minimize(
                distance_to_points_within_circle_vectorized,
                x0=(0, 0),
                constraints=constraints,
                bounds=((-bound, bound), (-bound, bound)),
                args=(_positive_influences_stacked,
                      _positive_points_stacked))
            _positive_xval, _positive_yval = _positive_optimize_result.x

        if len(_negative_influences) == 0:
            _negative_xval = 0.0
            _negative_yval = 0.0
        else:
            _negative_points_stacked = np.stack(_negative_points, axis=0)
            _negative_influences_stacked = np.stack(_negative_influences, axis=0)
            _negative_optimize_result = minimize(
                distance_to_points_within_circle_vectorized,
                x0=(0, 0),
                constraints=constraints,
                bounds=((-bound, bound), (-bound, bound)),
                args=(_negative_influences_stacked,
                      _negative_points_stacked))
            _negative_xval, _negative_yval = _negative_optimize_result.x

        _positive_xval = np.random.normal(_positive_xval, 0.01)
        _negative_xval = np.random.normal(_negative_xval, 0.01)
        _positive_yval = np.random.normal(_positive_yval, 0.01)
        _negative_yval = np.random.normal(_negative_yval, 0.01)
        v_positive_positions[train_vertex] = np.array([_positive_xval, _positive_yval])
        v_negative_positions[train_vertex] = np.array([_negative_xval, _negative_yval])
        v_positions[train_vertex] = np.array([(_positive_xval + _negative_xval) / 2,
                                              (_positive_yval + _negative_yval) / 2])

    # Run them in parallel
    # Parallel(n_jobs=-1)(
    #     delayed(_calculate_position)(train_vertex)
    #     for train_vertex in tqdm(train_vertices))
    for train_vertex in tqdm(train_vertices):
        _calculate_position(train_vertex)

    # Assign Edge properties
    g.edge_properties["colors"] = e_colors
    g.edge_properties["weights"] = e_weights
    g.edge_properties["signed_influences"] = e_signed_influences
    g.edge_properties["unsigned_influences"] = e_unsigned_influences
    # Assign Vertex properties
    g.vertex_properties["sizes"] = v_sizes
    g.vertex_properties["colors"] = v_colors
    g.vertex_properties["radius"] = v_radius
    g.vertex_properties["data_indices"] = v_data_indices
    g.vertex_properties["positions"] = v_positions
    g.vertex_properties["positive_positions"] = v_positive_positions
    g.vertex_properties["negative_positions"] = v_negative_positions

    return g, {
        "train_vertices": train_vertices,
        "eval_vertices_collections": eval_vertices_collections
    }


def get_recall_plot(model, example, faiss_index, full_influences_dict):
    # plt.rcParams["figure.figsize"] = [20, 5]
    recall_num_neighbors = [10, 100, 1000]
    num_neighbors = [10, 100, 1000, 10000, 50000, 100000]
    names = ["Most Helpful",
             "Most Harmful",
             "Most Influencetial",
             "Least Influential"]

    features = misc_utils.compute_BERT_CLS_feature(model, **example)
    features = features.cpu().detach().numpy()
    if list(full_influences_dict.keys()) != list(range(len(full_influences_dict))):
        raise ValueError

    full_influences = []
    for key in sorted(full_influences_dict):
        full_influences.append(full_influences_dict[key])

    sorted_indices_small_to_large = np.argsort(full_influences)
    sorted_indices_large_to_small = np.argsort(full_influences)[::-1]
    sorted_indices_abs_large_to_small = np.argsort(np.abs(full_influences))[::-1]
    sorted_indices_abs_small_to_large = np.argsort(np.abs(full_influences))

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
    recalls_collections = {}
    for i, (name, sorted_indices) in enumerate(zip(
            names,
            [sorted_indices_small_to_large,
             sorted_indices_large_to_small,
             sorted_indices_abs_large_to_small,
             sorted_indices_abs_small_to_large])):

        recalls_collection = []
        for recall_k in tqdm(recall_num_neighbors):
            recalls = []
            influential = sorted_indices[:recall_k]
            influential_set = set(influential.tolist())
            for k in num_neighbors:
                distances, indices = faiss_index.search(k=k, queries=features)
                indices_set = set(indices.squeeze(axis=0).tolist())
                recall = len(influential_set & indices_set) / len(influential_set)
                recalls.append(recall)

            recalls_collection.append(recalls)
            axes[i].plot(num_neighbors, recalls,
                         linestyle="--", marker="o",
                         label=f"recall@{recall_k}")

        axes[i].legend()
        axes[i].set_title(name)
        axes[i].set_xscale("log")
        axes[i].set_ylabel("Recall")
        axes[i].set_xlabel("Number of Nearest Neighbors")
        recalls_collections[name] = recalls_collection

    return recalls_collections


def plot_Xs_and_Ys_dict(
        axis: Subplot,
        Xs: List[float],
        Ys_dict: Dict[str, List[List[float]]],
        title: str,
        xlabel: str,
        ylabel: str,
        xscale_log: bool = True,
        yscale_log: bool = True,
        output_file_name: Optional[str] = None,
) -> None:

    color_map = {
        "helpful-1": "lightskyblue",
        "helpful-10": "deepskyblue",
        "helpful-100": "dodgerblue",
        "harmful-1": "lightcoral",
        "harmful-10": "salmon",
        "harmful-100": "red",
        "random-1": "darkgrey",
        "random-10": "dimgrey",
        "random-100": "black",
    }

    for tag in Ys_dict.keys():
        if tag not in color_map.keys():
            raise ValueError

        color = color_map[tag]
        data = np.array(Ys_dict[tag])
        is_random_data_point = "random" in tag
        # `data` should be [n, m]
        # where `n` is the number of independent trials
        # and `m` is the number of experiments within each trial
        if len(data.shape) != 2:
            raise ValueError(f"`data` should be an 2d array, {data.shape}")

        if data.shape[0] != 1:
            # i.e., it has multiple trials
            data_mean = data.mean(axis=0)
            data_max = data.max(axis=0)
            data_min = data.min(axis=0)
            # data_std = data.std(axis=0)
            axis.plot(
                Xs,
                data_mean,
                color=color,
                label=tag,
                linestyle=("--" if is_random_data_point else None))

            axis.fill_between(
                Xs,
                data_max,
                data_min,
                alpha=0.25,
                color=color)
        else:
            # i.e., only one trial
            axis.plot(
                Xs,
                data[0, ...],
                color=color)

    if xscale_log is True:
        axis.set_xscale("log")

    if yscale_log is True:
        axis.set_yscale("log")

    axis.set_xlabel(xlabel, fontsize=30)
    axis.set_ylabel(ylabel, fontsize=30)
    axis.set_title(title, fontsize=30)
    axis.legend(fontsize=15)

    if output_file_name is not None:
        plt.savefig(output_file_name)
