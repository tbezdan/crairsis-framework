import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn import preprocessing
import umap
import pacmap
import hdbscan
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
import matplotlib as mpl
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)


def post_segmentation(
    shap_subclusters_folder,
    df,
    folder_output,
    file_name,
    task_type,
    enable_subclustering,
    subcluster_prob_threshold,
    m_c_s_percentage=2,
    min_samples_perc=0.2,
    metrika="euclidean",
    cols=["#1d89e4", "#f52757"],
    dimensions=3,
    show_figs=False,
    scaled=True,
    write=True,
    class_label=None,
    dimensionality_reduction_method=None,
):

    df_temp = df[["id"]].reset_index(drop=True)
    df = df.drop(["id"], axis=1)

    if scaled:
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)

    n_neighbors = 15
    n_components = 3

    if dimensionality_reduction_method == "PaCMAP":
        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=0.5,
            FP_ratio=2.0,
        )
        logger.info("Dimensionality reduction method: PaCMAP")
    elif dimensionality_reduction_method == "UMAP":

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=n_components,
            metric=metrika,
        )
        logger.info("Dimensionality reduction method: UMAP")
    else:
        raise ValueError(
            f"Unsupported dimensionality reduction method: {dimensionality_reduction_method}. "
            "Please choose 'PaCMAP' or 'UMAP'."
        )

    data = reducer.fit_transform(df)

    data_ = pd.DataFrame(data)
    data_["id"] = df_temp["id"]

    if task_type == "regression":
        file_path = os.path.join(
            folder_output, f"{file_name}_segments_dimensionality_reduction.csv"
        )
    elif task_type == "classification":
        suffix = f"_class_{class_label}" if class_label is not None else ""
        file_path = os.path.join(
            folder_output,
            f"{file_name}_segments_dimensionality_reduction{suffix}.csv",
        )

    data_.to_csv(file_path, index=False)

    if show_figs:
        plot_kwds = {"alpha": 0.25, "s": 60, "linewidths": 0}

    if show_figs:
        if dimensions == 2:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=data.T[0], y=data.T[1], mode="markers", showlegend=True)
            )
        if dimensions == 3:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=data.T[0],
                    y=data.T[1],
                    z=data.T[2],
                    mode="markers",
                    showlegend=True,
                    opacity=0.5,
                )
            )
        fig.show()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(5, int(round(m_c_s_percentage * df.shape[0] / 100, 0))),
        prediction_data=True,
        metric=metrika,
        min_samples=max(5, int(round(min_samples_perc * df.shape[0] / 100, 0))),
    ).fit(data)

    cluster_labels = clusterer.fit_predict(data)
    clusters, counts = np.unique(cluster_labels, return_counts=True)

    clusters_hard = pd.DataFrame(clusterer.probabilities_)

    from matplotlib.cm import ScalarMappable

    pal = LinearSegmentedColormap.from_list("Custom", cols, N=len(counts))
    pp = []
    for i in range(pal.N):
        rgba = mpl.colors.rgb2hex(pal(i))
        pp.append(rgba)
    pal = pp

    if show_figs:
        colors = [
            sns.desaturate(pal[col], sat)
            for col, sat in zip(clusterer.labels_, clusterer.probabilities_)
        ]
        plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
        plt.show()

    def exemplars(cluster_id, condensed_tree):
        raw_tree = condensed_tree._raw_tree
        cluster_tree = raw_tree[raw_tree["child_size"] > 1]
        leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
        result = np.array([])
        for leaf in leaves:
            max_lambda = raw_tree["lambda_val"][raw_tree["parent"] == leaf].max()
            points = raw_tree["child"][
                (raw_tree["parent"] == leaf) & (raw_tree["lambda_val"] == max_lambda)
            ]
            result = np.hstack((result, points))
        return result.astype(int)

    tree = clusterer.condensed_tree_

    if show_figs:
        palette = pal
        plt.scatter(data.T[0], data.T[1], c="grey", **plot_kwds)
        for i, c in enumerate(tree._select_clusters()):
            c_exemplars = exemplars(c, tree)
            plt.scatter(
                data.T[0][c_exemplars],
                data.T[1][c_exemplars],
                c=palette[i],
                **plot_kwds,
            )
        plt.show()

    def min_dist_to_exemplar(point, cluster_exemplars, data):
        dists = cdist([data[point]], data[cluster_exemplars.astype(np.int32)])
        return dists.min()

    def dist_vector(point, exemplar_dict, data):
        result = {}
        for cluster in exemplar_dict:
            result[cluster] = min_dist_to_exemplar(point, exemplar_dict[cluster], data)
        return np.array(list(result.values()))

    def dist_membership_vector(point, exemplar_dict, data, softmax=False):
        if softmax:
            result = np.exp(1.0 / dist_vector(point, exemplar_dict, data))
            result[~np.isfinite(result)] = np.finfo(np.double).max
        else:
            result = 1.0 / dist_vector(point, exemplar_dict, data)
            result[~np.isfinite(result)] = np.finfo(np.double).max
        result /= result.sum()
        return result

    print("Distance Based Membership")
    exemplar_dict = {c: exemplars(c, tree) for c in tree._select_clusters()}

    if show_figs:
        colors = np.empty((data.shape[0], 3))
        for x in range(data.shape[0]):
            membership_vector = dist_membership_vector(x, exemplar_dict, data)
            color = np.argmax(membership_vector)
            saturation = membership_vector[color]
            colors[x] = sns.desaturate(pal[color], saturation)
        plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
        plt.show()

    print("Outlier Based Membership")

    def max_lambda_val(cluster, tree):
        cluster_tree = tree[tree["child_size"] > 1]
        leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster)
        max_lambda = 0.0
        for leaf in leaves:
            max_lambda = max(
                max_lambda, tree["lambda_val"][tree["parent"] == leaf].max()
            )
        return max_lambda

    def points_in_cluster(cluster, tree):
        leaves = hdbscan.plots._recurse_leaf_dfs(tree, cluster)
        return leaves

    def merge_height(point, cluster, tree, point_dict):
        cluster_row = tree[tree["child"] == cluster]
        cluster_height = cluster_row["lambda_val"][0]
        if point in point_dict[cluster]:
            merge_row = tree[tree["child"] == float(point)][0]
            return merge_row["lambda_val"]
        else:
            while point not in point_dict[cluster]:
                parent_row = tree[tree["child"] == cluster]
                cluster = parent_row["parent"].astype(np.float64)[0]
            for row in tree[tree["parent"] == cluster]:
                child_cluster = float(row["child"])
                if child_cluster == point:
                    return row["lambda_val"]
                if child_cluster in point_dict and point in point_dict[child_cluster]:
                    return row["lambda_val"]

    def per_cluster_scores(point, cluster_ids, tree, max_lambda_dict, point_dict):
        result = {}
        point_row = tree[tree["child"] == point]
        point_cluster = float(point_row[0]["parent"])
        max_lambda = max_lambda_dict[point_cluster] + 1e-8

        for c in cluster_ids:
            height = merge_height(point, c, tree, point_dict)
            result[c] = max_lambda / (max_lambda - height)
        return result

    def outlier_membership_vector(
        point, cluster_ids, tree, max_lambda_dict, point_dict, softmax=True
    ):
        if softmax:
            result = np.exp(
                np.array(
                    list(
                        per_cluster_scores(
                            point, cluster_ids, tree, max_lambda_dict, point_dict
                        ).values()
                    )
                )
            )
            result[~np.isfinite(result)] = np.finfo(np.double).max
        else:
            result = np.array(
                list(
                    per_cluster_scores(
                        point, cluster_ids, tree, max_lambda_dict, point_dict
                    ).values()
                )
            )
        result /= result.sum()
        return result

    cluster_ids = tree._select_clusters()
    raw_tree = tree._raw_tree
    all_possible_clusters = np.arange(
        data.shape[0], raw_tree["parent"].max() + 1
    ).astype(np.float64)
    max_lambda_dict = {c: max_lambda_val(c, raw_tree) for c in all_possible_clusters}
    point_dict = {c: set(points_in_cluster(c, raw_tree)) for c in all_possible_clusters}

    if show_figs:
        colors = np.empty((data.shape[0], 3))
        for x in range(data.shape[0]):
            membership_vector = outlier_membership_vector(
                x, cluster_ids, raw_tree, max_lambda_dict, point_dict, False
            )
            color = np.argmax(membership_vector)
            saturation = membership_vector[color]
            colors[x] = sns.desaturate(pal[color], saturation)
        plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)

        plt.show()

    print("The Middle Way")

    def combined_membership_vector(
        point,
        data,
        tree,
        exemplar_dict,
        cluster_ids,
        max_lambda_dict,
        point_dict,
        softmax=False,
    ):
        raw_tree = tree._raw_tree
        dist_vec = dist_membership_vector(point, exemplar_dict, data, softmax)
        outl_vec = outlier_membership_vector(
            point, cluster_ids, raw_tree, max_lambda_dict, point_dict, softmax
        )
        result = dist_vec * outl_vec
        result /= result.sum()
        return result

    if show_figs:
        colors = np.empty((data.shape[0], 3))
        for x in range(data.shape[0]):
            membership_vector = combined_membership_vector(
                x,
                data,
                tree,
                exemplar_dict,
                cluster_ids,
                max_lambda_dict,
                point_dict,
                False,
            )
            color = np.argmax(membership_vector)
            saturation = membership_vector[color]
            colors[x] = sns.desaturate(pal[color], saturation)
        plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
        plt.show()

    def prob_in_some_cluster(point, tree, cluster_ids, point_dict, max_lambda_dict):
        heights = []
        for cluster in cluster_ids:
            heights.append(merge_height(point, cluster, tree._raw_tree, point_dict))
        height = max(heights)
        nearest_cluster = cluster_ids[np.argmax(heights)]
        max_lambda = max_lambda_dict[nearest_cluster]
        return height / max_lambda

    colors = np.empty((data.shape[0], 3))
    if -1 in clusters:
        probabilities = np.zeros((data.shape[0], len(clusters) - 1))
    else:
        probabilities = np.zeros((data.shape[0], len(clusters)))
    for x in range(data.shape[0]):
        membership_vector = combined_membership_vector(
            x,
            data,
            tree,
            exemplar_dict,
            cluster_ids,
            max_lambda_dict,
            point_dict,
            False,
        )
        membership_vector *= prob_in_some_cluster(
            x, tree, cluster_ids, point_dict, max_lambda_dict
        )
        probabilities[x] = membership_vector
        color = np.argmax(membership_vector)
        saturation = membership_vector[color]
        colors[x] = sns.desaturate(pal[color], saturation)
    if -1 in clusters:
        probabilities = pd.DataFrame(
            probabilities, columns=["Cluster_" + str(item) for item in clusters[1:]]
        )
    else:
        probabilities = pd.DataFrame(
            probabilities, columns=["Cluster_" + str(item) for item in clusters]
        )

    probabilities[probabilities < 1e-5] = 0
    probabilities["sum"] = probabilities.sum(axis=1)
    probabilities["hard_cluster_label"] = cluster_labels
    probabilities["hard_cluster_probability"] = clusters_hard
    probabilities["id"] = df_temp["id"]

    stat = pd.DataFrame(
        {
            "cluster": clusters,
            "count": counts,
            "percentage": 100 * counts / np.sum(counts),
        }
    )
    if show_figs:
        plt.scatter(data.T[0], data.T[1], c=colors)
        plt.show()

    if write:
        suffix = f"_class_{class_label}" if class_label is not None else ""

        probabilities_file_path = os.path.join(
            folder_output,
            f"{file_name}_segments_probabilities_{dimensions}d{suffix}.csv",
        )
        probabilities.to_csv(probabilities_file_path, index=False)

        statistics_file_path = os.path.join(
            folder_output,
            f"{file_name}_segments_statistics_{dimensions}d{suffix}.csv",
        )
        stat.to_csv(statistics_file_path, index=False)

    if enable_subclustering:
        perform_subclustering(
            shap_subclusters_folder,
            data=data,
            probabilities=probabilities,
            clusters=cluster_labels,
            file_name=file_name,
            task_type=task_type,
            class_label=class_label,
            subcluster_prob_threshold=subcluster_prob_threshold,
            metrika=metrika,
            m_c_s_percentage=m_c_s_percentage,
            min_samples_perc=min_samples_perc,
            dimensions=dimensions,
            show_figs=show_figs,
            write=write,
        )

    return probabilities, stat


def perform_shap_clustering(
    shap_clusters_folder,
    shap_subclusters_folder,
    shap_folder,
    task_type,
    dimensionality_reduction_method,
    enable_subclustering,
    subcluster_prob_threshold,
):
    logger.info("Starting SHAP clustering process.")

    shap_files = [
        f
        for f in os.listdir(shap_folder)
        if f.endswith("_impacts_local.csv") or "_impacts_local_class" in f
    ]

    if not shap_files:
        logger.warning("No SHAP files found with the required suffixes.")
        return

    for shap_file in shap_files:
        try:
            # Extract the base file name before the suffix
            file_name = shap_file.split("_impacts_local")[0]

            # Check for the presence of a class label in the file name
            if "_class_" in shap_file:
                class_label = shap_file.split("_class_")[1].split(".csv")[0]
            else:
                class_label = None

            file_path = os.path.join(shap_folder, shap_file)
            folder_output = os.path.join(shap_clusters_folder)

            logger.info(f"Processing file: {shap_file}")
            df = pd.read_csv(file_path)

            print("*********************")
            print(class_label, "class_label")

            if task_type == "classification" and class_label is not None:
                logger.info(f"Performing clustering for class: {class_label}")
                post_segmentation(
                    shap_subclusters_folder,
                    df=df,
                    folder_output=folder_output,
                    file_name=file_name,
                    task_type=task_type,
                    enable_subclustering=enable_subclustering,
                    subcluster_prob_threshold=subcluster_prob_threshold,
                    show_figs=False,
                    class_label=class_label,
                    dimensionality_reduction_method=dimensionality_reduction_method,
                )
            else:
                logger.info("Performing clustering for general data.")
                post_segmentation(
                    shap_subclusters_folder,
                    df=df,
                    folder_output=folder_output,
                    file_name=file_name,
                    task_type=task_type,
                    enable_subclustering=enable_subclustering,
                    subcluster_prob_threshold=subcluster_prob_threshold,
                    show_figs=False,
                    dimensionality_reduction_method=dimensionality_reduction_method,
                )

            logger.info(f"Successfully processed file: {shap_file}")
        except Exception as e:
            logger.error(f"Error processing file {shap_file}: {e}", exc_info=True)

    logger.info("SHAP clustering process completed.")


def perform_subclustering(
    shap_subclusters_folder,
    data,
    probabilities,
    clusters,
    file_name,
    task_type,
    class_label,
    subcluster_prob_threshold,
    metrika="euclidean",
    m_c_s_percentage=2,
    min_samples_perc=0.2,
    dimensions=3,
    show_figs=False,
    write=True,
):
    """
    Perform subclustering for each primary cluster based on a probability threshold.
    Handles classification tasks by processing each class_label individually.
    """
    logger.info(f"Starting subclustering process. Task type: {task_type}")

    if task_type == "classification" and class_label is not None:
        logger.info(f"Processing subclustering for class label: {class_label}")

    unique_clusters = np.unique(clusters)
    logger.info(f"Found {len(unique_clusters)} unique clusters: {unique_clusters}")

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            logger.info(f"Skipping noise points for cluster {cluster_id}.")
            continue

        logger.info(f"Processing parent cluster {cluster_id}.")

        # Filter data points belonging to the current cluster with high probabilities

        cluster_mask = (
            probabilities[f"Cluster_{cluster_id}"] >= subcluster_prob_threshold
        )
        cluster_points = data[cluster_mask]
        cluster_probabilities = probabilities[cluster_mask].reset_index(drop=True)

        logger.info(
            f"Cluster {cluster_id}: Points after probability threshold filter: {cluster_points.shape[0]}"
        )

        if cluster_points.shape[0] < 5:
            logger.info(
                f"Skipping subclustering for cluster {cluster_id} (too few points)."
            )
            continue

        # Perform HDBSCAN subclustering
        subclusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(
                5, int(round(m_c_s_percentage * cluster_points.shape[0] / 100, 0))
            ),
            prediction_data=True,
            metric=metrika,
            min_samples=max(
                5, int(round(min_samples_perc * cluster_points.shape[0] / 100, 0))
            ),
        ).fit(cluster_points)

        subcluster_labels = subclusterer.labels_
        subcluster_hard_probabilities = subclusterer.probabilities_
        subcluster_counts = np.unique(subcluster_labels, return_counts=True)

        logger.info(
            f"Parent cluster {cluster_id}: Found {len(subcluster_counts[0])} subclusters."
        )

        # Save subcluster probabilities
        subcluster_probabilities = pd.DataFrame(
            hdbscan.all_points_membership_vectors(subclusterer)
        )

        subcluster_probabilities.columns = [
            f"subcluster_{i}" for i in range(subcluster_probabilities.shape[1])
        ]

        subclusters_hard = pd.DataFrame(subclusterer.probabilities_)

        subcluster_probabilities[subcluster_probabilities < 1e-5] = 0
        subcluster_probabilities["sum"] = subcluster_probabilities.sum(axis=1)
        subcluster_probabilities["subcluster_label"] = subcluster_labels
        subcluster_probabilities["subcluster_hard_probability"] = (
            subcluster_hard_probabilities
        )
        subcluster_probabilities["id"] = cluster_probabilities["id"]
        subcluster_probabilities["parent_cluster"] = cluster_id

        # Save subcluster statistics
        subcluster_stats = pd.DataFrame(
            {
                "subcluster": subcluster_counts[0],
                "count": subcluster_counts[1],
                "percentage": 100 * subcluster_counts[1] / len(cluster_points),
            }
        )

        if write:
            suffix = f"_class_{class_label}" if class_label is not None else ""
            prob_file_path = os.path.join(
                shap_subclusters_folder,
                f"{file_name}_cluster_{cluster_id}_subcluster_probabilities{suffix}.csv",
            )
            subcluster_probabilities.to_csv(prob_file_path, index=False)

            stats_file_path = os.path.join(
                shap_subclusters_folder,
                f"{file_name}_cluster_{cluster_id}_subcluster_statistics{suffix}.csv",
            )
            subcluster_stats.to_csv(stats_file_path, index=False)

        # Visualization
        if show_figs and dimensions == 2:
            plt.scatter(
                cluster_points.T[0],
                cluster_points.T[1],
                c=subcluster_labels,
                cmap="tab10",
                alpha=0.5,
            )
            plt.title(f"Subclusters for Parent Cluster {cluster_id}")
            plt.show()

    logger.info("Subclustering process completed.")
