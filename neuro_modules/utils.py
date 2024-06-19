from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances

import scipy
import torch
from torch import nn
from torch.nn import functional as F



def hungarian_algorithm(cost_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch-applies the hungarian algorithm to find a matching that minimizes
    the overall cost. Returns the matching indices as a LongTensor with shape
    (batch size, 2, min(num objects, num slots)). The first column is the row
    indices (the indices of the true objects) while the second column is the
    column indices (the indices of the slots). The row indices are always in
    ascending order, while the column indices are not necessarily. The outputs
    are on the same device as `cost_matrix` but gradients are detached.
    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the
    row indices will be [1,2,3] and the column indices will be [0,2,1].
    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).
    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots))
                with the costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects,
                num slots)) containing the indices for the resulting matching.

    From https://github.com/addtt/object-centric-library/utils/slot_matching.py.
    """

    # List of tuples of size 2 containing flat arrays
    raw_indices = list(
        map(scipy.optimize.linear_sum_assignment, cost_matrix.cpu().detach().numpy())
    )
    indices = torch.tensor(
        np.array(raw_indices), device=cost_matrix.device, dtype=torch.long
    )
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    return smallest_cost_matrix.to(cost_matrix.device), indices


def calculate_distances(labels, data, size):
    batch_size, _, _ = labels.shape
    
    labels_reshaped = labels.unsqueeze(1).repeat(1, size, 1, 1).reshape(-1, labels.size(2))
    data_reshaped = data.unsqueeze(2).repeat(1, 1, size, 1).reshape(-1, data.size(2))

    bce_losses = F.binary_cross_entropy(data_reshaped, labels_reshaped, reduction='none')

    cost_matrix = bce_losses.view(batch_size, size, size, -1).mean(dim=-1)

    return cost_matrix

def get_attended_bounding_box(mask):
    threshold_value = 0.80
    
    attended_region = (mask > threshold_value).astype(int)

    non_zero_indices = np.nonzero(attended_region)
    
    if non_zero_indices[0].size == 0:
        return (-1,-1,-1,-1)

    x_min = np.min(non_zero_indices[1])
    y_min = np.min(non_zero_indices[0])
    x_max = np.max(non_zero_indices[1])
    y_max = np.max(non_zero_indices[0])

    bounding_box = (x_min, y_min, x_max, y_max)

    return bounding_box


def average_precision(pred, attributes, distance_threshold):

  [batch_size, _, element_size] = attributes.shape
  [_, predicted_elements, _] = pred.shape

  def unsorted_id_to_image(detection_id, predicted_elements):
    """Find the index of the image from the unsorted detection index."""
    return int(detection_id // predicted_elements)

  flat_size = batch_size * predicted_elements
  flat_pred = np.reshape(pred, [flat_size, element_size])
  sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

  sorted_predictions = np.take_along_axis(
      flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
  idx_sorted_to_unsorted = np.take_along_axis(
      np.arange(flat_size), sort_idx, axis=0)

  def process_targets(target):
    """Unpacks the target into the CLEVR properties."""
    coords = target[:3]
    shape = np.argmax(target[3:6])
    color = np.argmax(target[6:14])
    object_size = np.argmax(target[14:16])
    material = np.argmax(target[16:18])
    real_obj = target[18]
    return coords, shape, color, object_size, material, real_obj
  
  def process_targets_shapes(target):
    """Unpacks the target into the SHAPES properties."""
    coords = target[:2]
    shape =np.argmax(target[2:5])
    colour = np.argmax(target[5:8])
    object_size = np.argmax(target[8:10])
    real_obj = target[10]
    return coords, shape, colour, object_size, real_obj

  true_positives = np.zeros(sorted_predictions.shape[0])
  false_positives = np.zeros(sorted_predictions.shape[0])

  detection_set = set()

  for detection_id in range(sorted_predictions.shape[0]):
    # Extract the current prediction.
    current_pred = sorted_predictions[detection_id, :]
    # Find which image the prediction belongs to. Get the unsorted index from
    # the sorted one and then apply to unsorted_id_to_image function that undoes
    # the reshape.
    original_image_idx = unsorted_id_to_image(
        idx_sorted_to_unsorted[detection_id], predicted_elements)
    # Get the ground truth image.
    gt_image = attributes[original_image_idx, :, :]

    # Initialize the maximum distance and the id of the groud-truth object that
    # was found.
    best_distance = 10000
    best_id = None

        # Unpack the prediction by taking the argmax on the discrete attributes.
    (pred_coords, pred_shape, pred_color, pred_object_size, pred_material,
     _) = process_targets(current_pred)
    
    # (pred_coords, pred_shape, pred_color,pred_object_size ,
    #  _) = process_targets_shapes(current_pred)

    # Loop through all objects in the ground-truth image to check for hits.
    for target_object_id in range(gt_image.shape[0]):
      target_object = gt_image[target_object_id, :]
      # Unpack the targets taking the argmax on the discrete attributes.
      # (target_coords, target_shape, target_object_size,
      #  target_color, target_real_obj) = process_targets_shapes(target_object)

      (target_coords, target_shape, target_color, target_object_size, 
       target_material, target_real_obj) = process_targets(target_object)
      # Only consider real objects as matches.
      if target_real_obj:
        # For the match to be valid all attributes need to be correctly
        # predicted.
        # pred_attr = [pred_object_size, pred_shape, pred_color]
        pred_attr =  [pred_object_size, pred_material, pred_shape, pred_color]
        # target_attr = [
        #     target_object_size, target_shape, target_color]
        target_attr = [
          target_object_size, target_material, target_shape, target_color]
        match = pred_attr == target_attr
        if match:
          # If a match was found, we check if the distance is below the
          # specified threshold. Recall that we have rescaled the coordinates
          # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
          # `pred_coords`. To compare in the original scale, we thus need to
          # multiply the distance values by 6 before applying the norm.
          distance = np.linalg.norm((target_coords - pred_coords) * 6.)

          # If this is the best match we've found so far we remember it.
          if distance < best_distance:
            best_distance = distance
            best_id = target_object_id
    if best_distance < distance_threshold or distance_threshold == -1:
      # We have detected an object correctly within the distance confidence.
      # If this object was not detected before it's a true positive.
      if best_id is not None:
        if (original_image_idx, best_id) not in detection_set:
          true_positives[detection_id] = 1
          detection_set.add((original_image_idx, best_id))
        else:
          false_positives[detection_id] = 1
      else:
        false_positives[detection_id] = 1
    else:
      false_positives[detection_id] = 1
  accumulated_fp = np.cumsum(false_positives)
  accumulated_tp = np.cumsum(true_positives)
  recall_array = accumulated_tp / np.sum(attributes[:, :, -1])
  precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

  return compute_average_precision(
      np.array(precision_array, dtype=np.float32),
      np.array(recall_array, dtype=np.float32))


def compute_average_precision(precision, recall):
  """Computation of the average precision from precision and recall arrays."""
  recall = recall.tolist()
  precision = precision.tolist()
  recall = [0] + recall + [1]
  precision = [0] + precision + [0]

  for i in range(len(precision) - 1, -0, -1):
    precision[i - 1] = max(precision[i - 1], precision[i])

  indices_recall = [
      i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
  ]

  average_precision = 0.
  for i in indices_recall:
    average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
  return average_precision


def assign_pixels_to_clusters(image, attention_masks, background_value=0):
    assigned_clusters = np.full(image.shape[:2], -1)

    background_mask = np.all(image == background_value, axis=-1)

    max_attention_mask = np.zeros(image.shape[:2], dtype=np.float32)
    for cluster_id, attention_mask in enumerate(attention_masks):
        cluster_update_mask = attention_mask > max_attention_mask
        assigned_clusters[cluster_update_mask] = cluster_id
        max_attention_mask = np.maximum(max_attention_mask, attention_mask)

    assigned_clusters[background_mask] = -1

    return assigned_clusters


def adjusted_rand_index(true_mask, pred_mask, name="ari_score"):
    r"""Computes the adjusted Rand index (ARI), a clustering similarity score.
    This implementation ignores points with no cluster label in `true_mask` (i.e.
    those points for which `true_mask` is a zero vector). In the context of
    segmentation, that means this function can ignore points in an image
    corresponding to the background (i.e. not to an object).
    Args:
        true_mask: `Tensor` of shape [batch_size, n_points, n_true_groups].
            The true cluster assignment encoded as one-hot.
        pred_mask: `Tensor` of shape [batch_size, n_points, n_pred_groups].
            The predicted cluster assignment encoded as categorical probabilities.
            This function works on the argmax over axis 2.
            name: str. Name of this operation (defaults to "ari_score").
    Returns:
        ARI scores as a tf.float32 `Tensor` of shape [batch_size].
    Raises:
        ValueError: if n_points <= n_true_groups and n_points <= n_pred_groups.
            We've chosen not to handle the special cases that can occur when you have
            one cluster per datapoint (which would be unusual).
    References:
        Lawrence Hubert, Phipps Arabie. 1985. "Comparing partitions"
            https://link.springer.com/article/10.1007/BF01908075
        Wikipedia
            https://en.wikipedia.org/wiki/Rand_index
        Scikit Learn
            http://scikit-learn.org/stable/modules/generated/\
            sklearn.metrics.adjusted_rand_score.html
    """
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    if n_points <= n_true_groups and n_points <= n_pred_groups:
        # This rules out the n_true_groups == n_pred_groups == n_points
        # corner case, and also n_true_groups == n_pred_groups == 0, since
        # that would imply n_points == 0 too.
        # The sklearn implementation has a corner-case branch which does
        # handle this. We chose not to support these cases to avoid counting
        # distinct clusters just to check if we have one cluster per datapoint.
        raise ValueError(
            "adjusted_rand_index requires n_groups < n_points. We don't handle "
            "the special cases that can occur when you have one cluster "
            "per datapoint."
        )

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    # We convert true and predicted clusters to one-hot ('oh') representations.
    true_mask_oh = true_mask.type(torch.float32)  # already one-hot
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).type(torch.float32)

    n_points = torch.sum(true_mask_oh, axis=[1, 2]).type(torch.float32)

    nij = torch.einsum("bji,bjk->bki", pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, axis=1)
    b = torch.sum(nij, axis=2)

    rindex = torch.sum(nij * (nij - 1), axis=[1, 2])
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2

    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator
    # If a divide by 0 occurs, set the ARI value to 1.
    zeros_in_denominator = torch.argwhere(denominator == 0).flatten()
    if zeros_in_denominator.nelement() > 0:
        ari[zeros_in_denominator] = 1

    # The case where n_true_groups == n_pred_groups == 1 needs to be
    # special-cased (to return 1) as the above formula gives a divide-by-zero.
    # This might not work when true_mask has values that do not sum to one:
    both_single_cluster = torch.logical_and(
        _all_equal(true_group_ids), _all_equal(pred_group_ids)
    )
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)

def _all_equal(values):
    """Whether values are all equal along the final axis."""
    return torch.all(torch.eq(values, values[..., :1]), axis=-1)


def aggregate_slots(slots, method='average'):
    if method == 'average':
        return np.mean(slots, axis=0)
    elif method == 'max':
        return np.max(slots, axis=0)
    elif method == 'concat':
        return np.concatenate(slots, axis=0)
    else:
        raise ValueError("Unsupported aggregation method.")
    

def get_diverse_slots(data, num_clusters):
  representations = np.array([agg_rep for _, agg_rep in data])

  num_samples, num_rows, num_columns = representations.shape
  flattened_slots = representations.reshape(num_samples, -1)
  print(flattened_slots.shape)
  
  kmeans = KMeans(n_clusters=num_clusters)
  kmeans.fit(flattened_slots)
  cluster_centers = kmeans.cluster_centers_
  closest, _ = pairwise_distances_argmin_min(cluster_centers, flattened_slots)
  diverse_sample_indices = [data[idx][0] for idx in closest]
  return diverse_sample_indices

def visualize_tsne(data, num_clusters):
    representations = np.array([agg_rep for _, _ ,agg_rep in data])
    num_samples, num_rows, num_columns = representations.shape
    flattened_slots = representations.reshape(num_samples, -1)
    
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(flattened_slots)
    cluster_centers = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(cluster_centers, flattened_slots)
    diverse_sample_indices = [data[idx][0] for idx in closest]
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(flattened_slots)
    
    plt.figure(figsize=(12, 8))
    
    classes = [data[idx][1] for idx in range(len(data))]
    
    for class_label in set(classes):
        indices = [i for i, x in enumerate(classes) if x == class_label]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {class_label}')
    
    plt.scatter(tsne_results[closest, 0], tsne_results[closest, 1], c='black', marker='x', label='Cluster Centers')
    plt.legend()
    plt.title('t-SNE visualization of slots')
    plt.show()
    
    return diverse_sample_indices