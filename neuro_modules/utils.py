from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
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
    # Threshold the mask to identify the attended region
    threshold_value = 0.80
    
    attended_region = (mask > threshold_value).astype(int)

    # Find the indices of non-zero elements in the attended region
    non_zero_indices = np.nonzero(attended_region)
    
    if non_zero_indices[0].size == 0:
        return (-1,-1,-1,-1)

    # Calculate the bounding box coordinates
    x_min = np.min(non_zero_indices[1])
    y_min = np.min(non_zero_indices[0])
    x_max = np.max(non_zero_indices[1])
    y_max = np.max(non_zero_indices[0])

    # Represent the bounding box coordinates
    bounding_box = (x_min, y_min, x_max, y_max)

    return bounding_box


def get_dissimilar_ranking(imgs, model ,num_slots):
    num_img = len(imgs)

    model.eval()
    inputs = (imgs / 127.5 ) - 1
    _, _, _, slots , _ = model(inputs)
    slots = slots.detach().numpy()


    pca_models = []
    reduced_slots = []
    for img_idx in range(num_img):
        pca = PCA(n_components=2)
        reduced_slot = pca.fit_transform(slots[img_idx])
        pca_models.append(pca)
        reduced_slots.append(reduced_slot)

    dissimilarity_scores = np.zeros((num_img, num_img))
    for i in range(num_img):
        for j in range(num_img):
            if i != j:
                distance_sum = 0
                for k in range(num_slots):
                    distance_sum += cosine(reduced_slots[i][k], reduced_slots[j][k])
                dissimilarity_scores[i, j] = distance_sum


    total_dissimilarity = np.sum(dissimilarity_scores, axis=1)
    ranking = np.argsort(total_dissimilarity)

    return ranking


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
    object_size = torch.argmax(target[3:5])
    material = torch.argmax(target[5:7])
    shape = torch.argmax(target[7:10])
    color = torch.argmax(target[10:18])
    real_obj = target[18]
    return coords, object_size, material, shape, color, real_obj
  
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
    # (pred_coords, pred_object_size, pred_material, pred_shape, pred_color,
    #  _) = process_targets(current_pred)
    
    (pred_coords, pred_shape, pred_color,pred_object_size ,
     _) = process_targets_shapes(current_pred)

    # Loop through all objects in the ground-truth image to check for hits.
    for target_object_id in range(gt_image.shape[0]):
      target_object = gt_image[target_object_id, :]
      # Unpack the targets taking the argmax on the discrete attributes.
      (target_coords, target_shape, target_object_size,
       target_color, target_real_obj) = process_targets_shapes(target_object)
      # Only consider real objects as matches.
      if target_real_obj:
        # For the match to be valid all attributes need to be correctly
        # predicted.
        pred_attr = [pred_object_size, pred_shape, pred_color]
        target_attr = [
            target_object_size, target_shape, target_color]
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


## TODO: Remove 
def average_precision_2(pred, attributes, distance_threshold):

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
    object_size = torch.argmax(target[3:5])
    material = torch.argmax(target[5:7])
    shape = torch.argmax(target[7:10])
    color = torch.argmax(target[10:18])
    real_obj = target[18]
    return coords, object_size, material, shape, color, real_obj
  
  def process_targets_shapes(target):
    """Unpacks the target into the SHAPES properties."""
    coords = target[:2]
    shape =np.argmax(target[2:5])
    colour = np.argmax(target[5:8])
    real_obj = target[8]
    return coords, shape, colour, real_obj

  true_positives = np.zeros(sorted_predictions.shape[0])
  false_positives = np.zeros(sorted_predictions.shape[0])

  detection_set = set()

  for detection_id in range(sorted_predictions.shape[0]):
    current_pred = sorted_predictions[detection_id, :]

    original_image_idx = unsorted_id_to_image(
        idx_sorted_to_unsorted[detection_id], predicted_elements)
    gt_image = attributes[original_image_idx, :, :]

    best_distance = 10000
    best_id = None
    (pred_coords, pred_shape, pred_color ,
     _) = process_targets_shapes(current_pred)

    # Loop through all objects in the ground-truth image to check for hits.
    for target_object_id in range(gt_image.shape[0]):
      target_object = gt_image[target_object_id, :]
      # Unpack the targets taking the argmax on the discrete attributes.
      (target_coords, target_shape,
       target_color, target_real_obj) = process_targets_shapes(target_object)
      # Only consider real objects as matches.
      if target_real_obj:
        pred_attr = [pred_shape, pred_color]
        target_attr = [target_shape, target_color]
        match = pred_attr == target_attr
        if match:
          distance = np.linalg.norm((target_coords - pred_coords) * 6.)
          # If this is the best match we've found so far we remember it.
          if distance < best_distance:
            best_distance = distance
            best_id = target_object_id
    if best_distance < distance_threshold or distance_threshold == -1:
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
