#  Copyright (c) 2019-2023 by Latent AI Inc.
#  All rights reserved.
#  This file is part of the LEIP(tm) SDK,
#  and is released under the "Latent AI Commercial Software License". Please see the LICENSE
#  file that should have been included as part of this package.


def post_process_torch(
    decoded_output, iou_threshold, max_det_per_image, prediction_confidence_threshold,  k=None
):
    import torch
    from torchvision.ops.boxes import batched_nms

    batch_detections = []
    for i in range(decoded_output.shape[0]): # for each batch, find detections
        sampled_decoded_output = decoded_output[i]

        if k:
            _, indices = torch.topk(sampled_decoded_output[:, 5], k, sorted=False)
            sampled_decoded_output = sampled_decoded_output[indices]

        boxes, classes, scores = sampled_decoded_output[:, :4], sampled_decoded_output[:, 4], sampled_decoded_output[:, 5]

        # Filter based on confidence threshold
        filtered_scores = scores > prediction_confidence_threshold
        boxes = boxes[filtered_scores]
        scores = scores[filtered_scores]
        classes = classes[filtered_scores]
        # test for zero filtered values
        
        # NMS
        top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=iou_threshold)

        # Filter based on NMS & Top K
        top_detection_idx = top_detection_idx[:max_det_per_image]
        boxes = boxes[top_detection_idx]
        scores = scores[top_detection_idx, None]
        classes = classes[top_detection_idx, None] + 1  # back to class idx with background class = 0

        detections = torch.cat([boxes, scores, classes.float()], dim=1)
        # # Filter based on confidence threshold
        # mask = detections[:,4] > prediction_confidence_threshold
        batch_detections.append(detections)
    
    return batch_detections


def take_top_class_torch(decoded_output):
    import torch

    # Extract bounding boxes
    boxes = decoded_output[:, :, :4]
    
    # Extract class scores and find top classes and scores
    class_scores = decoded_output[:, :, 4:]
    top_classes = torch.argmax(class_scores, dim=2)
    top_scores = torch.max(class_scores, dim=2).values
    
    # Stack transformed detections
    transformed_decoded_output = torch.cat([boxes, top_classes.unsqueeze(2).float(), top_scores.unsqueeze(2)], dim=2)
    
    return transformed_decoded_output


def postprocess(decoded_output, max_det_per_image, prediction_confidence_threshold, iou_threshold, k, config):
    if config.postprocess_library_torch:
        import torch as T
        decoded_output_torch = T.from_dlpack(decoded_output) 
        transformed_decoded_output_torch = take_top_class_torch(decoded_output_torch)

        # Do postprocessing: confidence based filtering -> NMS -> top K filtering
        batch_detections_torch = post_process_torch(transformed_decoded_output_torch, iou_threshold, max_det_per_image, prediction_confidence_threshold, k)
        return batch_detections_torch
    else:
        raise RuntimeError(f"Function does not exist for config.postprocess_library_torch {config.postprocess_library_torch}.")
    
