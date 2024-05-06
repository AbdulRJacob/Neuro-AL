import os
from collections import Counter

from neuro_modules.NAL import NAL
from symbolic_modules.aba_framework import ABAFramework

def calculate_aba_classification_accuracy(shape_nal: NAL, classID: str, c_label: str): 
   correct_predictions = 0
   total_predictions = 100

   USE_GROUND_TRUTH  = False

   learnt_framework = shape_nal.aba_framework
   
   for i in range(200,300):
       
    img_path = f"datasets/SHAPES_4/testing_data/c{classID}_s{i}/c{classID}_{i}.png"
    lab_path = f"datasets/SHAPES_4/testing_data/c{classID}_s{i}/labels.txt"

   
    learnt_framework.reset_inference()
    prediction = shape_nal.run_slot_attention_model(img_path, 5)
    shape_nal.populate_aba_framework_inference(prediction)
    s_models = learnt_framework.get_prediction()

    if shape_nal.get_classificaiton_result(s_models,c_label):
        correct_predictions += 1
 
   print(f"Classification Accuracy for Class c{classID}: {correct_predictions / total_predictions} ")

   return correct_predictions / total_predictions


def metrics(predicted_list, actual_list):
   
    predicted_counts = Counter(predicted_list)
    actual_counts = Counter(actual_list)

    true_positives = sum(min(predicted_counts[item], actual_counts[item]) for item in predicted_counts)
    false_positives = sum(max(predicted_counts[item] - actual_counts[item], 0) for item in predicted_list)
    false_negatives = sum(max(actual_counts[item] - predicted_counts[item], 0) for item in actual_list)

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate Accuracy
    accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0.0


    return precision, recall, f1_score, accuracy

    


def calculate_slots_classification_accuracy(data_path, dataset_size):
    ## Assume data_path is a directory of folders of the form 
    ## sample_data
    ##   |---img.png
    ##   |---labels.txt

    final_precision = final_accucary = final_recall = final_f1 = 0
    shape_precision = shape_accucary = shape_recall = shape_f1 = 0
    colour_precision = colour_accucary = colour_recall = colour_f1 = 0
    size_precision = size_accucary = size_recall = size_f1 = 0

    avg_confidence = 0

    shape_nal = SHAPES_NAL()
    shape_nal.model_params = os.getcwd() + "/models/shapes_m1.pt"
    shape_nal.init_model()

    class_id = 1

    for i in range(1,dataset_size):

        k = i % 100
        offset = 900

        if (k == 0):
            class_id+=1
     
        img_path = data_path + f"c{class_id}_s{k+offset}/c{class_id}_{k+offset}.png"
        label_path = data_path + f"c{class_id}_s{k+offset}/labels.txt"

        pred_slots, conf = shape_nal.get_prediction(img_path)
        true_slots = shape_nal.get_ground_truth(label_path)

        slot_info_pred = [(B, C, D) for ((_, B, C, D), _ ) in pred_slots]
        slot_info_ground = [(B, C, D) for ((_, B, C, D), _ ) in true_slots]

        if i == 5:

            shape_nal.visualise_slots(img_path)
            print("Predicted:")
            print(slot_info_pred)
            print("Ground Truth: ")
            print(slot_info_ground)


        ## Slot Metrics
        precision, recall, f1, accuracy = metrics(slot_info_pred,slot_info_ground)

        final_precision += precision
        final_recall += recall
        final_f1 += f1
        final_accucary += accuracy

        avg_confidence += conf

        shape_pred, colour_pred, size_pred = [list(t) for t in zip(*slot_info_pred)]
        shape_ground, colour_ground, size_ground = [list(t) for t in zip(*slot_info_ground)]

        ## Shape Metric
        precision, recall, f1, accuracy = metrics(shape_pred,shape_ground)
        shape_precision += precision
        shape_recall += recall
        shape_f1 += f1
        shape_accucary += accuracy

        ## Colour Metric
        precision, recall, f1, accuracy = metrics(colour_pred,colour_ground)
        colour_precision += precision
        colour_recall += recall
        colour_f1 += f1
        colour_accucary += accuracy      

        ## Size Metric
        precision, recall, f1, accuracy = metrics(size_pred,size_ground)
        size_precision += precision
        size_recall += recall
        size_f1 += f1
        size_accucary += accuracy    


        
    print("-------Slot Classification -----------")
    print("Accuracy: ", final_accucary / dataset_size)
    print("Precision: ", final_precision / dataset_size)
    print("Recall: ", final_recall / dataset_size)
    print("F1 Measure: ", final_f1 / dataset_size)
    print("Confidence: ",avg_confidence / dataset_size)

    print("-------Slot Classification (shapes) -----------")
    print("Accuracy: ", shape_accucary / dataset_size)
    print("Precision: ", shape_precision / dataset_size)
    print("Recall: ", shape_recall / dataset_size)
    print("F1 Measure: ", shape_f1 / dataset_size)

    print("-------Slot Classification (colour) -----------")
    print("Accuracy: ", colour_accucary / dataset_size)
    print("Precision: ", colour_precision / dataset_size)
    print("Recall: ", colour_recall / dataset_size)
    print("F1 Measure: ", colour_f1 / dataset_size)

    print("-------Slot Classification (size) -----------")
    print("Accuracy: ", size_accucary / dataset_size)
    print("Precision: ", size_precision / dataset_size)
    print("Recall: ", size_recall / dataset_size)
    print("F1 Measure: ", size_f1 / dataset_size)


if __name__ == "__main__":  
    dataset_path = "datasets/SHAPES/testing_data/"
    dataset_size = 1000