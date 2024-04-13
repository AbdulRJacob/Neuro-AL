

def get_classificaiton_result(s_models, c_label) -> bool: 
    is_instance = False

    for symbol_sequence in s_models:
        for symbol in symbol_sequence:
            if f"{c_label}(img_"in str(symbol): 
                is_instance = True

    return is_instance

from NAL_Shapes.shapes_nal import SHAPES_NAL
from data_structures.aba_framework import ABAFramework

def calculate_aba_classification_accuracy(learnt_framework: ABAFramework, classID: str, c_label: str): 
   correct_predictions = 0
   total_predictions = 100

   USE_GROUND_TRUTH  = False

   shape_nal = SHAPES_NAL()
   shape_nal.aba_framework = learnt_framework
   
   for i in range(200,300):
       
    img_path = f"datasets/MOCK/testing_data/c{classID}_s{i}/c{classID}_{i}.png"
    lab_path = f"datasets/MOCK/testing_data/c{classID}_s{i}/labels.txt"

   
    learnt_framework.reset_inference()
    prediction, _ = shape_nal.get_prediction(img_path)
    shape_nal.populate_aba_framework_inference(prediction)
    s_models = learnt_framework.get_prediction()[0]

    if get_classificaiton_result(s_models,c_label):
        correct_predictions += 1
 

   print(f"Classification Accuracy for Class c{classID}: {correct_predictions / total_predictions} ")
    


def calculate_slots_classification_accuracy(data_path, dataset_size):
    ## Assume data_path is a directory of folders of the form 
    ## sample_data
    ##   |---img.png
    ##   |---labels.txt

    total_slot = 0
    total_correct_slot = 0
    missed_slots = 0
    # total_correct_img = 0

    class_id = 1

    for i in range(1,dataset_size):

        k = i % 100
        offset = 900

        if (k == 0):
            class_id+=1
     
        img_path = data_path + f"c{class_id}_s{k+offset}/c{class_id}_{k+offset}.png"
        label_path = data_path + f"c{class_id}_s{k+offset}/labels.txt"

        pred_slots = SHAPES_NAL.get_prediction(img_path)
        true_slots = SHAPES_NAL.get_ground_truth(label_path)

        if i == 84:
            SHAPES_NAL.get_visualisation(img_path)
            print("Predicted:")
            print(pred_slots)
            print("Ground Truth: ")
            print(true_slots)

        # same = all([pred_slots[key] == true_slots[key]for key in true_slots])


        for j in range(0,9):
            if  j not in pred_slots.keys():
                missed_slots += 1
                total_slot+=1
                continue
            else:
                if  pred_slots[j] == true_slots[j]:
                    total_correct_slot+=1
            
            total_slot+=1

        
    print("-------Slot Classification Accuracy-----------")
    print("Slot Classification: ", total_correct_slot / total_slot)
    print("Missing slot pecentage: ", missed_slots / total_slot)
