import yaml
import random

import training.clevr_train_symbolic as sym



def check_predicate_presence(s_models, total_model, pred_name="c"):
    present = 0

    for model in s_models:
        for symbol in model:
            if f"{pred_name}(img_"in str(symbol): 
                present+=1
                break

        
    absent = total_model - present
    
    return present > absent


def clevr_nal_inference(img_path: str, aba_path: list[str], class_order, include_pos = False):

    """
        CLEVR Hans Classification Task

        img_path: images to perform inference on
        aba_path: list of lenght 2. Inference for CLEVR-hans is perfomed in two passes using both learnt rules
        include_pos: Boolean denoting whether to add co-oodrinates to the framework 

        Task:
            Idenfication of concept predicate c(...) denotes positive instance of class 

    
    """

    nal = sym.get_clevr_nal_model()
    NUM_SLOTS = 11

    prediction, _ = nal.run_slot_attention_model(img_path,NUM_SLOTS,num_coords=3,isPath=True)
    nal.populate_aba_framework_inference(prediction,include_pos)

    # First Pass
    nal.load_framework(aba_path[0])
    all_models = nal.run_learnt_framework()
    total_model = len(all_models)

    has_predicate = check_predicate_presence(all_models,total_model)
    
    if has_predicate:
        return class_order[0]
    
    # Second Pass
    nal.load_framework(aba_path[1])
    all_models = nal.run_learnt_framework()
    total_model = len(all_models)
   
    has_predicate = check_predicate_presence(all_models,total_model)

    if has_predicate:
        return class_order[1]
    else:
        return class_order[2]


if __name__ == "__main__":

    with open("config/clevr_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process an image file.")

    # Add the --image argument
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")

    # Parse the arguments
    args = parser.parse_args()

    test_img = args.image
    class_order = config["sym_training"]["class_order"]

    pass_1_framework = config["inference"]["first_pass"]
    pass_2_framework = config["inference"]["second_pass"]

    result = clevr_nal_inference(test_img,[pass_1_framework,pass_2_framework],class_order)

    print("===============================")
    print("Predicted Class: " + str(result))
    print("===============================")





