import yaml
import random

import training.shapes_train_symbolic as sym

def shapes_nal_inference(img_path: str, aba_path: str, include_pos = False):

    """
        Inference for Classification Task
        if predicate c(...) is in majority stable models we classify image as positve  
    """

    nal = sym.get_shape_9_nal_model()
    NUM_SLOTS = 10

    prediction, _ = nal.run_slot_attention_model(img_path,NUM_SLOTS)
    nal.populate_aba_framework_inference(prediction,include_pos)

    nal.load_framework(aba_path)
    all_models = nal.run_learnt_framework()

    total_model = len(all_models)
    present = 0


    for model in all_models:
        for symbol in model:
            if f"c(img_"in str(symbol): 
                present+=1

        
        absent = total_model - present
        return present > absent
    
    return 0


if __name__ == "__main__":

    with open("config/shapes_config.yaml", 'r') as file:
        config = yaml.safe_load(file)


    data_dir = config['data']['data_dir']
    num_of_imgs = config['data']['dataset_size']

    model_dir = config['training']['model_dir']
    rule_id = config['sym_training']['rule_id']

    rule_to_class = {1 : [1,2], 2: [3,4], 3:[5,6], 4:[7,8], 5:[9,10]}

    img_id = random.randint(num_of_imgs,num_of_imgs + num_of_imgs -1)

    ## Example Training and Inference
    test_img = data_dir + f"testing_data/c{rule_to_class[rule_id][0]}_s{img_id}/c{rule_to_class[rule_id][0]}_{img_id}.png"
    aba_path = model_dir + f"shapes_9_bk_r{rule_to_class[rule_id][0]}_SOLVED.aba"

    print("Testing with image " + test_img + " using aba framework " + aba_path)
    prediction = shapes_nal_inference(test_img,aba_path)

    if prediction:
        print("positive")
    else:
        print("negative")