def shapes_nal_inference(img_path: str, aba_path: str, include_pos = False):

    """
        Inference for Classification Task
        if predicate c(...) is in majority stable models we classify image as positve  
    """
    nal = get_shape_9_nal_model()
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

    ## Example Training and Inference
    test_img = "datasets/SHAPES/training_data/c1_s10/c1_10.png"
    aba_path = "shapes_9_bk_r1_SOLVED.aba"
    prediction = shapes_nal_inference(test_img,aba_path)

    if prediction:
        print("positive")
    else:
        print("negative")