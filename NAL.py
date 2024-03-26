from neuro_modules.SHAPES import SHAPES, SHAPESDATASET
from ABA import ABA
import os
import numpy as np
import neuro_modules.slots_eval as se
import subprocess



def generate_dataset(rule, label, num_of_images):
    dir = ["datasets/SHAPES/training_data/","datasets/SHAPES/testing_data/"]

    for directory in dir:
        if not os.path.exists(directory):
            os.makedirs(directory)

    
    num_train, num_test = num_of_images
    ## Generate Training Data
    shape_generator = SHAPES(300,300,dir[0]) 

    shape_generator.generate_shape(rule, False, label, num_train)

    shape_generator.change_filepath(dir[1])

    shape_generator.generate_shape(rule,False,label,num_test,num_train)



def get_true_slot_info(img_id, labels =""):
    with open(labels, "r") as file:
        lines = file.readlines()

        data = []
        for line in lines:
            elements = line.strip().split(",") 
            if len(elements) == 4: 
                data.append(tuple(elements))

        
        return data
    
def get_predicted_slot_info(img_path):
    return se.get_predicted_symbols(img_path)
    

def image_to_background(slot_nums, slots_info,isPostive):
    pass

if __name__ == "__main__":
   
   print("--------- Generating SHAPES ---------")
   rule = ["c(A) :-red(O1),circle(O1),in(A,O1), not exception1(A), not exception2(A).", "exception1(A) :- red(O2),circle(O2),small(O2), in(A,O2), image(A).", "exception2(A) :- green(O3), circle(O3), in(A,O3), green(O4), triangle(O4), in(A,O4), green(O5), square(O5), in(A,O5), image(A)."]
   train_test_split = (900,100)
   label = "c10"
   # generate_dataset(rule,label,train_test_split)

   

   se.get_classification_accuracy(os.getcwd() + "/datasets/SHAPES/testing_data/",1000)

#    aba_framework_1 = ABA(f_name="bk_true.aba")   
#    aba_framework_2 = ABA(f_name="bk_pred.aba", predict=True)
#    aba_framework_1.init_aba_shape()
#    aba_framework_2.init_aba_shape()

#    for i in range(0,10):
#        is_pos = i <= 4
#        aba_framework_1.add_background_knowledge(get_true_slot_info(i+1,f"datasets/testing_data/test_s{i}/labels.txt"),is_pos)
#        aba_framework_2.add_background_knowledge(get_predicted_slot_info(f"datasets/testing_data/test_s{i}/test_{i}.png"),is_pos)
       
#    aba_framework_1.write_aba_framework()
#    aba_framework_2.write_aba_framework()
#    aba_framework_1.generate_command()
#    aba_framework_2.generate_command()

#    result = subprocess.run(["python3", "evaluation.py"], capture_output=True, text=True)
#    print(result.stdout)
