import os
import numpy as np

from datasets.SHAPES.SHAPES import SHAPES


def generate_dataset(rule, label, num_of_images):
    dir = ["datasets/MOCK/training_data/","datasets/MOCK/testing_data/"]

    for directory in dir:
        if not os.path.exists(directory):
            os.makedirs(directory)

    
    num_train, num_test = num_of_images
    ## Generate Training Data
    shape_generator = SHAPES(300,300,dir[0]) 

    shape_generator.generate_shape(rule, False, label, num_train)

    shape_generator.change_filepath(dir[1])

    shape_generator.generate_shape(rule,False,label,num_test,num_train)


if __name__ == "__main__":
   
   print("--------- Generating SHAPES ---------")

   rule = ["c(A) :- blue(O1), square(O1), small(O1), in(A,O1), green(O2), triangle(O2), in(A,O2), image(A)"]
   n_rule = ["c(A) :- not exception(A)", "exception(A) :- blue(O1), square(O1), small(O1), in(A,O1), green(O2), triangle(O2), in(A,O2), image(A)"]
   train_test_split = (200,100)
   label = "c20"

   generate_dataset(rule,label,train_test_split)

