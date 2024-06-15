import os
import numpy as np

from datasets.SHAPES_9.SHAPES import SHAPES
from datasets.SHAPES_9.SHAPES import SHAPESDATASET


def generate_dataset_SHAPES9(rule, label, num_of_images, isPostive):
    dir = ["../../../../../../mnt/d/fyp/SHAPES_9/training_data/","../../../../../../mnt/d/fyp/SHAPES_9/testing_data/"]

    for directory in dir:
        if not os.path.exists(directory):
            os.makedirs(directory)

    
    num_train, num_test = num_of_images
    ## Generate Training Data
    shape_generator = SHAPES(300,300,dir[0]) 

    shape_generator.generate_shape(rule, isPostive, label, num_train)

    shape_generator.change_filepath(dir[1])

    shape_generator.generate_shape(rule, isPostive, label, num_test, num_train)


def generate_dataset_SHAPES4(rule, label, num_of_images, isPostive):
    dir = ["../../../../../../mnt/d/fyp/SHAPES_4/training_data/","../../../../../../mnt/d/fyp/SHAPES_4/testing_data/"]

    for directory in dir:
        if not os.path.exists(directory):
            os.makedirs(directory)

    
    num_train, num_test = num_of_images
    ## Generate Training Data
    shape_generator = SHAPES_4(200,200,dir[0]) 

    shape_generator.generate_shape(rule, isPostive, label, num_train)

    shape_generator.change_filepath(dir[1])

    shape_generator.generate_shape(rule, isPostive, label, num_test, num_train)


if __name__ == "__main__":
   
   print("--------- Generating SHAPES 9 ---------")


   train_test_split = (1000,500)
   labels = [["c1","c2"],["c3","c4"],["c5","c6"],["c7","c8"],["c9","c10"],["c11","c12"]]
   rules = [["c(A) :- blue(O1), square(O1), in(A,O1), image(A)"],
            ["c(A) :- green(O1), triangle(O1), small(O1) in(A,O1), image(A)"],
            ["c(A) :- blue(O1), triangle(O1), in(A,O1), red(O2), circle(O2), large(O2) in(A,O2), image(A)"],
            ["c(A) :- blue(O1), square(O1), in(A,O1), green(O2), triangle(O2), in(A,O2), above(O1,O2), image(A)"],
            ["c(A) :- red(O1), triangle(O1), in(A,O1), green(O2), circle(O2), small(O2), in(A,O2), left(O1,O2), image(A)"],
            ["c(A) :- not exception1(A), image(A).", "exception1(A) :- blue(O1), circle(O1),in(A,O1), image(A)."]]
   
   for i in range(0,len(labels)):
        rule = rules[i] 
        label = labels[i]
        l1 = label[0]
        l2 = label[1]

        pos1 = int(l1.replace("c","")) % 2 == 0
        pos2 = int(l2.replace("c","")) % 2 == 0

        generate_dataset_SHAPES9(rule,l1,train_test_split,pos1)
        generate_dataset_SHAPES9(rule,l2,train_test_split,pos2)



