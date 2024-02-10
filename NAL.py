from SHAPES import SHAPES
from ABA import ABA



def get_image():
    shape_generator = SHAPES(300,300,"")
    shape_generator.generate_shape("A blue square", "testF",1)


def get_slot_info(img_id, labels =""):
    with open(labels, "r") as file:
        lines = file.readlines()

        data = []
        for line in lines:
            elements = line.strip().split(",") 
            if len(elements) == 4: 
                data.append(tuple(elements))

        
        return data

def image_to_background(slot_nums, slots_info,isPostive):
    pass

if __name__ == "__main__":
   get_image()
#    aba = ABA(9,True)
#    aba.init_aba_shape()
#    aba.add_background_knowledge(get_slot_info(0,"test_s0/labels.txt"),True)
#    aba.add_background_knowledge(get_slot_info(1,"testF_s0/labels.txt"),False)
#    aba.write_aba_framework()
#    aba.add_pos_example()
#    aba.add_neg_example()