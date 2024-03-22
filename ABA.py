class ABA:

    def __init__(self,slots = 9,f_name = "aba_shape.pl",predict=False):

        self.num_slots = slots
        self.img_id = 1
        self.obj_id = 1
        self.ruleId = 1
        self.posId = 1
        self.negId = 1
        self.f_name = f_name
        self.plabels = []
        self.nlabels = []
        self.b_k = []
        self.predict = predict


    def init_aba_shape(self):

        with open(self.f_name, 'w') as file:
            file.write("% SHAPES ABA Framework")
            file.write('\n')
                
            file.close()

    
    def add_background_knowledge(self,entities,isPositve):
        # print(len(entities))
        
        for i in range(len(entities)):
            # print(i)
            if self.predict:
                shape, colour, size = entities[i]
            else:
                 _ ,shape, colour, size = entities[i]


            shape = shape.lower()
            colour = colour.lower()

            if shape == "":
                continue

            in_rule = f"in(A,B) :- A=img_{self.img_id},B={shape}_{self.obj_id}."
            shape_rule = f"{shape}({shape}_{self.obj_id})."
            colour_rule = f"{colour}({shape}_{self.obj_id})."
            img_rule = f"image(img_{self.img_id})."
            self.obj_id +=1
            
            self.b_k.append(in_rule)
            self.b_k.append(shape_rule)
            self.b_k.append(colour_rule)
            self.b_k.append(img_rule)

        if isPositve:
            self.plabels.append(f"c(img_{self.img_id})")
        else:
            self.nlabels.append(f"c(img_{self.img_id})")


        self.img_id += 1

    def add_pos_example(self):
        p_e = []
        for p in self.plabels:
            p_e.append(f"pos(p{self.posId}, c({p})).")
            self.posId += 1

        with open(self.f_name, 'a') as file:
            for item in p_e:
                file.write(item)
                file.write('\n')
            
            file.close()

    
    def add_neg_example(self,):
        n_e = []
        for n in self.nlabels:
            n_e.append(f"neg(p{self.negId}, c({n})).")
            self.negId += 1

        with open(self.f_name, 'a') as file:
            for item in n_e:
                file.write(item)
                file.write('\n')
            
            file.close()

    def get_ruleId(self):
        id = self.ruleId
        self.ruleId += 1
        return id
    
    def write_aba_framework(self):
        with open(self.f_name, 'a') as file:
            for item in self.b_k:
                file.write(item)
                file.write('\n')
            
            file.close()

    def generate_command(self):
        command = f"aba_asp('<aba_asp_directory>/{self.f_name}',[{', '.join(self.plabels)}],[{', '.join(self.nlabels)}])."
        print(command)
        return command
         