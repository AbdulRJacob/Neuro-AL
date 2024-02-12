class ABA:

    def __init__(self,slots,isABALearn):

        self.num_slots = slots
        self.img_id = 1
        self.obj_id = 1
        self.ruleId = 1
        self.posId = 1
        self.negId = 1
        self.f_name = "aba_shapes.pl"
        self.plabels = []
        self.nlabels = []
        self.b_k = []


    def init_aba_shape(self):
    #     labels = {0: "top_left",
    #               1: "top_center",
    #               2: "top_right",
    #               3: "mid_left",
    #               4: "mid_center",
    #               5: "mid_right",
    #               6: "bot_left",
    #               7: "bot_center",
    #               8: "bot_right",
    #               }
        
    #     rules = []
        
    #     for i in range(self.num_slots):
    #         attr = labels[i]
    #         slot = f"slot{i}"
    #         rules.append(f"my_rule(r{self.ruleId},{attr}(X,Y), [{slot}(X,Y)]).")
    #         self.ruleId += 1

        with open(self.f_name, 'w') as file:
            file.write("% SHAPES ABA Framework")
            file.write('\n')
                
            file.close()

    
    def add_background_knowledge(self,entities,isPositve):
        
        for i in range(self.num_slots):
            _ ,shape, colour, size = entities[i]

            shape = shape.lower()
            colour = colour.lower()

            if shape == "":
                continue

            in_rule = f"my_rule(r{self.get_ruleId()},in(X,Y,Z),[X=img_{self.img_id}, Y={shape}, Z={shape}_{self.obj_id}])."
            colour_rule = f"my_rule(r{self.get_ruleId()},{colour}(X),[X={shape}_{self.obj_id}])."
            self.obj_id +=1
            
            self.b_k.append(in_rule)
            self.b_k.append(colour_rule)

        if isPositve:
            self.plabels.append(f"img_{self.img_id}")
        else:
            self.nlabels.append(f"img_{self.img_id}")


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
         