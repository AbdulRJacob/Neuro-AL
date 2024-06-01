import subprocess
import os
import re
import clingo.control as cc

from data_structures.aba_elements import Rule, Atom, Example
import symbolic_modules.s_utils as utils


class ABAFramework:

    def __init__(self) -> None:
        
        self.background_knowledge: dict[str, list[Rule]] = {}
        self.background_pred_names: list[str] = []
        self.inference: list[Rule] = []
        self.positive_examples: list[Example] = []
        self.negative_examples: list[Example] = []
        self.assumptions: list[Atom] = []
        self.contraries: list[tuple[Atom, Atom]] = []
        self.learnt_rules: list[Rule] = []
        self.aba_solver_path: str = ""
        self.filename: str = ""


    def set_aba_sovler_path(self, path: str) -> bool:
        if "aba_asp.pl" not in path:
            return False
        
        self.aba_solver_path = path
        return True 
    
    def add_bk_fact(self,label: str, pred_name: str, arity: int, args : list[str]):
        if label not in self.background_knowledge.keys():
            self.background_knowledge[label] = []

        new_rule = Rule.add_fact(pred_name,arity,args)
        new_rule.rID = len(self.background_knowledge[label])

        self.background_knowledge[label].append(new_rule)

    def add_bk_rule(self, rule: str, label="bk_rules", pred_name=""):
        if label not in self.background_knowledge.keys():
            self.background_knowledge[label] = []

        new_rule = Rule.parse_rule(rule)
        new_rule.rID = len(self.background_knowledge[label])

        self.background_knowledge[label].append(new_rule)

        if pred_name != "":
            self.background_pred_names.append(pred_name)
    

    def add_inference_bk_fact(self, pred_name: str, arity: int, args : list[str]):

            new_rule = Rule.add_fact(pred_name,arity,args)
            new_rule.rID = len(self.inference)

            self.inference.append(new_rule)

    def add_inference_bk_rule(self, rule: str,):

        new_rule = Rule.parse_rule(rule)
        new_rule.rID = len(self.inference)

        self.inference.append(new_rule)
    
    def reset_inference(self):
        self.inference = []

    def add_example(self, pred: str, args: list[str], isPositive: bool):
        eg = Example.generate_example(pred,args,isPositive)

        if isPositive:
            self.positive_examples.append(eg)
        else:
            self.negative_examples.append(eg)


    def write_aba_framework(self,filename: str):
        f = open(filename, "w")
        content = self.get_content()
        command = self.get_command(filename)
        f.write(content)
        f.write(command)
        f.write('\n')
        f.write('\n')
        f.close()

        self.filename = filename

    def write_solved_framework(self,filename: str):
        f = open(filename, "w")
        content = "% Background Knowledge \n"

        if "bk_rules" in self.background_knowledge.keys():
            for rule in self.background_knowledge["bk_rules"]:
                content += str(rule)
                content += "\n"

        content +=  "% Learnt Rules \n"
        for rule in self.learnt_rules:
            content += str(rule)
            content += "\n"


        f.write(content)
        f.close()

    def ground_aba_framework(self,filename: str):
        control = cc.Control()
        control.load(filename)

        control.ground([("base", [])])

        with open(filename, "w") as f:
            for atom in control.symbolic_atoms:
                f.write(str(atom.symbol) + ".\n")

            f.write('\n')
            f.write(self.get_command(filename))

    def write_ordered_facts(self,pred_list):
        
        ordered_predicates, remaining_lines = utils.order_facts(pred_list, self.filename)

        with open(self.filename, 'w') as file:
            for predicate in ordered_predicates:
                file.write(predicate + '\n')
            for line in remaining_lines:
                file.write(line + '\n')
    
        
    def get_content(self):
        content = "% Background Knowledge \n"

        for bk in list(self.background_knowledge.values()):
            for rule in bk:
                content += str(rule) + " \n"
        
        return content


    def get_command(self,filename: str) -> str:
        pos = [str(p) for p in self.positive_examples]
        neg = [str(n) for n in self.negative_examples]

        command = f"% aba_asp('{filename}',[{', '.join(pos)}],[{', '.join(neg)}])."

        return command
    
    def run_aba_framework(self,id = 0) -> bool:

        if self.aba_solver_path == "":
            print("Error: Need to set aba_solver_path")
            return False
        if self.filename == "":
            print("Error Need to write out aba framework")
            return False

        command = self.get_command(self.filename).replace(".aba", "").replace("%","")

        prolog_commands = f"""
            consult('{self.aba_solver_path}').
            {command}
            """
        
        log_name = f"logs_{id}.txt"
        log_file = open(log_name, 'w')

        process = subprocess.Popen(['swipl', '-q'], stdin=subprocess.PIPE, stdout=log_file, stderr=subprocess.PIPE, text=True)
        output, errors = process.communicate(prolog_commands)

        generated_files = ["asp.clingo","cc.pl","cc.clingo","clingo.stderr.log"]

        for file in generated_files:
            if os.path.exists(file):
                os.remove(file)


        filename_aba = self.filename.replace(".aba", ".sol.aba")
        filename_asp = self.filename.replace(".aba", ".sol.asp")


        res_aba = self.load_assumptions_and_contraries(filename_aba)
        res_asp = self.load_learnt_rules(filename_asp)

        if res_aba and res_asp:
            print("Framework Successfully learnt")
            self.write_solved_framework(self.filename.replace(".aba", "_SOLVED.aba"))

        return res_asp and res_aba
    

    def load_background_knowledge(self, filepath: str):
        self.background_knowledge = {}

        with open(filepath, "r") as f:
            for line in f:
                rule = line.strip()

                if '%' in rule or rule == "":
                    continue  # comment

                self.add_bk_rule(rule,"loaded_rule")

    def load_solved_framework(self, filepath: str):
        rule_type = 0     # 0 == BK | 1 == Learnt Rule

        with open(filepath, "r") as f:
            for line in f:
                rule = line.strip()

                if rule == "":
                    continue

                if rule == "% Background Knowledge":
                    rule_type = 0
                    continue

                if rule == "% Learnt Rules":
                    rule_type = 1
                    continue

                if rule_type == 0:
                    self.add_bk_rule(rule)
                else:
                    l_rule = Rule.parse_rule(rule)
                    l_rule.rID = len(self.learnt_rules)
                    self.learnt_rules.append(l_rule)


    def load_learnt_rules(self, filepath:str):
        if self.background_knowledge == {}:
            print("Error: Must load Background Knowledge")
            return False 

        
        b_k = [str(s) for s in sum(self.background_knowledge.values(),[])]

        f = open(filepath, "r")
        
        for line in f:
            rule  = line.strip()

            if rule in b_k or rule == "" or rule.split("(")[0] in self.background_pred_names:
                continue

            l_rule = Rule.parse_rule(rule)
            l_rule.rID = len(self.learnt_rules)
            self.learnt_rules.append(l_rule)

        
        f.close()
        return True


    def load_assumptions_and_contraries(self, filepath: str):
         
        if self.background_knowledge == {}:
            print("Error: Must load Background Knowledge")
            return False 


        # if self.positive_examples == []:
        #     print("Error: Must have postives to extract learnt rules")
        #     return False
        

        f = open(filepath, "r")
        
        b_k = [str(s) for s in sum(self.background_knowledge.values(),[])]

        for line in f:
            rule  = line.strip()

            if rule in b_k:
                continue
            
            if "assumption" in rule and not "c_alpha" in rule:
                rule = rule[:-1].split("(", 1)[1]
                self.assumptions.append(Atom.parse_atom(rule))
            
            if "contrary" in rule:
                rule = rule.split(":-")[0].strip()
                rule = rule[:-1].split("(", 1)[1]
                rule = re.split(r',(?![^()]*\))',rule)
                self.contraries.append((Atom.parse_atom(rule[0]),Atom.parse_atom(rule[1])))


        f.close()

        return True
    
    def compute_stable_models(self, restrict=""):
        if len(self.inference) == 0:
            print("Error: Need to have some background knowledge to get prediction")
            return False
        
        ctrl = cc.Control(["0"])

        if "bk_rules" in self.background_knowledge.keys():
            for rule in self.background_knowledge["bk_rules"]:
                ctrl.add("base", [], str(rule))

        for rule in self.inference:
            ctrl.add("base", [], str(rule))

        for rule in self.learnt_rules:
            ctrl.add("base", [], str(rule))

        if restrict != "":
             ctrl.add("base", [], restrict)


        ctrl.ground([("base", [])])


        models = []
        
        on_model = lambda x: models.append(x.symbols(shown=True))

       

        with ctrl.solve(yield_=True) as hnd:
            count = 0
            for m in hnd:
                count += 1
                on_model(m)

        return models