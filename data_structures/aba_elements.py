from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import re
import string


import clingo.symbol as cs



@dataclass
class Atom:
    predicate: cs.Symbol

    def parse_atom(input: str, isPositive=True):

        is_arithmetic = bool(re.search(r'[-+*/<>=]',input))
        is_number =  lambda x : bool(re.match('[-+]?[0-9]*\.?[0-9]+',x))
        input = input.replace(".", "")

        if is_arithmetic:
             pattern = r'(<|>|<=|>=)'
             args = re.split(pattern, input)
             predicate_name = args.pop(1)
             atom = cs.Function(predicate_name, [cs.Number(int(arg.strip())) if is_number(arg.strip()) else cs.Function(arg.strip(), [], True) for arg in args])
        else:

            predicate_name, atom_args = input.split("(")

            atom_args = atom_args.strip(')').split(',')
            atom = cs.Function(predicate_name, [cs.Function(arg.strip(), [], True) for arg in atom_args], isPositive)

        return Atom(atom)
    
    def __str__(self):
        pred_name = self.predicate.name

        if pred_name in [">", "<", "<=", "=>"]:
            arithmetic = []
            for args in self.predicate.arguments:
                if args.type == cs.SymbolType.Function:
                    arithmetic.append(args.name)
                else:
                    arithmetic.append(str(args.number))

            arithmetic.insert(1,pred_name)

            return " ".join(arithmetic)
        
        else:
            return str(self.predicate).replace("-", "not ")


@dataclass
class Rule:
    rID: int
    head: Atom
    body: list[cs.Symbol | Atom]


    def split_and_clean_rule(input: str) -> tuple[str,list[tuple[str,bool]]]:
        if not ":-" in input:
            input.strip()
            input.replace(".", "")
            return (input, [])

        head, body = input.split(":-")
        head = head.strip()

        arguments = []
        
        for arg in re.split(r',(?![^()]*\))', body) :
            positive = True
            if "not" in arg:
                arg = arg.replace("not", "")
                positive = False

            arg = arg.strip()
            arg = arg.replace(".","")
            arguments.append((arg,positive))

        return (head,arguments)
    
    def parse_rule(input: str) ->Rule: 
        head_str, body_str = Rule.split_and_clean_rule(input)

        head = Atom.parse_atom(head_str)
        
        body = []

        for arg, postive in body_str:
            if "=" in arg:
                variable_name, value = arg.split("=")
                variable = cs.Function(variable_name.strip(), [])

                is_number = bool(re.match('[-+]?[0-9]*\.?[0-9]+',value))

                if is_number:
                    numeric_const = cs.Number(int(value.strip()))
                    body.append(cs.Function("=", [variable, numeric_const],postive))
                else:
                    alpha_const = cs.Function(value.strip(), [])
                    body.append(cs.Function("=", [variable, alpha_const],postive))
                
            else:
                body.append(Atom.parse_atom(arg,postive))

        return Rule(None,head,body)
    

    def add_fact(pred_name: str, arity: int, args :list[str]) -> Rule:
        ## Check:
        if (len(args) != arity):
            return None
        
        vars = [letter for letter in string.ascii_uppercase[:arity]]
        const_var_pairs = zip(vars,args)
        assignments = [f"{t[0]}={t[1]}" for t in const_var_pairs]

        head = pred_name + "(" + ','.join(vars) + ") :- "
        body = ', '.join(assignments) + "."

        rule_string = head + body

        return Rule.parse_rule(rule_string)
            

    def __str__(self):
        arguments = []

        for args in self.body:

            if isinstance(args, Atom):
                arguments.append(str(args))
            else:  
                body_preds, body_arg = args.arguments

                if body_arg.type == cs.SymbolType.Function:
                    res = f"{body_preds.name}={body_arg.name}"
                else:
                    res = f"{body_preds.name}={body_arg.number}"

                arguments.append(res)

        body = ', '.join(arguments)

        return f"{str(self.head)} :- {body}."
    

@dataclass
class Example:
    label: str
    pred: Atom
    isPositive: bool


    def generate_example(pred: str, args: list[str], isPositive: bool):
        example_string = pred + "(" + ",".join(args) + ")"
        atom = Atom.parse_atom(example_string)

        return Example(pred,atom,isPositive)



    def get_predicate(self) -> str:
        return str(self.pred)

    def get_label(self) -> str:
        return self.label

    def isPostiveExampe(self) -> bool:
        return self.isPositve

    def __str__(self):
        return str(self.pred)



    
    


