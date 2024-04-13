import unittest
from data_structures.aba_elements import Rule, Atom, Example
import clingo.symbol as cs



class TestAtom(unittest.TestCase):

    def test_parse_atom(self):
        atom = "image(A)."
        parse_atom = Atom.parse_atom(atom).predicate
        parsed_result = cs.Function('image', [cs.Function('A', [], True)], True)
        self.assertEqual(parse_atom,parsed_result, "Parsing Error")

    def test_parse_binary_atom(self):
        atom = "in(A,B)."
        parse_atom = Atom.parse_atom(atom).predicate
        parsed_result = cs.Function('in', [cs.Function('A', [], True),cs.Function('B', [], True)], True)
        self.assertEqual(parse_atom,parsed_result, "Parsing Error")

    def test_parse_negative_atom(self):
        atom = "in(A,B)."
        parse_atom = Atom.parse_atom(atom,False).predicate
        parsed_result = cs.Function('in', [cs.Function('A', [], True),cs.Function('B', [], True)], False)
        self.assertEqual(parse_atom,parsed_result, "Parsing Error")

    def test_parse_negative_atom_string(self):
        atom = "in(A,B)."
        parse_atom = str(Atom.parse_atom(atom,False))
        str_atom = "not in(A,B)"
    
        self.assertEqual(parse_atom,str_atom, "Parsing Error")


class TestRule(unittest.TestCase):

    def test_parse_fact_rule_head(self):
        rule = "image(A) :- A=img_1."
        parse_rule = Rule.parse_rule(rule).head
        parsed_result = Atom(cs.Function('image', [cs.Function('A', [], True)], True))
        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_fact_rule_body(self):
        rule = "image(A) :- A=img_1."
        parse_rule = Rule.parse_rule(rule).body
        parsed_result = [cs.Function('=', [cs.Function('A', [], True),cs.Function('img_1', [], True)], True)]
        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_fact_string(self):
        rule = "image(A) :- A=img_1."
        parse_rule = Rule.parse_rule(rule)
        string_result = str(parse_rule)
        self.assertEqual(rule,string_result, "Parsing Error")

    def test_parse_binary_fact_head(self):
        rule = "in(A,B) :- A=img_1, B=entity_1."
        parse_rule = Rule.parse_rule(rule).head
        parsed_result = Atom(cs.Function('in', [cs.Function('A', [], True),cs.Function('B', [], True)], True))
        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_binary_fact_body(self):
        rule = "in(A,B) :- A=img_1, B=entity_1."
        parse_rule = Rule.parse_rule(rule).body
        parsed_result = [cs.Function('=', [cs.Function('A', [], True),cs.Function('img_1', [], True)], True),
                         cs.Function('=', [cs.Function('B', [], True), cs.Function('entity_1', [], True)], True)]
        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_binary_fact_string(self):
        rule = "in(A,B) :- A=img_1, B=entity1."
        parse_rule = Rule.parse_rule(rule)
        string_result = str(parse_rule)
        self.assertEqual(rule,string_result, "Parsing Error")

    def test_parse_number_fact_body(self):
        rule = "box(X1,Y1,X2,Y2) :- X1=0, Y1=0, X2=2, Y2=2."
        parse_rule = Rule.parse_rule(rule).body
        parsed_result = [cs.Function('=', [cs.Function('X1', [], True),cs.Number(0)], True),
                         cs.Function('=', [cs.Function('Y1', [], True),cs.Number(0)], True),
                         cs.Function('=', [cs.Function('X2', [], True),cs.Number(2)], True),
                         cs.Function('=', [cs.Function('Y2', [], True),cs.Number(2)], True),]
        
        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_number_fact_head(self):
        rule = "box(X1,Y1,X2,Y2) :- X1=0, Y1=0, X2=2, Y2=2."
        parse_rule = Rule.parse_rule(rule).head
        parsed_result = Atom(cs.Function('box', [cs.Function('X1', [], True),
                                                 cs.Function('Y1', [], True),
                                                 cs.Function('X2', [], True),
                                                 cs.Function('Y2', [], True)], True))

        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_number_fact_string(self):
        rule = "box(X1,Y1,X2,Y2) :- X1=0, Y1=0, X2=2, Y2=2."
        parse_rule = Rule.parse_rule(rule)
        string_result = str(parse_rule)

        self.assertEqual(rule,string_result, "Parsing Error")

    def test_parse_rule_head(self):
        rule = "bird(A) :- falcon(A)."
        parse_rule = Rule.parse_rule(rule).head
        parsed_result =  Atom(cs.Function('bird', [cs.Function('A', [], True)], True))

        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_rule_body(self):
        rule = "bird(A) :- falcon(A)."
        parse_rule = Rule.parse_rule(rule).body
        parsed_result =  [Atom(cs.Function('falcon', [cs.Function('A', [], True)], True))]

        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_rule_neg_body(self):
        rule = "bird(A) :- not falcon(A)."
        parse_rule = Rule.parse_rule(rule).body
        parsed_result =  [Atom(cs.Function('falcon', [cs.Function('A', [], True)], False))]

        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_rule_postive_string(self):
        rule = "bird(A) :- falcon(A)."
        parse_rule = Rule.parse_rule(rule)
        string_result = str(parse_rule)

        self.assertEqual(rule,string_result, "Parsing Error")

    def test_parse_rule_negative_string(self):
        rule = "bird(A) :- not falcon(A)."
        parse_rule = Rule.parse_rule(rule)
        string_result = str(parse_rule)

        self.assertEqual(rule,string_result, "Parsing Error")
  
    def test_parse_arithmetic_rule_head(self):
        rule = "bird_in_air(A) :- falcon(A), Y2 - Y1 > 0."
        parse_rule = Rule.parse_rule(rule).head
        parsed_result =  Atom(cs.Function('bird_in_air', [cs.Function('A', [], True)], True))

        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_arithmetic_rule_body(self):
        rule = "bird_in_air(A) :- falcon(A), Y2 - Y1 > 0."
        parse_rule = Rule.parse_rule(rule).body
        parsed_result =  [Atom(cs.Function('falcon', [cs.Function('A', [], True)], True)),
                          Atom(cs.Function('>', [cs.Function('Y2 - Y1', [], True), 
                                                 cs.Number(0)], True))]

        self.assertEqual(parse_rule,parsed_result, "Parsing Error")

    def test_parse_arithmetic_rule_string(self):
        rule = "bird_in_air(A) :- falcon(A), Y2 - Y1 > 0."
        parse_rule = Rule.parse_rule(rule)
        string_result = str(parse_rule)

        self.assertEqual(rule,string_result, "Parsing Error")

    def test_create_fact(self):
        pred_name = "in"
        arity = 2
        args = ["img_1", "entity_1"]
        created_fact = Rule.add_fact(pred_name,arity,args)
        generated_fact = "in(A,B) :- A=img_1, B=entity_1."
        self.assertEqual(str(created_fact), generated_fact , "Fact Generation Error")

    def test_create_fact_fails_when_arity_is_inconsistent(self):
        pred_name = "in"
        arity = 1
        args = ["img_1", "entity_1"]
        created_fact = Rule.add_fact(pred_name,arity,args)
        self.assertEqual(created_fact, None, "Fact Generation Error")

class TestExample(unittest.TestCase):

    def test_add_positive_example(self):
        example = Example.generate_example(pred="c",args=["img_1"],isPositive=True)
        result = "c(img_1)"
        self.assertEqual(str(example),result, "Parsing Error")

    def test_add_multiple_pred_example(self):
        example = Example.generate_example(pred="c",args=["img_1","small"],isPositive=True)
        result = "c(img_1,small)"
        self.assertEqual(str(example),result, "Parsing Error")



if __name__ == '__main__':
    unittest.main()