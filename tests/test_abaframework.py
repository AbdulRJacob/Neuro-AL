import unittest
import clingo.symbol as cs

from data_structures.aba_elements import Rule, Atom, Example
from data_structures.aba_framework import ABAFramework


ABA_ASP_PATH = "../symbolic_modules/aba_asp/aba_asp.pl"
FILEPATH = "test_resources/"


class TestABAFramework(unittest.TestCase):

    def get_test_aba_framework_1(self):
        aba_framework = ABAFramework()
        aba_framework.set_aba_sovler_path(ABA_ASP_PATH)

        
        aba_framework.add_bk_fact(label="img_1",pred_name="in",arity=2, args=["img_1","square_1"])
        aba_framework.add_bk_fact(label="img_1",pred_name="square",arity=1, args=["square_1"])
        aba_framework.add_bk_fact(label="img_1",pred_name="image",arity=1, args=["img_1"])
        aba_framework.add_bk_fact(label="img_2",pred_name="in",arity=2, args=["img_2","circle_2"])
        aba_framework.add_bk_fact(label="img_2",pred_name="circle",arity=1, args=["circle_2"])
        aba_framework.add_bk_fact(label="img_2",pred_name="image",arity=1, args=["img_2"])
        aba_framework.add_bk_fact(label="img_3",pred_name="in",arity=2, args=["img_3","triangle_3"])
        aba_framework.add_bk_fact(label="img_3",pred_name="triangle",arity=1, args=["triangle_3"])
        aba_framework.add_bk_fact(label="img_3",pred_name="image",arity=1, args=["img_3"])

        aba_framework.add_example(pred="c",args=["img_1"],isPositive=True)
        aba_framework.add_example(pred="c",args=["img_2"],isPositive=False)
        aba_framework.add_example(pred="c",args=["img_3"],isPositive=False)

        return aba_framework
    
    def get_test_aba_framework_2(self):
        aba_framework = ABAFramework()
        aba_framework.set_aba_sovler_path(ABA_ASP_PATH)

        
        aba_framework.add_bk_fact(label="img_1",pred_name="in",arity=2, args=["img_1","circle_1"])
        aba_framework.add_bk_fact(label="img_1",pred_name="circle",arity=1, args=["circle_1"])
        aba_framework.add_bk_fact(label="img_1",pred_name="blue",arity=1, args=["circle_1"])
        aba_framework.add_bk_fact(label="img_1",pred_name="image",arity=1, args=["img_1"])
        aba_framework.add_bk_fact(label="img_2",pred_name="in",arity=2, args=["img_2","square_3"])
        aba_framework.add_bk_fact(label="img_2",pred_name="in",arity=2, args=["img_2","circle_2"])
        aba_framework.add_bk_fact(label="img_2",pred_name="blue",arity=1, args=["circle_2"])
        aba_framework.add_bk_fact(label="img_2",pred_name="blue",arity=1, args=["square_3"])
        aba_framework.add_bk_fact(label="img_2",pred_name="square",arity=1, args=["square_3"])
        aba_framework.add_bk_fact(label="img_2",pred_name="circle",arity=1, args=["circle_2"])
        aba_framework.add_bk_fact(label="img_2",pred_name="image",arity=1, args=["img_2"])
   

        aba_framework.add_example(pred="c",args=["img_1"],isPositive=True)
        aba_framework.add_example(pred="c",args=["img_2"],isPositive=False)
  

        return aba_framework

    def test_adding_background_facts(self):
        aba_framework = ABAFramework()

        aba_framework.add_bk_fact(label="img_1",pred_name="image",arity=1, args=["img_1"])
        b_k = {'img_1': [Rule(0, Atom(cs.Function('image', [cs.Function('A', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('img_1', [], True)], True)])]}


        self.assertEqual(aba_framework.background_knowledge, b_k, "Error adding facts into background knowledge")

    def test_adding_multiple_background_facts_1(self):
        aba_framework = ABAFramework()

        aba_framework.add_bk_fact(label="img_1",pred_name="image",arity=1, args=["img_1"])
        aba_framework.add_bk_fact(label="img_1",pred_name="in",arity=2, args=["img_1","circle_1"])
        aba_framework.add_bk_fact(label="img_1",pred_name="shape",arity=1, args=["circle_1"])

        b_k = {'img_1': [Rule(0, Atom(cs.Function('image', [cs.Function('A', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('img_1', [], True)], True)]), 
                         Rule(1, Atom(cs.Function('in', [cs.Function('A', [], True), cs.Function('B', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('img_1', [], True)], True), cs.Function('=', [cs.Function('B', [], True), cs.Function('circle_1', [], True)], True)]), 
                         Rule(2, Atom(cs.Function('shape', [cs.Function('A', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('circle_1', [], True)], True)])]
                }

        self.assertEqual(aba_framework.background_knowledge, b_k, "Error adding facts into background knowledge")

    def test_adding_multiple_background_facts_2(self):
        aba_framework = ABAFramework()

        aba_framework.add_bk_fact(label="img_1",pred_name="image",arity=1, args=["img_1"])
        aba_framework.add_bk_fact(label="img_1",pred_name="in",arity=2, args=["img_1","circle_1"])
        aba_framework.add_bk_fact(label="img_1",pred_name="circle",arity=1, args=["circle_1"])

        aba_framework.add_bk_fact(label="img_2",pred_name="image",arity=1, args=["img_2"])
        aba_framework.add_bk_fact(label="img_2",pred_name="in",arity=2, args=["img_2","square_1"])
        aba_framework.add_bk_fact(label="img_2",pred_name="square",arity=1, args=["square_1"])

        b_k = {'img_1': [Rule(0, Atom(cs.Function('image', [cs.Function('A', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('img_1', [], True)], True)]), 
                         Rule(1, Atom(cs.Function('in', [cs.Function('A', [], True), cs.Function('B', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('img_1', [], True)], True), cs.Function('=', [cs.Function('B', [], True), cs.Function('circle_1', [], True)], True)]), 
                         Rule(2, Atom(cs.Function('circle', [cs.Function('A', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('circle_1', [], True)], True)])],
               'img_2': [Rule(0, Atom(cs.Function('image', [cs.Function('A', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('img_2', [], True)], True)]), 
                         Rule(1, Atom(cs.Function('in', [cs.Function('A', [], True), cs.Function('B', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('img_2', [], True)], True), cs.Function('=', [cs.Function('B', [], True), cs.Function('square_1', [], True)], True)]), 
                         Rule(2, Atom(cs.Function('square', [cs.Function('A', [], True)], True)), [cs.Function('=', [cs.Function('A', [], True), cs.Function('square_1', [], True)], True)])]
                }

        self.assertEqual(aba_framework.background_knowledge, b_k, "Error adding facts into background knowledge")


    def test_adding_background_rule(self):
        aba_framework = ABAFramework()
        rule = "c(A) :- in(A,B)."

        aba_framework.add_bk_rule(rule)
        b_k = {'bk_rules': [Rule(0, Atom(cs.Function('c', [cs.Function('A', [], True)], True)), [Atom(cs.Function('in', [cs.Function('A', [], True), cs.Function('B', [], True)], True))])]}


        self.assertEqual(aba_framework.background_knowledge, b_k, "Error adding rules into background knowledge")

    def test_add_examples(self):
        aba_framework = ABAFramework()

        aba_framework.add_example(pred="c",args=["img_1"],isPositive=True)
        result_pos = [Example('c',Atom(cs.Function('c', [cs.Function('img_1', [], True)], True)), True)]

        aba_framework.add_example(pred="c",args=["img_2"],isPositive=False)
        result_neg = [Example('c',Atom(cs.Function('c', [cs.Function('img_2', [], True)], True)), False)]

        self.assertEqual(aba_framework.positive_examples, result_pos, "Error adding postive examples into background knowledge")
        self.assertEqual(aba_framework.negative_examples, result_neg, "Error adding negative example into background knowledge")


    def test_running_aba_framework_with_no_assumption(self):
        aba_framework = self.get_test_aba_framework_1()
        filename = FILEPATH + "test1_bk.aba"
        aba_framework.write_aba_framework(filename)

        success = aba_framework.run_aba_framework()
        learnt_rule = Rule.parse_rule("c(A) :- square(B), in(A,B).")
        learnt_rule.rID = 0
        learnt_rules = [learnt_rule]

        self.assertEqual(success, True, "Error Running ABA Framework")
        self.assertEqual(learnt_rules, aba_framework.learnt_rules, "Error Populating Learnt Rules")
        self.assertEqual([], aba_framework.assumptions, "Error Should be no assumptions")
        self.assertEqual([], aba_framework.contraries, "Error Should be no contraries")

    def test_running_aba_framework_with_assumption(self):
        aba_framework = self.get_test_aba_framework_2()
        filename = FILEPATH + "test2_bk.aba"
        aba_framework.write_aba_framework(filename)

        success = aba_framework.run_aba_framework()

        self.assertEqual(success, True, "Error Running ABA Framework")
        self.assertGreater(len(aba_framework.learnt_rules), 0, "Error Populating Learnt Rules")
        self.assertGreater(len(aba_framework.assumptions), 0, "Error Should have some assumptions")
        self.assertGreater(len(aba_framework.contraries), 0, "Error Should have some contraries")

    
    def test_loading_aba_framework(self):
        filename_bk = FILEPATH + "test2_bk.aba"
        filename_aba = FILEPATH + "test2_bk.sol.aba"
        filename_asp= FILEPATH + "test2_bk.sol.asp"

        aba_framework = ABAFramework()

        aba_framework.load_background_knowledge(filename_bk)
        self.assertGreater(len(aba_framework.background_knowledge["loaded_rule"]), 0, "Error Populating Background Knowledge")

        aba_framework.load_learnt_rules(filename_asp)
        self.assertGreater(len(aba_framework.learnt_rules), 0, "Error Populating Learnt Rules")

        aba_framework.load_assumptions_and_contraries(filename_aba)
        self.assertGreater(len(aba_framework.assumptions), 0, "Error Should have some assumptions")
        self.assertGreater(len(aba_framework.contraries), 0, "Error Should have some contraries")
        


if __name__ == '__main__':
    unittest.main()