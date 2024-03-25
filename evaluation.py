from clingo.control import Control


def get_answer_sets(asp_file_path):

    ctl = Control()
    ctl.load(asp_file_path)

    ctl.ground([("base", [])])

    models = []
    def on_model(model):
        models.append(model.symbols(shown=True))

    ctl.solve(on_model=on_model)

    
    return models

def calculate_accuracy(predicted, true):
    correct_predictions = len(set(predicted) & set(true))
    total_predictions = len(predicted)
    accuracy = correct_predictions / total_predictions
    return accuracy



true_models = get_answer_sets("symbolic_modules/aba_asp/examples/true_smodels.asp")
pred_models = get_answer_sets("symbolic_modules/aba_asp/examples/bk_pred.sol.asp")
label_models = get_answer_sets("symbolic_modules/aba_asp/examples/bk_true.sol.asp")


filtered_true = []
filtered_pred = []
filtered_label = []


for symbol_sequence in true_models:
    for symbol in symbol_sequence:
        if "c(img_"in str(symbol): 
            filtered_true.append(symbol)

for symbol_sequence in pred_models:
    for symbol in symbol_sequence:
        if "c(img_"in str(symbol): 
            filtered_pred.append(symbol)

for symbol_sequence in label_models:
    for symbol in symbol_sequence:
        if "c(img_"in str(symbol): 
            filtered_label.append(symbol)


s_filtered_true = []
s_filtered_pred = []
s_filtered_label = []


for symbol_sequence in true_models:
    for symbol in symbol_sequence:
        if not "alpha_"in str(symbol): 
            s_filtered_true.append(symbol)

for symbol_sequence in pred_models:
    for symbol in symbol_sequence:
        if not "alpha_"in str(symbol): 
            s_filtered_pred.append(symbol)

for symbol_sequence in label_models:
    for symbol in symbol_sequence:
        if not "alpha_"in str(symbol): 
            s_filtered_label.append(symbol)




print("-------Filtering: Classification Predicate------------")
print("Slot Accuracy: ", calculate_accuracy(filtered_pred,filtered_true))
# print("Slot Precision: ", calculate_precision(filtered_pred,filtered_true))
print("Labels Accuracy: ", calculate_accuracy(filtered_label,filtered_true))
# print("Labels Precision: ", calculate_precision(filtered_label,filtered_true))

print("-------Without Filtering------------") ## Not indicative because of extra assumption involved.
print("Slot Accuracy: ", calculate_accuracy(pred_models[0],true_models[0]))
# print("Slot Precision: ", calculate_precision(pred_models,true_models))
print("Labels Accuracy: ", calculate_accuracy(label_models[0],true_models[0]))
# print("Laaels Precision: ", calculate_precision(label_models,true_models))


print("-------Filtering: No Alpha Predicate-----------")
print("Slot Accuracy: ", calculate_accuracy(s_filtered_pred,s_filtered_true))
# print("Slot Precision: ", calculate_precision(filtered_pred,filtered_true))
print("Labels Accuracy: ", calculate_accuracy(s_filtered_label,s_filtered_true))
# print("Labels Precision: ", calculate_precision(filtered_label,filtered_true))

