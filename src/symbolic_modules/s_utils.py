def extract_predicate(line):
    line = line.strip()
    if line.endswith('.'):
        line = line[:-1]  
    parts = line.split(":-", 1)
    predicate_part = parts[0].strip()
    predicate_parts = predicate_part.split("(", 1)
    predicate_name = predicate_parts[0].strip()

    if len(predicate_parts) > 1:
        predicate_arguments = [arg.strip() for arg in predicate_parts[1][:-1].split(",")]
    else:
        predicate_arguments = []
    return predicate_name, predicate_arguments

def order_facts(order_list, file_path):
    predicates = {}
    remaining_lines = []  
    with open(file_path, 'r') as file:
        for line in file:
            predicate_name, _ = extract_predicate(line)
            if predicate_name:
                if predicate_name not in predicates:
                    predicates[predicate_name] = []
                predicates[predicate_name].append(line.strip())
            else:
                remaining_lines.append(line.strip())

    ordered_predicates = []
    for predicate_name in order_list:
        if predicate_name in predicates:
            ordered_predicates.extend(predicates[predicate_name])

    return ordered_predicates, remaining_lines