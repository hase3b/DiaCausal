import json
import pandas as pd
from modules.model_options import name_to_path

def compare_networks(learned_network, expert_model):

    with open(name_to_path[expert_model], 'r') as f:
        expert_model = json.load(f)

    with open(name_to_path[learned_network], 'r') as f:
        learned_network = json.load(f)

    expert_edges = set()
    for edge in expert_model['edges']:
        expert_edges.add((edge['from'], edge['to']))

    learned_edges = set()
    learned_edges_reversed = set()
    for edge in learned_network['edges']:
        learned_edges.add((edge['from'], edge['to']))
        learned_edges_reversed.add((edge['to'], edge['from']))

    results = []
    same_links = 0
    reverse_links = 0
    no_links = 0

    for i, edge in enumerate(expert_model['edges']):
        parent = edge['from']
        child = edge['to']
        link_status = ""

        if (parent, child) in learned_edges:
            link_status = "Same Link"
            same_links += 1
        elif (child, parent) in learned_edges_reversed:
            link_status = "Reverse Link"
            reverse_links += 1
        else:
            link_status = "No Link"
            no_links += 1
        
        results.append({
            "#": i + 1,
            "Parent": parent,
            "Child": child,
            "Learned Structure": link_status
        })

    df = pd.DataFrame(results)

    # Add summary rows
    summary_data = [
        {"#": "1", "Link Status": "Total Same Links:", "Count" : same_links},
        {"#": "2", "Link Status": "Total Reverse Links:", "Count" : reverse_links},
        {"#": "3", "Link Status": "Total No Links:", "Count" : no_links}
    ]
    df2 = pd.DataFrame(summary_data)

    return df, df2