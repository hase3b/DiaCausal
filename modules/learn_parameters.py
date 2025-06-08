import pandas as pd
import json
import os
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.readwrite import BIFWriter
import logging

# Configure logging to suppress pgmpy INFO messages
logging.getLogger('pgmpy').setLevel(logging.WARNING)

# Read Data (CSV File)
def read_data(csv_path):
  data = pd.read_csv(csv_path)
  return data.astype(int)

# Function to load network structure from custom JSON format
def load_structure_from_json(json_path):
    """Loads network structure (nodes and edges) from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            network_data = json.load(f)
        
        nodes = [node['id'] for node in network_data.get('nodes', [])]
        edges = [(edge['from'], edge['to']) for edge in network_data.get('edges', [])]
        
        model = DiscreteBayesianNetwork(ebunch=edges)
        # Add any nodes that might not be part of an edge
        model.add_nodes_from(nodes)
        
        return model
    except FileNotFoundError:
        print(f"Error: Network structure file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return None
    except Exception as e:
        print(f"Error loading structure from {json_path}: {e}")
        return None

# Main function
def param_learning(data_path, input_dir, output_dir, estimator_type='mle', equivalent_sample_size=10):
    """Loads structures, learns parameters, and saves CBNs."""
    # Read data
    data = read_data(data_path)

    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each network structure file in the input directory
    found_files = False
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            found_files = True
            json_path = os.path.join(input_dir, filename)
        
            # Load structure
            model_structure = load_structure_from_json(json_path)
            
            if model_structure:
                # Check if all nodes in the structure exist in the data
                missing_nodes = [node for node in model_structure.nodes() if node not in data.columns]
                if missing_nodes:
                    print(f"Error: Nodes {missing_nodes} from structure '{filename}' not found in data columns.")
                    print("Skipping parameter learning for this structure.")
                    continue
                    
                # Select relevant columns from data
                model_data = data[list(model_structure.nodes())]
                
                # Learn parameters
                try:
                    if estimator_type.lower() == 'mle':
                        estimator = MaximumLikelihoodEstimator(model=model_structure, data=model_data)
                        model_structure.fit(data=model_data, estimator=type(estimator))
                    elif estimator_type.lower() == 'bayes':
                        estimator = BayesianEstimator(model=model_structure, data=model_data)
                        model_structure.fit(data=model_data, estimator=type(estimator), prior_type='BDeu', equivalent_sample_size=equivalent_sample_size)
                    else:
                        print(f"Error: Unknown estimator type '{estimator_type}'. Use 'mle' or 'bayes'.")
                        continue # Skip this file
                    
                    # Save the learned model (CBN) in BIF format
                    output_filename_base = os.path.splitext(filename)[0]
                    output_path = os.path.join(output_dir, f"{output_filename_base}.bif")
                    writer = BIFWriter(model_structure)
                    writer.write_bif(output_path)
                    
                except Exception as e:
                    print(f"Error during parameter learning or saving for {filename}: {e}")
            else:
                 print(f"Skipping file {filename} due to loading errors.")

    if not found_files:
        print(f"No .json files found in the input directory '{input_dir}'.")