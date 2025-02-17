import glob
import pickle
import os
import json
import logging
import matplotlib.pyplot as plt
from experiments.NeuronSimilarity import NeuronSimilarity
import numpy as np

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG,  # Adjust to INFO or WARNING to reduce verbosity
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def process_experiment_file(experiment_file_path):
    """Run the experiment if not already processed."""
    logger.info(f"Processing experiment file: {experiment_file_path}")
    try:
        with open(experiment_file_path, 'rb') as f:
            experiment_data = pickle.load(f)
        NS = NeuronSimilarity(experiment_data)
        NS.run_experiment()
        logger.info(f"Experiment for {experiment_file_path} completed successfully.")
        
    except Exception as e:
        logger.error(f"Error running experiment for {experiment_file_path}: {e}")
        raise

def load_neuron_similarity(file_path):
    """Load and return the neuron similarity analysis from a pickle file."""
    logger.info(f"Loading analysis file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise

def extract_data_from_metrics(neuron_sim):
    """Extract data from the NeuronSimilarity metrics."""
    logger.info(f"Extracting data from neuron similarity metrics.")
    data = {}
    
    for step_key, step in neuron_sim.metrics.items():
        if step_key not in data:
            data[step_key] = []
        
        logger.debug(f"Processing step {step_key} with {len(step['average_similarities'])} layers.")
        for similarity in step['average_similarities']:
            layer_name = similarity['layer_name']
            avg_similarity = float(similarity['average_similarity'])  # Ensure we convert np.float32 to float
            
            data[step_key].append({
                'layer_name': layer_name,
                'average_similarity': avg_similarity
            })
    
    return data

def save_json_data(model_folder, data):
    """Save the data as JSON in the model's folder."""
    data_filename = os.path.join(model_folder, "non_zero_similarities_data.json")
    try:
        with open(data_filename, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Data saved as {data_filename}")
    except Exception as e:
        logger.error(f"Error saving data to {data_filename}: {e}")
        raise

def generate_and_save_plot_for_layer(layer_name, similarities, pruning_steps, model_folder):
    """Generate and save plot for a single layer."""
    plt.figure(figsize=(10, 6))
    similarities = np.clip(similarities, 0, 1)
    similarities = np.nan_to_num(similarities)
    plt.plot(pruning_steps, similarities, label=layer_name, marker='o', linestyle='-', markersize=6, linewidth=2)

    # Add horizontal line for baseline (e.g., 0)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Customize plot with better labels and titles
    plt.title(f"Non-Zero Neuron Similarity for Layer: {layer_name}", fontsize=16)
    plt.xlabel("Pruning Step", fontsize=14)
    plt.ylabel("Non-Zero Similarity", fontsize=14)

    # Adding grid lines for easier interpretation
    plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)

    # Adjust the legend and plot style
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.ylim(-.1, 1.1)

    # Save the individual plot for the current layer
    plot_filename = os.path.join(model_folder, f"non_zero_similarity_{layer_name}.png")
    try:
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free memory
        logger.info(f"Plot saved as {plot_filename}")
    except Exception as e:
        logger.error(f"Error saving plot for {layer_name}: {e}")
        raise

def process_model(fileset):
    """Process a model's analysis and generate results."""
    
    modelname = fileset.split("/")[4][:fileset.split("/")[4].find("_pretrain")]
    logger.info(f"Extracted model name: {modelname}")

    # Define the full path to the analysis file
    file_path = fileset[:fileset.rfind("/")] + "/neuron_similarity/neuron_similarity.pkl"
    logger.info(f"Looking for analysis file: {file_path}")

    # Check if the analysis file exists, if not load and run the experiment
    if not glob.glob(file_path+"*"):
        logger.warning(f"Analysis file not found for {modelname}. Attempting to run the experiment...")
        experiment_files = glob.glob(file_path[:file_path.rfind("cuda")+4]+"/*pkl")
        
        if experiment_files:
            experiment_file = experiment_files[0]
            process_experiment_file(experiment_file)
    else:
        logger.info(f"Analysis file found for {modelname}")
    
    # Load neuron similarity analysis
    neuron_sim = load_neuron_similarity(file_path)

    # Extract data from neuron similarity analysis
    data = extract_data_from_metrics(neuron_sim)

    # Prepare for plotting
    layer_names = {entry['layer_name'] for key in data for entry in data[key]}  # Get unique layer names
    logger.debug(f"Found {len(layer_names)} unique layers.")
    
    # Create a dictionary to store non-zero similarities for each layer
    non_zero_similarities = {layer_name: [] for layer_name in layer_names}
    pruning_steps = list(data.keys())  # List of pruning steps

    # Create a folder for the model to save the plots and data
    model_folder = os.path.join('./plots', modelname)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        logger.info(f"Created folder for model: {model_folder}")
    else:
        logger.info(f"Folder already exists for model: {model_folder}")

    # Save the data as JSON
    save_json_data(model_folder, non_zero_similarities)

    # Populate non-zero similarities for each layer and step, and generate plot in situ
    for step in pruning_steps:
        for entry in data[step]:
            layer_name = entry['layer_name']
            avg_similarity = entry['average_similarity']
            non_zero_similarities[layer_name].append(avg_similarity)
        
        # Generate plot for each layer as data for that layer is processed
        for layer_name, similarities in non_zero_similarities.items():
            generate_and_save_plot_for_layer(layer_name, similarities, pruning_steps, model_folder)

def main():
    """Main function to process all analysis files."""
    analysis_path = "/scratch/jgafur/LTH_output/*_pretrain10_finetune10_steps21_batch64_devicecuda/*.pkl"
    logger.info(f"Found analysis files: {glob.glob(analysis_path)}")
    for fileset in glob.glob(analysis_path)[::-1]:
        if not "vgg" in fileset.lower():
            try:
                process_model(fileset)
            except Exception as e:
                logger.error(f"Error processing model from {fileset}: {e}")

if __name__ == "__main__":
    main()
