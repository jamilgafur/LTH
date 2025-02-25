import glob
import pickle
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from experiments.NeuronSimilarity import NeuronSimilarity

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG,  # Adjust to INFO or WARNING to reduce verbosity
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def process_experiment_file(experiment_file_path):
    """Run the experiment if not already processed."""
    with open(experiment_file_path, 'rb') as f:
        experiment_data = pickle.load(f)
    NS = NeuronSimilarity(experiment_data)
    NS.run_experiment()
    logger.info(f"Experiment for {experiment_file_path} completed successfully.")

def load_neuron_similarity(file_path):
    """Load and return the neuron similarity analysis from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_data_from_metrics(neuron_sim):
    """Extract data from the NeuronSimilarity metrics."""
    data = {}
    for step_key, step in neuron_sim.metrics.items():
        if step_key not in data:
            data[step_key] = []
        
        for i, similarity in enumerate(step['average_similarities']):
            layer_name = similarity['layer_name']
            avg_similarity = float(similarity['average_similarity'])
            similarity_matrix = np.array(step['similarity_matrices'][i]['similarity_matrix'])

            # Subtract identity matrix and calculate mean of max values for each row
            identity_matrix = np.eye(similarity_matrix.shape[0])
            adjusted_similarity_matrix = similarity_matrix - identity_matrix
            mean_max = float(np.mean(np.max(adjusted_similarity_matrix, axis=1)))
            
            data[step_key].append({
                'layer_name': layer_name,
                'average_similarity': avg_similarity,
                'mean_max': mean_max
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

def generate_and_save_plots(model_folder, non_zero_similarities, pruning_steps):
    """Generate and save plots for the non-zero similarities, with error bars."""
    for layer_name, similarities in non_zero_similarities.items():
        plt.figure(figsize=(10, 6))
        # similarities = np.clip(similarities, 0, 1)
        # similarities = np.nan_to_num(similarities)

        # Calculate the error bars (upper and lower)
        lower_error = np.array(similarities['mean_max']) - np.array(similarities['Q25'])
        upper_error = np.array(similarities['Q75']) - np.array(similarities['mean_max'])
        
        # Plot with error bars
        plt.errorbar(pruning_steps, similarities['mean_max'], yerr=[lower_error, upper_error],
                     label='Mean Max', fmt='o', linestyle='-', markersize=6, linewidth=2, capsize=5)

        # Add horizontal line for baseline (e.g., 0)
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

        # Customize plot with better labels and titles
        plt.title(f"Non-Zero Neuron Similarity for Layer: {layer_name}", fontsize=16)
        plt.xlabel("Pruning Sparsity", fontsize=14)
        plt.ylabel("Similarity", fontsize=14)

        plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.ylim(-0.1, 1.1)

        # Save the individual plot for the current layer
        plot_filename = os.path.join(model_folder, f"non_zero_similarity_{layer_name}.png")
        try:
            plt.savefig(plot_filename)
            logger.info(f"Plot saved as {plot_filename}")
        except Exception as e:
            logger.error(f"Error saving plot for {layer_name}: {e}")
            raise

        # Close the plot after saving to avoid memory issues
        plt.close()

def ensure_folder_exists(folder_path):
    """Ensure the folder exists and create it if it doesn't."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Created folder: {folder_path}")
    else:
        logger.info(f"Folder already exists: {folder_path}")

def load_or_run_experiment(file_path, modelname):
    """Check if analysis exists, or if not, run the experiment."""
    if not glob.glob(file_path+"*"):
        logger.warning(f"Analysis file not found for {modelname}. Attempting to run the experiment...")
        experiment_files = glob.glob(file_path[:file_path.rfind("cuda")+4]+"/*pkl")
        if experiment_files:
            process_experiment_file(experiment_files[0])
    else:
        logger.info(f"Analysis file found for {modelname}")

def process_model(fileset, plotdata="mean_max"):
    """Process a model's analysis and generate results."""
    
    modelname = fileset.split("/")[4][:fileset.split("/")[4].find("_pretrain")]
    logger.info(f"Extracted model name: {modelname}")

    # Define the full path to the analysis file
    file_path = fileset[:fileset.rfind("/")] + "/neuron_similarity/neuron_similarity.pkl"
    logger.info(f"Looking for analysis file: {file_path}")

    # Load or run experiment based on file existence
    load_or_run_experiment(file_path, modelname)
    
    # Load neuron similarity analysis
    neuron_sim = load_neuron_similarity(file_path)

    # Extract data from neuron similarity analysis
    data = extract_data_from_metrics(neuron_sim)

    # Prepare for plotting
    layer_names = {entry['layer_name'] for key in data for entry in data[key]}  # Get unique layer names
    logger.debug(f"Found {len(layer_names)} unique layers.")
    
    # Create a dictionary to store non-zero similarities for each layer
    non_zero_similarities = {layer_name: {'mean_max': [], 'Q25': [], 'Q75': []} for layer_name in layer_names}
    pruning_steps = list(data.keys())  # List of pruning steps

    # Populate non-zero similarities, Q25, and Q75 for each layer and step
    for step in pruning_steps:
        for entry in data[step]:
            layer_name = entry['layer_name']
            avg_similarity = entry[plotdata]
            non_zero_similarities[layer_name]['mean_max'].append(avg_similarity)
            
            # Calculate the 25th and 75th percentiles for the current pruning step's similarity values
            layer_similarities = [entry['mean_max'] for entry in data[step] if entry['layer_name'] == layer_name]
            Q25 = np.percentile(layer_similarities, 25)
            Q75 = np.percentile(layer_similarities, 75)
            
            non_zero_similarities[layer_name]['Q25'].append(Q25)
            non_zero_similarities[layer_name]['Q75'].append(Q75)

    # Create a folder for the model to save the plots and data
    model_folder = os.path.join('./plots', modelname)
    ensure_folder_exists(model_folder)

    # Save the data as JSON
    save_json_data(model_folder, non_zero_similarities)

    # Generate and save plots for each layer
    generate_and_save_plots(model_folder, non_zero_similarities, pruning_steps)

def main():
    """Main function to process all analysis files."""
    for model_name in ["LeNet", "ResNet20"]:
        process_data = "mean_max"
        analysis_path = f"/scratch/jgafur/LTH_output/{model_name}_pretrain3_finetune3_steps3_batch64_devicecuda/*.pkl"
        logger.info(f"Found analysis files: {glob.glob(analysis_path)}")
        for fileset in glob.glob(analysis_path)[::-1]:
            if not "vgg" in fileset.lower():
                process_model(fileset, process_data)

if __name__ == "__main__":
    main()
