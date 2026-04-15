import os
import pandas as pd

def create_run_directory(dataset_name, base_dir="results", run_prefix="run", use_id=True, specific_name=None):
    """
    Creates a unique directory for storing results.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'synthetic', 'shapeggen').
        base_dir (str): Base folder to store all results.
        run_prefix (str): Prefix for the run folder (e.g., 'run').
        use_id (bool): If True, increments an ID (e.g., 'run_001').
        specific_name (str): If provided, uses this exact name instead of an ID (e.g., 'run_seed_42').
        
    Returns:
        str: Absolute or relative path to the created run directory.
    """
    results_base_folder = os.path.join(base_dir, dataset_name)
    os.makedirs(results_base_folder, exist_ok=True)
    
    if specific_name is not None:
        run_folder_path = os.path.join(results_base_folder, specific_name)
        os.makedirs(run_folder_path, exist_ok=True)
        return run_folder_path

    if use_id:
        run_id = 1
        while True:
            run_folder_name = f"{run_prefix}_{run_id:03d}"
            run_folder_path = os.path.join(results_base_folder, run_folder_name)
            if not os.path.exists(run_folder_path):
                os.makedirs(run_folder_path)
                break
            run_id += 1
        return run_folder_path

    # Fallback to base folder if no run specific is given
    return results_base_folder

def save_metrics_to_csv(results_dict, file_path):
    """
    Saves a dictionary of results to a CSV file.
    """
    df = pd.DataFrame(results_dict).T
    df.to_csv(file_path, index=True)
    print(f"Metrics saved to: {file_path}")
    return df

def save_text_file(content_dict, file_path):
    """
    Saves dictionary keys and values as text to a file.
    """
    with open(file_path, "w") as f:
        for key, value in content_dict.items():
            print(f"{key}: {value}", file=f)
    print(f"Info saved to: {file_path}")
