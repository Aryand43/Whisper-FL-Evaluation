import torch
import os
import argparse
from tqdm import tqdm
from typing import List, Dict, Union, Any
import logging

# Set up logging for better error reporting
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _recursive_add_weighted_tensors(target_dict: Dict[str, Any], source_dict: Dict[str, Any], weight: float):
    """
    Recursively traverses dictionaries, finds weight tensors, and adds the 
    weighted contribution from the source to the target dictionary.
    
    This addresses the concern of 'preconverged weights' by ensuring only 
    terminal tensor data (weights) are combined, skipping nested metadata.
    """
    
    for key, target_value in target_dict.items():
        source_value = source_dict.get(key)
        
        if source_value is None:
            # If a key exists in the target but not the source, just preserve the target value.
            logging.debug(f"Key '{key}' missing in source dictionary. Preserving target value.")
            continue

        if isinstance(target_value, dict) and isinstance(source_value, dict):
            # Recursive Case: Both are dictionaries, continue traversal
            _recursive_add_weighted_tensors(target_value, source_value, weight)
        
        elif isinstance(target_value, torch.Tensor) and isinstance(source_value, torch.Tensor):
            # Base Case: Found Tensors (model weights). Perform weighted addition.
            if target_value.shape == source_value.shape:
                try:
                    target_value.add_(source_value * weight)
                except RuntimeError as e:
                    logging.error(f"Tensor operation failed for key '{key}': {e}")
                    raise
            else:
                # Shape mismatch indicates incompatible model layers (critical error for aggregation)
                logging.error(f"Tensor shape mismatch for key '{key}': Target {target_value.shape} vs Source {source_value.shape}. Stopping.")
                raise ValueError("Incompatible model architectures detected.")
        
        # Non-tensor, non-dictionary values (e.g., config integers, strings) are skipped and 
        # preserved from the first model (target_dict)
        elif not (isinstance(target_value, dict) or isinstance(target_value, torch.Tensor)):
             logging.debug(f"Skipping metadata key '{key}'.")
             pass # Skip non-tensor/non-dict metadata


class WeightAggregator:
    """Utility class for loading, combining, and saving model weights with partitioning."""
    
    def __init__(self, model_paths: List[str], weights: List[float]):
        if len(model_paths) != len(weights):
            raise ValueError("The number of model paths must match the number of weights.")
            
        # Normalize weights to sum to 1.0 (for proper partitioning)
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        self.model_paths = model_paths
        logging.info(f"Normalized Weights (Partitions): {self.weights}")

    def load_state_dicts(self) -> List[Dict[str, torch.Tensor]]:
        """Loads all state dictionaries directly from the file paths provided."""
        state_dicts = []
        
        logging.info("Starting model state dictionary loading...")
        for i, path in enumerate(tqdm(self.model_paths, desc="Loading Models")):
            try:
                # Load the state dictionary to CPU to prevent VRAM issues during aggregation
                state_dict = torch.load(path, map_location='cpu')
                state_dicts.append(state_dict)
                logging.info(f"Successfully loaded state dict from: {path} (Model {i+1})")
            except Exception as e:
                logging.error(f"CRITICAL LOAD FAILURE for {path}: {e}")
                raise RuntimeError(f"Model loading failed for {path}") from e

        if len(state_dicts) != len(self.model_paths):
             logging.error("Mismatch: Not all models were successfully loaded.")
        return state_dicts

    def combine_weights(self, state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combines the loaded state dictionaries using the configured weights and recursion."""
        if not state_dicts:
            raise ValueError("Cannot combine weights: no state dictionaries were loaded.")

        # --- Step 1: Initialize the target dictionary (Model 1 weighted) ---
        weighted_state_dict = {}
        first_state_dict = state_dicts[0]
        first_weight = self.weights[0]
        
        logging.info("Initializing target state dictionary with first model's weighted contribution...")
        try:
            for key, tensor in tqdm(first_state_dict.items(), desc="Initial Weighting"):
                if isinstance(tensor, torch.Tensor):
                    weighted_state_dict[key] = tensor.clone() * first_weight
                else:
                    # Preserve metadata from the first model
                    weighted_state_dict[key] = tensor 

        except Exception as e:
            logging.error(f"CRITICAL INITIALIZATION FAILURE: {e}")
            raise RuntimeError("Initialization failed.") from e

        # --- Step 2: Recursively add contributions from remaining models ---
        logging.info("Recursively aggregating weights from remaining models...")
        for i, source_dict in enumerate(tqdm(state_dicts[1:], desc="Aggregating Models")):
            weight = self.weights[i + 1]
            try:
                # Use the recursive helper to add the weighted contribution
                _recursive_add_weighted_tensors(weighted_state_dict, source_dict, weight)
            except Exception as e:
                logging.error(f"Aggregation failed for Model {i+2}: {e}")
                raise RuntimeError("Weight combination failed due to nested error.") from e

        logging.info("Weighted combination complete.")
        return weighted_state_dict

    def save_model(self, state_dict: Dict[str, Any], output_path: str):
        """Saves the final aggregated state dictionary to a .pt file."""
        try:
            logging.info(f"Attempting to save aggregated model to: {output_path}")
            torch.save(state_dict, output_path)
            logging.info(f"Successfully saved aggregated model to {output_path}")
        except Exception as e:
            logging.error(f"CRITICAL SAVE FAILURE: {e}")
            raise RuntimeError("Model saving failed.") from e

def main():
    """Main function to parse arguments and execute the aggregation pipeline."""
    parser = argparse.ArgumentParser(
        description="Weighted aggregation of multiple fine-tuned Whisper model checkpoints."
    )
    parser.add_argument(
        '--models', 
        nargs='+', 
        required=True, 
        help='List of direct paths to model checkpoint files (e.g., model_A.pt).'
    )
    parser.add_argument(
        '--weights', 
        nargs='+', 
        type=float, 
        required=True, 
        help='List of corresponding weights (e.g., 0.6 0.2 0.1). Must match the number of models.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='aggregated_model.pt', 
        help='File path to save the final aggregated PyTorch (.pt) checkpoint.'
    )
    
    args = parser.parse_args()

    # --- Main Pipeline Execution with Global Try/Except for Debugging ---
    try:
        # Step 1: Initialize Aggregator
        aggregator = WeightAggregator(args.models, args.weights)

        # Step 2: Load State Dictionaries
        state_dicts = aggregator.load_state_dicts()

        # Step 3: Combine Weights
        aggregated_weights = aggregator.combine_weights(state_dicts)

        # Step 4: Save Final Model
        aggregator.save_model(aggregated_weights, args.output)
        
        logging.info("--- Pipeline Completed Successfully ---")

    except Exception as e:
        # Catch any uncaught critical error from the previous steps
        logging.critical(f"FATAL ERROR in main execution: {e}. Pipeline terminated.")


if __name__ == "__main__":
    # Prevent accidental CUDA usage during initial CPU-bound tensor manipulation
    os.environ['CUDA_VISIBLE_DEVICES'] = '' 
    main()