import torch
import os
import argparse
from tqdm import tqdm
from typing import List, Dict, Union, Any
import logging

# Set up logging for better error reporting
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info("--- AGGREGATOR SCRIPT INITIALIZED ---")

class WeightAggregator:
    """
    Utility class for loading, combining, and saving model weights with partitioning.
    Includes logic to strip common wrapper keys during loading and fix model prefixes.
    """
    
    def __init__(self, model_paths: List[str], weights: List[float]):
        if len(model_paths) != len(weights):
            raise ValueError("The number of model paths must match the number of weights.")
            
        # Normalize weights to sum to 1.0 (for proper partitioning)
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        self.model_paths = model_paths
        logging.info(f"Normalized Weights (Partitions): {self.weights}")

    def load_state_dicts(self) -> List[Dict[str, torch.Tensor]]:
        """
        Loads state dictionaries, unloads wrappers, and ensures keys have the 
        correct 'model.' prefix expected by WhisperForConditionalGeneration.
        """
        state_dicts = []
        
        logging.info("Starting model state dictionary loading and unwrapping...")
        for i, path in enumerate(tqdm(self.model_paths, desc="Loading Models")):
            try:
                # Load the state dictionary to CPU to prevent VRAM issues during aggregation
                state_dict = torch.load(path, map_location='cpu')
                
                # 1. UNWRAP COMMON CHECKPOINT STRUCTURES
                if 'model_state_dict' in state_dict:
                    logging.warning(f"Model {i+1}: Found and extracted 'model_state_dict' wrapper.")
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    logging.warning(f"Model {i+1}: Found and extracted 'state_dict' wrapper.")
                    state_dict = state_dict['state_dict']
                
                # 2. FIX KEY PREFIX (CRITICAL FIX FOR EVALUATE.PY)
                # If the first key does not start with 'model.', we assume the HuggingFace prefix is missing.
                first_key = next(iter(state_dict.keys()), None)
                if first_key and not first_key.startswith('model.'):
                    logging.info(f"Model {i+1}: Fixing missing 'model.' prefix in keys.")
                    
                    new_state_dict = {}
                    for key, tensor in state_dict.items():
                        # Exclude non-tensor keys (like 'dims', 'optimizer') if they somehow remain
                        if isinstance(tensor, torch.Tensor):
                            new_state_dict['model.' + key] = tensor
                    state_dict = new_state_dict
                
                # 3. Final Metadata Cleanup (optional keys)
                if 'dims' in state_dict:
                    del state_dict['dims']
                    
                state_dicts.append(state_dict)
                logging.info(f"Successfully prepared state dict from: {path} (Model {i+1})")
            
            except Exception as e:
                logging.critical(f"CRITICAL LOAD FAILURE for {path}: {e}")
                raise RuntimeError(f"Model loading failed for {path}") from e

        if len(state_dicts) != len(self.model_paths):
             logging.error("Mismatch: Not all models were successfully loaded.")
        return state_dicts

    def combine_weights(self, state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combines the loaded state dictionaries using the configured weights and recursion."""
        
        if not state_dicts:
            raise ValueError("Cannot combine weights: no state dictionaries were loaded.")

        # Initialize the averaged state dict with the first model's weights * its partition
        weighted_state_dict = {}
        first_state_dict = state_dicts[0]
        first_weight = self.weights[0]
        
        logging.info("Initializing target state dictionary with first model's weighted contribution...")
        
        # Check if helper is available (defined below in the real file structure)
        if '_recursive_add_weighted_tensors' not in globals():
             logging.critical("FATAL: _recursive_add_weighted_tensors helper function is missing from global scope.")
             raise NotImplementedError("Recursive helper is missing.")


        try:
            for key, tensor in tqdm(first_state_dict.items(), desc="Initial Weighting"):
                if isinstance(tensor, torch.Tensor):
                    weighted_state_dict[key] = tensor.clone() * first_weight
                else:
                    # Preserve metadata from the first model
                    weighted_state_dict[key] = tensor 

            # Recursively add contributions from remaining models
            logging.info("Recursively aggregating weights from remaining models...")
            for i, source_dict in enumerate(tqdm(state_dicts[1:], desc="Aggregating Models")):
                weight = self.weights[i + 1]
                _recursive_add_weighted_tensors(weighted_state_dict, source_dict, weight)

            logging.info("Weighted combination complete.")
            return weighted_state_dict
            
        except Exception as e:
            logging.critical(f"CRITICAL COMBINATION FAILURE: {e}")
            raise RuntimeError("Weight combination failed.") from e

    def save_model(self, state_dict: Dict[str, Any], output_path: str):
        """Saves the final aggregated state dictionary to a .pt file."""
        try:
            logging.info(f"Attempting to save aggregated model to: {output_path}")
            torch.save(state_dict, output_path)
            logging.info(f"Successfully saved aggregated model to {output_path}")
        except Exception as e:
            logging.critical(f"CRITICAL SAVE FAILURE: {e}")
            raise RuntimeError("Model saving failed.") from e

# --- Recursive Helper Function (Must be defined outside the class) ---

def _recursive_add_weighted_tensors(target_dict: Dict[str, Any], source_dict: Dict[str, Any], weight: float):
    """Recursively traverses and performs weighted tensor addition."""
    
    for key, target_value in target_dict.items():
        source_value = source_dict.get(key)
        
        if source_value is None:
            continue

        if isinstance(target_value, dict) and isinstance(source_value, dict):
            _recursive_add_weighted_tensors(target_value, source_value, weight)
        
        elif isinstance(target_value, torch.Tensor) and isinstance(source_value, torch.Tensor):
            if target_value.shape == source_value.shape:
                try:
                    target_value.add_(source_value * weight)
                except RuntimeError as e:
                    logging.error(f"Tensor operation failed for key '{key}': {e}")
                    raise
            else:
                logging.error(f"Tensor shape mismatch for key '{key}': Target {target_value.shape} vs Source {source_value.shape}. Stopping.")
                raise ValueError("Incompatible model architectures detected.")
        
        # Skip non-tensor/non-dict values

def main():
    """Main function to parse arguments and execute the aggregation pipeline."""
    # Ensure torch is not using CUDA devices before main to prevent accidental GPU memory usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '' 
    
    parser = argparse.ArgumentParser(
        description="Weighted aggregation of multiple fine-tuned Whisper model checkpoints."
    )
    parser.add_argument(
        '--models', 
        nargs='+', 
        required=True, 
        help='List of root directories for each model (or direct path to .pt file).'
    )
    parser.add_argument(
        '--weights', 
        nargs='+', 
        type=float, 
        required=True, 
        help='List of corresponding weights (e.g., 0.6 0.2 0.2). Must match the number of models.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='aggregated_model.pt', 
        help='File path to save the final aggregated PyTorch (.pt) checkpoint.'
    )
    
    args = parser.parse_args()

    # --- Main Pipeline Execution with Global Try/Except ---
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
        # This catches any uncaught critical error from the previous steps
        logging.critical(f"FATAL ERROR in main execution: {e}. Pipeline terminated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR IN MAIN EXECUTION: {e}")