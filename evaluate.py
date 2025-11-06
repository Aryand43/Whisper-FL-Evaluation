import torch
import argparse
import os
import evaluate
import wandb
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
import re
import logging
from typing import List, Dict, Any
from getpass import getpass

# Set up logging for console visibility
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class EvaluationManager:
    """
    Manages the full lifecycle of model evaluation: W&B login, data/model loading, 
    transcription, and corpus-level metric calculation (WER/BLEU).
    """

    def __init__(self, model_path: str, run_name: str, dataset_name: str, batch_size: int, device: str):
        self.model_path = model_path
        self.run_name = run_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device
        self.wer_metric = evaluate.load("wer")
        self.bleu_metric = evaluate.load("bleu")
        self.processor = None
        self.model = None

    def _login_to_wandb(self):
        """Prompts the user for W&B API key and initializes the run."""
        # Check environment variable first (best practice)
        if 'WANDB_API_KEY' not in os.environ:
            logging.warning("WANDB_API_KEY not found. Prompting user for key.")
            try:
                # Use getpass for non-echoing input in interactive environments (like Colab)
                api_key = getpass("Please enter your Weights & Biases API Key: ")
            except Exception:
                api_key = input("Please enter your Weights & Biases API Key: ")
            
            if api_key:
                os.environ['WANDB_API_KEY'] = api_key
            else:
                logging.error("W&B API key not provided. Skipping logging.")
                return False

        try:
            # Initialize W&B with project and run name
            wandb.init(project="Whisper-FL-Evaluation", name=self.run_name, job_type="evaluation")
            logging.info(f"Initialized W&B run: {self.run_name}")
            return True
        except Exception as e:
            logging.critical(f"Failed to initialize W&B: {e}")
            return False

    def _normalize_text(self, text: str) -> str:
        """Applies basic text normalization (lowercase, punctuation removal for German ASR)."""
        text = text.lower()
        # Remove common German punctuation but keep words and umlauts
        text = re.sub(r"[^\w\säöüß]", "", text)  
        text = re.sub(r"\s\s+", " ", text).strip() # Remove extra spaces
        return text

    def _load_and_prepare_data(self):
        """Loads the i4ds/spc_r dataset and prepares the 'test' split."""
        logging.info(f"Loading and preparing dataset: {self.dataset_name}...")
        try:
            # Load the specific 'test' split (679 rows)
            raw_datasets = load_dataset(self.dataset_name, split="test") 
            raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))

            def preprocess_function(example):
                audio = example["audio"]
                # Process audio array into input features (log-mel spectrograms)
                example["input_features"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
                # Tokenize the ground truth text for padding later
                example["labels"] = self.processor.tokenizer(example["text"]).input_ids
                return example

            processed_datasets = raw_datasets.map(
                preprocess_function, 
                remove_columns=raw_datasets.column_names, 
                num_proc=os.cpu_count()
            )
            logging.info(f"Dataset preparation complete. Total test samples: {len(processed_datasets)}")
            return processed_datasets
            
        except Exception as e:
            logging.critical(f"CRITICAL DATASET FAILURE: {e}. Check dataset name or internet connection.")
            return None

    def _load_model_and_processor(self):
        """Loads the Whisper-Medium architecture and injects the aggregated weights."""
        logging.info(f"Loading model architecture and processor, mapping to {self.device}...")
        try:
            # Load the Whisper-Medium architecture (as the base for training)
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(self.device)

            # Load the custom aggregated weights from the .pt file
            state_dict = torch.load(self.model_path, map_location=self.device)
            # Inject weights into the architecture
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval() # Set model to evaluation mode
            logging.info(f"Successfully loaded and injected aggregated weights from: {self.model_path}")
            return True

        except Exception as e:
            logging.critical(f"CRITICAL MODEL LOADING FAILURE: {e}. Check if the .pt file is correctly formatted or if VRAM is sufficient.")
            return False

    def _calculate_and_log_metrics(self, predictions: List[str], references: List[str]):
        """Calculates corpus-level WER/BLEU and logs results to W&B."""
        logging.info("Calculating corpus-level metrics...")
        
        # --- WER (Word Error Rate) ---
        # The WER metric calculates the total edit distance across the entire corpus of normalized text
        wer_score = self.wer_metric.compute(predictions=predictions, references=references) * 100
        
        # --- BLEU Score (requires list of list of references) ---
        bleu_references = [[r] for r in references]
        bleu_results = self.bleu_metric.compute(predictions=predictions, references=bleu_references)
        
        logging.info(f"Corpus Word Error Rate (WER): {wer_score:.2f}%")
        logging.info(f"Corpus BLEU Score: {bleu_results['bleu']:.2f}")

        # W&B Logging
        if wandb.run:
            wandb.log({
                "corpus_wer": wer_score,
                "corpus_bleu": bleu_results["bleu"],
                # Logs precision scores for 1-gram up to 4-gram for plotting
                "bleu_precision_weights": {f"{i+1}-gram": p for i, p in enumerate(bleu_results["precisions"])}, 
            })
            logging.info("Metrics successfully logged to W&B.")

    def run_pipeline(self):
        """Executes the full evaluation pipeline, wrapping all logic in safety blocks."""
        if not self._login_to_wandb():
            return

        # Use an outer try block to ensure wandb.finish() is called on catastrophic failures
        try:
            if not self._load_model_and_processor():
                return

            processed_datasets = self._load_and_prepare_data()
            if processed_datasets is None:
                return

            # --- Inference Loop Setup ---
            predictions = []
            references = []
            logging.info("Starting inference (transcription)...")
            
            # Data collator handles dynamic padding for batched inference
            def data_collator(features):
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                label_features = [{"input_ids": feature["labels"]} for feature in features]
                
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")["input_ids"]
                
                # Replace padding in labels with -100 (default ignore index)
                labels_batch[labels_batch == self.processor.tokenizer.pad_token_id] = -100
                
                batch["labels"] = labels_batch
                return batch

            data_loader = torch.utils.data.DataLoader(
                processed_datasets, 
                batch_size=self.batch_size,
                collate_fn=data_collator
            )

            # --- Inference Execution ---
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Transcribing Test Set"):
                    input_features = batch["input_features"].to(self.device)
                    
                    # Generate predictions (greedy decoding is usually standard for evaluation)
                    predicted_ids = self.model.generate(
                        input_features,
                        max_length=self.model.config.max_target_length,
                    )
                    
                    # Decode predictions and ground truth
                    transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                    
                    labels = batch["labels"]
                    labels[labels == -100] = self.processor.tokenizer.pad_token_id 
                    ground_truth = self.processor.batch_decode(labels, skip_special_tokens=True)

                    # Normalize and collect results for corpus-level metrics
                    for pred, ref in zip(transcriptions, ground_truth):
                        predictions.append(self._normalize_text(pred))
                        references.append(self._normalize_text(ref))

            self._calculate_and_log_metrics(predictions, references)

        except RuntimeError as e:
            logging.critical(f"RUNTIME ERROR during inference (Out of Memory - OOM): {e}. Try lowering the batch size.")
        except Exception as e:
            logging.critical(f"An unexpected error occurred during pipeline execution: {e}")
        finally:
            wandb.finish()


def main():
    """Main function to parse arguments and execute the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate an aggregated Whisper checkpoint and log metrics to W&B."
    )
    parser.add_argument(
        '--model_checkpoint', 
        type=str, 
        required=True, 
        help='Path to the aggregated model checkpoint file (e.g., aggregated_model.pt).'
    )
    parser.add_argument(
        '--run_name', 
        type=str, 
        required=True, 
        help='Name for the W&B run (e.g., "weighted_0.7_0.3").'
    )
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='i4ds/spc_r', 
        help='Hugging Face dataset name (default: i4ds/spc_r).'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8, 
        help='Inference batch size.'
    )
    
    args = parser.parse_args()
    
    # Determine device dynamically
    execution_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    manager = EvaluationManager(
        model_path=args.model_checkpoint,
        run_name=args.run_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        device=execution_device
    )
    manager.run_pipeline()


if __name__ == "__main__":
    main()