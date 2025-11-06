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
logging.info("--- 1. LOGGER INITIALIZED ---") # Sanity Check 1

class EvaluationManager:
    """
    Manages the full lifecycle of model evaluation: W&B login, data/model loading, 
    transcription, and corpus-level metric calculation (WER/BLEU).
    """

    def __init__(self, model_path: str, run_name: str, dataset_name: str, batch_size: int, device: str):
        logging.info("--- 2. STARTING EvaluationManager __init__ ---") # Sanity Check 2
        
        self.model_path = model_path
        self.run_name = run_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device
        self.processor = None
        self.model = None

        try:
            logging.info("--- 2.1. SANITY CHECK: Initializing WER Metric ---") # Sanity Check 2.1
            self.wer_metric = evaluate.load("wer")
            logging.info("--- 2.2. SANITY CHECK: Initializing BLEU Metric ---") # Sanity Check 2.2
            self.bleu_metric = evaluate.load("bleu")
        except Exception as e:
            logging.critical(f"CRITICAL INIT FAILURE: Could not load Hugging Face 'evaluate' metrics: {e}")
            raise # Re-raise to crash hard and expose the error

        logging.info("--- 2. FINISHED EvaluationManager __init__ ---") # Sanity Check 2
        # ... rest of the methods remain unchanged for now ...
        
    def _login_to_wandb(self):
        # ... (login logic) ...
        # [No changes needed in _login_to_wandb for pinpointing the silent crash]
        
        # Original logic remains here

        # ...
        if 'WANDB_API_KEY' not in os.environ:
            logging.warning("WANDB_API_KEY not found. Prompting user for key.")
            try:
                api_key = getpass("Please enter your Weights & Biases API Key: ")
            except Exception:
                api_key = input("Please enter your Weights & Biases API Key: ")
            
            if api_key:
                os.environ['WANDB_API_KEY'] = api_key
            else:
                logging.error("W&B API key not provided. Skipping logging.")
                return False

        try:
            wandb.init(project="Whisper-FL-Evaluation", name=self.run_name, job_type="evaluation")
            logging.info(f"Initialized W&B run: {self.run_name}")
            return True
        except Exception as e:
            logging.critical(f"Failed to initialize W&B: {e}")
            return False

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\säöüß]", "", text)  
        text = re.sub(r"\s\s+", " ", text).strip()
        return text

    def _load_and_prepare_data(self):
        logging.info("--- 3. STARTING Data Preparation ---") # Sanity Check 3
        try:
            raw_datasets = load_dataset(self.dataset_name, split="test")  
            raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))

            def preprocess_function(example):
                audio = example["audio"]
                example["input_features"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
                example["labels"] = self.processor.tokenizer(example["text"]).input_ids
                return example

            processed_datasets = raw_datasets.map(
                preprocess_function, 
                remove_columns=raw_datasets.column_names, 
                num_proc=os.cpu_count()
            )
            logging.info(f"--- 3. FINISHED Data Preparation. Samples: {len(processed_datasets)} ---") # Sanity Check 3
            return processed_datasets
            
        except Exception as e:
            logging.critical(f"CRITICAL DATASET FAILURE: Check dataset name, columns, and audio dependencies (ffmpeg, torchcodec): {e}")
            raise # Crash hard

    def _load_model_and_processor(self):
        logging.info("--- 4. STARTING Model and Processor Loading ---") # Sanity Check 4
        try:
            # 4.1 Load Processor
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
            # 4.2 Load Model Architecture
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(self.device)

            # 4.3 Load Checkpoint
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            logging.info("--- 4. FINISHED Model Loading ---") # Sanity Check 4
            return True

        except Exception as e:
            logging.critical(f"CRITICAL MODEL LOADING FAILURE: Could not load checkpoint or model architecture: {e}")
            raise # Crash hard

    def _calculate_and_log_metrics(self, predictions: List[str], references: List[str]):
        # ... (metric calculation logic) ...
        # [No changes needed here as the crash happens before inference]
        logging.info("Calculating corpus-level metrics...")
        
        wer_score = self.wer_metric.compute(predictions=predictions, references=references) * 100
        bleu_references = [[r] for r in references]
        bleu_results = self.bleu_metric.compute(predictions=predictions, references=bleu_references)
        
        logging.info(f"Corpus Word Error Rate (WER): {wer_score:.2f}%")

        if wandb.run:
            wandb.log({
                "corpus_wer": wer_score,
                "corpus_bleu": bleu_results["bleu"],
                "bleu_precision_weights": {f"{i+1}-gram": p for i, p in enumerate(bleu_results["precisions"])}, 
            })
            logging.info("Metrics successfully logged to W&B.")

    def run_pipeline(self):
        # 5. STARTING PIPELINE
        try:
            if not self._login_to_wandb():
                return
            
            # 5.1 Call the loaded methods
            if not self._load_model_and_processor():
                return

            processed_datasets = self._load_and_prepare_data()
            if processed_datasets is None:
                return
            
            # ... (inference loop and final log) ...
            # [The inference loop will crash on its own if necessary, we focus on setup]
            
            predictions = []
            references = []
            
            # Data collator definition remains here...
            def data_collator(features):
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                label_features = [{"input_ids": feature["labels"]} for feature in features]
                
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")["input_ids"]
                
                labels_batch[labels_batch == self.processor.tokenizer.pad_token_id] = -100
                batch["labels"] = labels_batch
                return batch

            data_loader = torch.utils.data.DataLoader(
                processed_datasets, 
                batch_size=self.batch_size,
                collate_fn=data_collator
            )

            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Transcribing Test Set"):
                    input_features = batch["input_features"].to(self.device)
                    predicted_ids = self.model.generate(input_features, max_length=self.model.config.max_target_length)
                    transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                    
                    labels = batch["labels"]
                    labels[labels == -100] = self.processor.tokenizer.pad_token_id
                    ground_truth = self.processor.batch_decode(labels, skip_special_tokens=True)

                    for pred, ref in zip(transcriptions, ground_truth):
                        predictions.append(self._normalize_text(pred))
                        references.append(self._normalize_text(ref))

            self._calculate_and_log_metrics(predictions, references)


        except RuntimeError as e:
            logging.critical(f"RUNTIME ERROR during pipeline: {e}")
        except Exception as e:
            logging.critical(f"UNEXPECTED ERROR in run_pipeline: {e}")
        finally:
            wandb.finish()


def main():
    """Main function to parse arguments and execute the evaluation pipeline."""
    # 6. ARGS PARSE CHECK
    logging.info("--- 6. STARTING Args and Main Execution ---")

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
    
    # 7. MANAGER INSTANTIATION CHECK
    logging.info("--- 7. INSTANTIATING EvaluationManager ---")

    manager = EvaluationManager(
        model_path=args.model_checkpoint,
        run_name=args.run_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        device=execution_device
    )
    
    # 8. PIPELINE EXECUTION CHECK
    logging.info("--- 8. EXECUTING run_pipeline ---")
    manager.run_pipeline()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Final safety net for errors before logging is fully set up
        print(f"FATAL ERROR IN MAIN EXECUTION: {e}")