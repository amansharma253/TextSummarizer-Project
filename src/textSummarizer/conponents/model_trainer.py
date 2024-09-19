from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from src.textSummarizer.entity import ModelTrainerConfig
import torch
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use the T5 tokenizer instead of the Pegasus tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        # Load T5 model instead of Pegasus
        model_t5 = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        # Use the T5 model in the data collator
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_t5)
        
        # Loading the dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Define training arguments (keeping your settings as they are)
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
        )

        # Initialize the Trainer with the T5 model
        trainer = Trainer(model=model_t5, args=trainer_args,
                          tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                          train_dataset=dataset_samsum_pt["train"], 
                          eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        # Save the T5 model
        model_t5.save_pretrained(os.path.join(self.config.root_dir, "t5-samsum-model"))

        # Save the tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
