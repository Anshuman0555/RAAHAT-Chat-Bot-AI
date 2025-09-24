import os
import zipfile
import gc
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import load_dataset # <-- CHANGE: Import the modern dataset library
import threading
import subprocess
import sys

# --- 🚨 IMPORTANT ETHICAL DISCLAIMER ---
DISCLAIMER = """
This chatbot is for educational and informational purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of qualified mental health providers with any questions you may have regarding mental health conditions.
"""

# --- ANSI Colors for Terminal Output ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --- Global Variables ---
model = None
tokenizer = None
chatbot_pipeline = None
is_training = False
BASE_MODEL = "distilgpt2"
TRAINED_MODEL_PATH = "./trained_mental_health_bot"

# --- Dataset Handling (Updated) ---
def setup_kaggle_dataset(dataset_path="nroshd/mental-health-fasta-chatbot"):
    try:
        zip_path = "mental-health-fasta-chatbot.zip"
        data_dir = "mental_health_data"
        csv_file = os.path.join(data_dir, "Mental-Health-FAQ.csv")
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_path], check=True, capture_output=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        return csv_file
    except Exception as e:
        print(f"{bcolors.FAIL}Kaggle dataset setup failed: {e}{bcolors.ENDC}")
        return None

# <-- MAJOR CHANGE: Data formatting is now simpler for the new library
def format_data_for_training(csv_file_path, output_csv_path="formatted_data.csv"):
    try:
        print(f"{bcolors.OKBLUE}Formatting data for training...{bcolors.ENDC}")
        df = pd.read_csv(csv_file_path)
        df.rename(columns={'Questions': 'prompt', 'Answers': 'response'}, inplace=True)
        
        # Combine prompt and response into a single text column
        df['text'] = df.apply(lambda row: f"User: {row['prompt']}\nAssistant: {row['response']}", axis=1)
        
        # Save only the text column to a new CSV
        df[['text']].to_csv(output_csv_path, index=False)
        return output_csv_path
    except Exception as e:
        print(f"{bcolors.FAIL}Data formatting failed: {e}{bcolors.ENDC}")
        return None

# --- Model Training (Updated) ---
def train_model_async(data_file_path):
    global model, tokenizer, chatbot_pipeline, is_training
    try:
        is_training = True

        # <-- FIX: Unload the global model to release file locks before training
        print(f"{bcolors.OKBLUE}Unloading the current model to prepare for training...{bcolors.ENDC}")
        if chatbot_pipeline is not None:
            # Clear the pipeline and models from memory
            del chatbot_pipeline
            del model
            del tokenizer
            chatbot_pipeline = None
            model = None
            tokenizer = None
            # Ask the garbage collector to clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n{bcolors.HEADER}{bcolors.BOLD}--- Starting Model Fine-Tuning ---{bcolors.ENDC}")

        # Load a fresh base model for the fine-tuning process
        local_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        local_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
        )
        
        # ... the rest of the function remains the same ...

        if local_tokenizer.pad_token is None:
            local_tokenizer.pad_token = local_tokenizer.eos_token
        
        print(f"{bcolors.OKBLUE}Loading and processing dataset...{bcolors.ENDC}")
        dataset = load_dataset('csv', data_files=data_file_path)

        def tokenize_function(examples):
            return local_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=local_tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=TRAINED_MODEL_PATH,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=local_model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            data_collator=data_collator,
        )

        print(f"{bcolors.OKBLUE}Fine-tuning is now in progress...{bcolors.ENDC}")
        trainer.train()
        
        trainer.save_model(TRAINED_MODEL_PATH)
        local_tokenizer.save_pretrained(TRAINED_MODEL_PATH)

        print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}--- Fine-Tuning Complete! ---{bcolors.ENDC}")
        # The script will now load the newly saved model
        load_model()
        
    except Exception as e:
        print(f"{bcolors.FAIL}Training failed: {e}. The chatbot will continue using the previous model.{bcolors.ENDC}")
        # If training failed, try to reload the original model to keep the bot running
        if chatbot_pipeline is None:
            load_model()
    finally:
        is_training = False
        print(f"{bcolors.WARNING}You can now chat with the bot again.{bcolors.ENDC}\nYou: {bcolors.ENDC}", end="")
        sys.stdout.flush()# --- Model Loading and Inference ---
def load_model():
    global model, tokenizer, chatbot_pipeline
    model_path = TRAINED_MODEL_PATH if os.path.exists(TRAINED_MODEL_PATH) else BASE_MODEL
    print(f"{bcolors.OKBLUE}Loading model from '{model_path}'...{bcolors.ENDC}")
    try:
        # <-- FIX: Added torch_dtype here as well for consistency
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = 0 if torch.cuda.is_available() else -1
        chatbot_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        status = "fine-tuned" if os.path.exists(TRAINED_MODEL_PATH) else "base"
        print(f"{bcolors.OKGREEN}Successfully loaded '{model_path}' ({status} model).{bcolors.ENDC}")
    except Exception as e:
        print(f"{bcolors.FAIL}Error loading model: {e}{bcolors.ENDC}")
        chatbot_pipeline = None

def generate_response(user_input):
    """Generate a response using the loaded pipeline."""
    global chatbot_pipeline, tokenizer

    # <-- CRITICAL SAFETY FILTER ---
    # Expanded the list of keywords for better crisis detection.
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "kms", "wanna die", "end my life", 
        "end it all", "don't want to live", "hang myself", "slit my wrists"
    ]
    
    # Check if any keyword is in the user's message (in lowercase).
    if any(keyword in user_input.lower() for keyword in CRISIS_KEYWORDS):
        # In India, you can contact iCALL (091529 87821) or Vandrevala Foundation (99996 66555).
        return (
            "It sounds like you are going through a very difficult time. Please know that help is available and you don't have to go through this alone. "
            "You can connect with people who can support you by contacting iCALL at 091529 87821 or the Vandrevala Foundation at 99996 66555. "
            "Please reach out, they are there to help."
        )
    # --- END OF SAFETY FILTER -->

    if chatbot_pipeline is None:
        return "Chatbot is not available."
    try:
        prompt = f"User: {user_input}\nAssistant:"
        
        result = chatbot_pipeline(
            prompt,
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )
        
        full_text = result[0]['generated_text']
        response = full_text.split("Assistant:")[-1].strip()
        
        return response if response else "I'm here to listen. Could you tell me more?"
    except Exception as e:
        print(f"\n{bcolors.FAIL}Error generating response: {e}{bcolors.ENDC}")
        return "I'm having trouble forming a response right now."
# --- CLI Handling ---
def start_training(use_kaggle):
    global is_training
    if is_training:
        print(f"{bcolors.WARNING}Training is already in progress.{bcolors.ENDC}")
        return

    def train_thread_target():
        if use_kaggle:
            csv_path = setup_kaggle_dataset()
        else:
            print(f"{bcolors.OKBLUE}Using sample dataset for fine-tuning.{bcolors.ENDC}")
            sample_data = {
                'Questions': ["What is anxiety?", "How can I deal with stress?", "What are signs of depression?", "Is it normal to feel sad sometimes?", "What is cognitive behavioral therapy?", "How can I improve my sleep?", "Why is self-care important?", "What if a friend is struggling?"],
                'Answers': ["Anxiety is a feeling of unease, such as worry or fear. It's a natural response to stress.", "Techniques include exercise, mindfulness, and talking to someone you trust.", "Signs include persistent sadness, loss of interest, and changes in sleep or appetite.", "Yes, feeling sad is a normal emotion. It's a concern when it is persistent and impacts your daily life.", "CBT is a therapy that helps you manage problems by changing the way you think and behave.", "Maintain a consistent schedule, create a relaxing routine, and avoid caffeine before bed.", "Self-care helps manage stress and increase energy. Hobbies and relaxation are vital.", "Encourage them to talk, listen without judgment, and suggest they speak with a professional."]
            }
            df = pd.DataFrame(sample_data)
            csv_path = "sample_data.csv"
            df.to_csv(csv_path, index=False)
        
        if csv_path:
            formatted_path = format_data_for_training(csv_path)
            if formatted_path:
                train_model_async(formatted_path)
    
    threading.Thread(target=train_thread_target).start()

def print_help():
    print("\n" + "="*50)
    print(f"{bcolors.BOLD}Chatbot Commands:{bcolors.ENDC}")
    print(f"  {bcolors.OKGREEN}/train kaggle{bcolors.ENDC} - Fine-tune using the full Kaggle dataset.")
    print(f"  {bcolors.OKGREEN}/train sample{bcolors.ENDC} - Fine-tune using the built-in sample dataset.")
    print(f"  {bcolors.OKGREEN}/help{bcolors.ENDC}        - Show this help message.")
    print(f"  {bcolors.OKGREEN}/exit{bcolors.ENDC}        - Quit the chatbot.")
    print("="*50 + "\n")

def main_cli():
    print_help()
    while True:
        if not is_training:
            user_input = input(f"{bcolors.OKGREEN}You: {bcolors.ENDC}")
        else:
            user_input = input()

        clean_input = user_input.strip().replace('\\', '/')
        
        if clean_input.lower() in ['/quit', '/exit']:
            print(f"{bcolors.OKBLUE}Goodbye!{bcolors.ENDC}")
            break
        elif clean_input.lower() == '/help':
            print_help()
        elif clean_input.lower() == '/train kaggle':
            start_training(use_kaggle=True)
        elif clean_input.lower() == '/train sample':
            start_training(use_kaggle=False)
        else:
            response = generate_response(user_input)
            print(f"{bcolors.OKCYAN}{bcolors.BOLD}Bot:{bcolors.ENDC} {response}")

if __name__ == "__main__":
    print("="*50)
    print(f"{bcolors.BOLD}Starting Mental Health Support Chatbot...{bcolors.ENDC}")
    print(f"{bcolors.WARNING}{DISCLAIMER}{bcolors.ENDC}")
    print("="*50)
    
    load_model()
    main_cli()