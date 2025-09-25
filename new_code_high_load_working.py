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
from datasets import load_dataset, Dataset
import threading
import subprocess
import sys
import re

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
BASE_MODEL = "microsoft/DialoGPT-small"  # Better for conversational AI
TRAINED_MODEL_PATH = "./trained_mental_health_bot"

# --- Enhanced Mental Health Dataset ---
def create_comprehensive_dataset():
    """Create a more comprehensive mental health dataset"""
    mental_health_data = [
        # Basic mental health concepts
        {"prompt": "What is mental health?", "response": "Mental health refers to our emotional, psychological, and social well-being. It affects how we think, feel, and act, and helps determine how we handle stress, relate to others, and make choices."},
        {"prompt": "What is anxiety?", "response": "Anxiety is a natural response to stress characterized by feelings of worry, nervousness, or fear. While occasional anxiety is normal, persistent anxiety that interferes with daily life may indicate an anxiety disorder."},
        {"prompt": "What is depression?", "response": "Depression is a common mental health disorder characterized by persistent sadness, loss of interest in activities, and difficulty carrying out daily tasks for at least two weeks."},
        
        # Coping strategies
        {"prompt": "How can I reduce anxiety?", "response": "Deep breathing exercises, mindfulness meditation, regular physical activity, limiting caffeine, and talking to a trusted person can help manage anxiety symptoms."},
        {"prompt": "What helps with depression?", "response": "Professional therapy, medication if prescribed, regular exercise, maintaining social connections, and establishing a daily routine can help manage depression."},
        {"prompt": "How to manage stress?", "response": "Time management, relaxation techniques, regular breaks, physical activity, and maintaining work-life balance are effective stress management strategies."},
        
        # Support and resources
        {"prompt": "How can I help a friend with mental health issues?", "response": "Listen without judgment, offer support, encourage professional help, check in regularly, and educate yourself about their condition."},
        {"prompt": "When should I seek professional help?", "response": "Consider seeking help when symptoms persist for weeks, interfere with daily functioning, cause significant distress, or if you have thoughts of self-harm."},
        
        # Self-care
        {"prompt": "What is self-care?", "response": "Self-care involves taking active steps to maintain your physical and mental health through activities like exercise, proper nutrition, adequate sleep, and relaxation."},
        {"prompt": "How to practice mindfulness?", "response": "Focus on the present moment, observe your thoughts without judgment, practice deep breathing, and try guided meditation exercises."},
        
        # Crisis situations
        {"prompt": "I'm having a panic attack", "response": "Focus on your breathing - try the 4-7-8 technique. Name things you can see, hear, and feel around you. Remember that panic attacks are temporary and will pass."},
        {"prompt": "I feel overwhelmed", "response": "Break tasks into smaller steps, prioritize what's most important, take short breaks, and consider talking to someone about what you're experiencing."},
        
        # General support responses
        {"prompt": "I'm feeling sad", "response": "I'm sorry you're feeling this way. It's okay to feel sad sometimes. Would you like to talk more about what's bothering you?"},
        {"prompt": "I'm stressed about work", "response": "Work stress is common. Setting boundaries, taking regular breaks, and prioritizing tasks can help. Remember to make time for activities you enjoy outside of work."},
        {"prompt": "I can't sleep", "response": "Try establishing a relaxing bedtime routine, avoid screens before bed, keep your bedroom cool and dark, and consider relaxation techniques like deep breathing."},
        {"prompt": "I feel lonely", "response": "Loneliness can be difficult. Reaching out to friends or family, joining social groups, or volunteering can help build connections. Many people feel lonely at times."},
        {"prompt": "How to build confidence?", "response": "Set small achievable goals, practice self-compassion, challenge negative self-talk, and acknowledge your strengths and accomplishments."},
        
        # Professional boundaries
        {"prompt": "Can you cure me?", "response": "I'm an AI assistant here to provide information and support, but I'm not a medical professional. For treatment and diagnosis, please consult with a qualified healthcare provider."},
        {"prompt": "What medication should I take?", "response": "I cannot provide medical advice or prescribe medications. Please consult with a doctor or psychiatrist for appropriate treatment options."},
        {"prompt": "Are you a therapist?", "response": "No, I'm an AI assistant designed to provide mental health information and support. For therapy and clinical treatment, please seek help from licensed professionals."}
    ]
    
    # Add variations of common questions
    variations = []
    for item in mental_health_data:
        prompt = item["prompt"]
        response = item["response"]
        
        # Add different phrasings
        variations.append({"prompt": prompt.lower(), "response": response})
        variations.append({"prompt": prompt + "?", "response": response})
        variations.append({"prompt": "Tell me about " + prompt.lower().replace("what is ", "").replace("?", ""), "response": response})
        variations.append({"prompt": "I need help with " + prompt.lower().replace("what is ", "").replace("?", ""), "response": response})
    
    return mental_health_data + variations

def format_conversation_data(data):
    """Format data for conversational training"""
    formatted_data = []
    for item in data:
        # Create multiple conversation formats
        text1 = f"User: {item['prompt']}\nAssistant: {item['response']}"
        text2 = f"Human: {item['prompt']}\nAI: {item['response']}"
        text3 = f"Q: {item['prompt']}\nA: {item['response']}"
        
        formatted_data.extend([text1, text2, text3])
    
    return formatted_data

def setup_training_data():
    """Setup comprehensive training data"""
    try:
        print(f"{bcolors.OKBLUE}Creating comprehensive mental health dataset...{bcolors.ENDC}")
        data = create_comprehensive_dataset()
        formatted_texts = format_conversation_data(data)
        
        # Create DataFrame and save
        df = pd.DataFrame(formatted_texts, columns=['text'])
        csv_path = "comprehensive_mental_health_data.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"{bcolors.OKGREEN}Created dataset with {len(formatted_texts)} training examples{bcolors.ENDC}")
        return csv_path
        
    except Exception as e:
        print(f"{bcolors.FAIL}Dataset creation failed: {e}{bcolors.ENDC}")
        return None

# --- Improved Model Training ---
def train_model_async(data_file_path):
    global model, tokenizer, chatbot_pipeline, is_training
    try:
        is_training = True

        # Unload current model
        print(f"{bcolors.OKBLUE}Preparing for training...{bcolors.ENDC}")
        if chatbot_pipeline is not None:
            del chatbot_pipeline
            del model
            del tokenizer
            chatbot_pipeline = None
            model = None
            tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n{bcolors.HEADER}{bcolors.BOLD}--- Starting Model Fine-Tuning ---{bcolors.ENDC}")

        # Load tokenizer and model
        local_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        local_tokenizer.pad_token = local_tokenizer.eos_token
        
        local_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        
        print(f"{bcolors.OKBLUE}Loading dataset...{bcolors.ENDC}")
        dataset = load_dataset('csv', data_files=data_file_path)
        
        def tokenize_function(examples):
            # Use a larger max_length for better context
            return local_tokenizer(
                examples['text'], 
                truncation=True, 
                padding='max_length', 
                max_length=256,
                return_tensors="pt"
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=local_tokenizer, 
            mlm=False,
            pad_to_multiple_of=8
        )

        # Improved training arguments
        training_args = TrainingArguments(
            output_dir=TRAINED_MODEL_PATH,
            overwrite_output_dir=True,
            num_train_epochs=5,  # More epochs for better learning
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            learning_rate=5e-5,  # Better learning rate
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=local_model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            data_collator=data_collator,
        )

        print(f"{bcolors.OKBLUE}Training started...{bcolors.ENDC}")
        trainer.train()
        
        trainer.save_model(TRAINED_MODEL_PATH)
        local_tokenizer.save_pretrained(TRAINED_MODEL_PATH)

        print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}--- Training Complete! ---{bcolors.ENDC}")
        load_model()
        
    except Exception as e:
        print(f"{bcolors.FAIL}Training failed: {e}{bcolors.ENDC}")
        if chatbot_pipeline is None:
            load_model()
    finally:
        is_training = False
        print(f"{bcolors.WARNING}Training finished. You can chat with the bot now.{bcolors.ENDC}")

def load_model():
    global model, tokenizer, chatbot_pipeline
    model_path = TRAINED_MODEL_PATH if os.path.exists(TRAINED_MODEL_PATH) else BASE_MODEL
    print(f"{bcolors.OKBLUE}Loading model from '{model_path}'...{bcolors.ENDC}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
            
        device = 0 if torch.cuda.is_available() else -1
        chatbot_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device=device,
            torch_dtype=torch.float32
        )
        status = "fine-tuned" if os.path.exists(TRAINED_MODEL_PATH) else "base"
        print(f"{bcolors.OKGREEN}Successfully loaded {status} model.{bcolors.ENDC}")
    except Exception as e:
        print(f"{bcolors.FAIL}Error loading model: {e}{bcolors.ENDC}")
        chatbot_pipeline = None

def generate_response(user_input):
    """Generate response with better prompting and safety"""
    global chatbot_pipeline, tokenizer

    # Enhanced safety filter
    CRISIS_KEYWORDS = ["suicide", "kill myself", "kms", "wanna die", "end my life", "self harm", "hurting myself"]
    
    if any(keyword in user_input.lower() for keyword in CRISIS_KEYWORDS):
        return (
            "I'm very concerned about what you're sharing. Please know that your life has value and there is help available right now.\n\n"
            "🌐 Immediate Resources:\n"
            "• US/Canada: Call/text 988 (Suicide & Crisis Lifeline)\n"
            "• UK: Call 111 or 999 in emergency\n"
            "• International: Find local crisis lines at findahelpline.com\n\n"
            "Please reach out to these resources or a trusted person immediately. You don't have to face this alone."
        )

    if chatbot_pipeline is None:
        return "I'm still getting ready to help. Please try again in a moment."

    try:
        # Better prompt engineering
        prompt = f"""The following is a conversation with a mental health support assistant. The assistant is helpful, empathetic, and provides appropriate information while maintaining professional boundaries.

User: {user_input}
Assistant:"""
        
        result = chatbot_pipeline(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8,  # Slightly higher temperature for more varied responses
            top_p=0.9,       # Use top-p sampling for better quality
            top_k=50,
            repetition_penalty=1.2,  # Reduce repetition
            early_stopping=True
        )
        
        full_text = result[0]['generated_text']
        response = full_text.split("Assistant:")[-1].strip()
        
        # Clean up response
        response = re.split(r'[.!?]', response)[0] + '.'  # Take first complete sentence
        response = response.replace('"', '').replace('\n', ' ').strip()
        
        # Fallback responses if output is poor
        if len(response) < 10 or response.lower() in ['i', 'you', 'the', 'a']:
            return "I understand you're reaching out for support. Could you tell me more about what you're experiencing?"
            
        return response
        
    except Exception as e:
        print(f"{bcolors.FAIL}Error generating response: {e}{bcolors.ENDC}")
        return "I want to make sure I understand you correctly. Could you rephrase that?"

def start_training():
    global is_training
    if is_training:
        print(f"{bcolors.WARNING}Training is already in progress.{bcolors.ENDC}")
        return

    def train_thread_target():
        csv_path = setup_training_data()
        if csv_path:
            train_model_async(csv_path)
        else:
            print(f"{bcolors.FAIL}Failed to create training data.{bcolors.ENDC}")
            is_training = False
    
    threading.Thread(target=train_thread_target, daemon=True).start()

def print_help():
    print("\n" + "="*50)
    print(f"{bcolors.BOLD}Mental Health Support Chatbot{bcolors.ENDC}")
    print("="*50)
    print(f"{bcolors.OKGREEN}/train{bcolors.ENDC}    - Train the model with comprehensive mental health data")
    print(f"{bcolors.OKGREEN}/help{bcolors.ENDC}     - Show this help message")
    print(f"{bcolors.OKGREEN}/exit{bcolors.ENDC}     - Quit the chatbot")
    print("="*50)
    print(f"{bcolors.WARNING}Remember: I provide support but cannot replace professional help.{bcolors.ENDC}")
    print("="*50)

def main_cli():
    print_help()
    while True:
        try:
            if not is_training:
                user_input = input(f"\n{bcolors.OKGREEN}You: {bcolors.ENDC}").strip()
            else:
                user_input = input().strip()

            if not user_input:
                continue
                
            if user_input.lower() in ['/quit', '/exit']:
                print(f"{bcolors.OKBLUE}Take care of yourself. Goodbye!{bcolors.ENDC}")
                break
            elif user_input.lower() == '/help':
                print_help()
            elif user_input.lower() == '/train':
                start_training()
            else:
                response = generate_response(user_input)
                print(f"{bcolors.OKCYAN}{bcolors.BOLD}Support:{bcolors.ENDC} {response}")
                
        except KeyboardInterrupt:
            print(f"\n{bcolors.OKBLUE}Take care. Goodbye!{bcolors.ENDC}")
            break
        except Exception as e:
            print(f"{bcolors.FAIL}Error: {e}{bcolors.ENDC}")

if __name__ == "__main__":
    print("="*60)
    print(f"{bcolors.BOLD}Mental Health Support Assistant{bcolors.ENDC}")
    print(f"{bcolors.WARNING}{DISCLAIMER}{bcolors.ENDC}")
    print("="*60)
    
    load_model()
    main_cli()