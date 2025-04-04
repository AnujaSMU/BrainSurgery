import torch
import os
import json
from tqdm import tqdm
from transformers import set_seed
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")


class ActivationCollector:
    def __init__(self, model_name=model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

        self.activations = []

    def collect_activations(self, layer_name):
        """
        Hook function to collect activations for a specific layer
        """
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations.append(output.detach().cpu())
        return hook_fn

    def register_hooks(self):
        """
        Register hooks on transformer layers of Gemma
        """
        for name, module in self.model.named_modules():
            if 'model.layers' in name:  # Gemma uses 'model.layers.N'
                module.register_forward_hook(self.collect_activations(name))

    def reset_activations(self):
        self.activations = []

    def generate_text(self, input_text, max_new_tokens=50):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        generated_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


        # Collect activations using Gemma
        self.reset_activations()
        input_ids = self.tokenizer(generated_text, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        self.register_hooks()
        with torch.no_grad():
            self.model(input_ids)

        return generated_text, self.activations


class ActivationDatasetBuilder:
    def __init__(self, collector, prompts, save_dir="activation_dataset"):
        self.collector = collector
        self.prompts = prompts
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.dataset = []

    def generate_dataset(self, max_new_tokens=40):
        for i, prompt in enumerate(tqdm(self.prompts, desc="Generating activations")):
            try:
                generated_text, activations = self.collector.generate_text(prompt, max_new_tokens=max_new_tokens)
                tokens = self.collector.tokenizer(generated_text, return_tensors="pt")["input_ids"][0]
                token_strings = [self.collector.tokenizer.decode([t]) for t in tokens]
                activation_tensor = activations[-1]  # Use last collected layer

                self.dataset.append({
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "tokens": token_strings,
                    "activations": activation_tensor.squeeze(0)  # Remove batch dim
                })
            except Exception as e:
                print(f"Error on prompt {i}: {e}")

    def save_dataset(self):
        meta_data = []

        for i, entry in enumerate(self.dataset):
            act_path = os.path.join(self.save_dir, f"activations_{i}.pt")
            torch.save(entry["activations"], act_path)

            meta_data.append({
                "index": i,
                "prompt": entry["prompt"],
                "generated_text": entry["generated_text"],
                "tokens": entry["tokens"],
                "activations_path": act_path
            })

        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(meta_data, f, indent=2)

        print(f"Saved {len(self.dataset)} examples to {self.save_dir}")


prompts = [
    "Once upon a time in a distant land", #To explore story structure, named entities, and sequence continuity
    "What is the capital of France?", #To probe knowledge recall and factual associations
    "Write a Python function to reverse a string.", #To test code generation and logical reasoning
    "Alice: Hi Bob! How are you?", #To analyze conversational context and response generation
    "The theory of relativity explains...", #To evaluate scientific reasoning and coherence
    "Tell me a joke that would make even a robot laugh.", #To assess humor generation and creativity
    "If we were both code, I'd say you're the syntax to my logic."  #Flirty and metaphorical language       
]


set_seed(42)

# Run pipeline
collector = ActivationCollector(model_name=model_name)
builder = ActivationDatasetBuilder(collector, prompts)
builder.generate_dataset()
builder.save_dataset()
