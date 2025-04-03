import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActivationHookWrapper:
    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.activations = {}
        self.hooks = []
        self._register_hooks()
    
    def _hook_fn(self, module, input, output, layer_name):
        self.activations[layer_name] = output.detach().cpu()
    
    def _register_hooks(self):
        """Registers forward hooks on each layer of interest."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):  # Example: Collect activations from Linear layers
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: self._hook_fn(m, inp, out, name)
                )
                self.hooks.append(hook)
    
    def generate_text(self, prompt, max_length=50):
        """Generates text and collects activations."""
        self.activations.clear()  # Clear previous activations
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text, self.activations
    
    def remove_hooks(self):
        """Removes all hooks to avoid memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Example usage
if __name__ == "__main__":
    wrapper = ActivationHookWrapper()
    text, activations = wrapper.generate_text("Hello world")
    print("Generated Text:", text)
    print("Collected Activations:", {k: v.shape for k, v in activations.items()})
    wrapper.remove_hooks()
