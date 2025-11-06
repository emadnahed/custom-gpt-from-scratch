"""
Simple text generation demo script
"""

import torch
import pickle
from model.transformer import GPT

# Load checkpoint
print("Loading trained model...")
checkpoint = torch.load('out/ckpt.pt', map_location='cpu')
model_config = checkpoint['model_config']
model = GPT(model_config)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to('cpu')

# Load vocabulary
vocab = checkpoint['vocab']
stoi = vocab['stoi']
itos = vocab['itos']

print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"Vocabulary size: {len(itos)}")
print("\n" + "="*70)

# Generate text with different prompts
prompts = [
    "",  # Random start
    "ROMEO:",
    "JULIET:",
    "The",
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}' (or random if empty)")
    print("-" * 70)

    # Encode prompt
    if prompt:
        context = torch.tensor([stoi.get(c, 0) for c in prompt], dtype=torch.long, device='cpu').unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device='cpu')

    # Generate
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=150,
            temperature=0.8,
            top_k=200
        )

    # Decode and print
    generated_text = ''.join([itos[int(i)] for i in generated[0]])
    print(generated_text)
    print()

print("="*70)
print("Generation complete!")
