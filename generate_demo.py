"""
Simple text generation demo script
"""

import os
import sys
import torch
from gpt_from_scratch.model import GPT

# Check if checkpoint exists
checkpoint_path = 'out/ckpt.pt'
if not os.path.exists(checkpoint_path):
    print(f"Error: Model checkpoint not found at '{checkpoint_path}'")
    print("Please train a model first using 'python gpt.py train'")
    sys.exit(1)

try:
    # Load checkpoint
    print("Loading trained model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Validate checkpoint structure
    required_keys = ['model_config', 'model', 'vocab']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing required key: {key}")
    
    model_config = checkpoint['model_config']
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to('cpu')

    # Load vocabulary
    vocab = checkpoint['vocab']
    if 'stoi' not in vocab or 'itos' not in vocab:
        raise KeyError("Vocabulary in checkpoint is missing required keys ('stoi' or 'itos')")
    
    stoi = vocab['stoi']
    itos = vocab['itos']

    print(f"✓ Model loaded! Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"✓ Vocabulary size: {len(itos)}")
    print("\n" + "="*70)

except (FileNotFoundError, KeyError, RuntimeError) as e:
    print(f"Error loading model: {e}")
    if isinstance(e, FileNotFoundError):
        print("  - The model checkpoint file was not found")
    elif isinstance(e, KeyError):
        print(f"  - The checkpoint is missing required data: {e}")
    elif 'UnicodeDecodeError' in str(e):
        print("  - The checkpoint file is corrupted or in an unexpected format")
    sys.exit(1)

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
