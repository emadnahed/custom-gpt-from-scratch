"""
Interactive Text Generation

Easy interface for generating text from your trained model
"""

import os
import torch
import pickle
from model.transformer import GPT


def print_header(text):
    """Print a fancy header"""
    print(f"\n{'='*70}")
    print(f"{text.center(70)}")
    print(f"{'='*70}\n")


def interactive_generate():
    """Interactive generation interface"""
    print_header("GPT TEXT GENERATION")

    # Check for checkpoint
    if not os.path.exists('out/ckpt.pt'):
        print("✗ No trained model found!")
        print("Train a model first with: python gpt.py train")
        return

    # Load model
    print("Loading model...")
    try:
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

        print(f"✓ Model loaded! Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"✓ Vocabulary size: {len(itos)}")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Generation loop
    while True:
        print("\n" + "-"*70)
        print("TEXT GENERATION OPTIONS")
        print("-"*70)

        # Get prompt
        print("\nEnter your prompt (or press Enter for random start):")
        prompt = input("> ").strip()

        # Get parameters
        print("\nGeneration parameters:")
        try:
            max_tokens = int(input("Max tokens to generate [200]: ").strip() or "200")
            temperature = float(input("Temperature (0.1-2.0) [0.8]: ").strip() or "0.8")
            top_k = int(input("Top-k filtering [200]: ").strip() or "200")
        except ValueError:
            print("Invalid input, using defaults")
            max_tokens = 200
            temperature = 0.8
            top_k = 200

        # Encode prompt
        if prompt:
            # The .get(c, 0) handles characters not in the vocabulary
            context = torch.tensor(
                [stoi.get(c, 0) for c in prompt],
                dtype=torch.long,
                device='cpu'
            ).unsqueeze(0)
        else:
            # Random start
            context = torch.zeros((1, 1), dtype=torch.long, device='cpu')

        # Generate
        print("\nGenerating...\n")
        print("="*70)

        try:
            with torch.no_grad():
                generated = model.generate(
                    context,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )

            # Decode
            generated_text = ''.join([itos[int(i)] for i in generated[0]])
            print(generated_text)

        except Exception as e:
            print(f"✗ Generation failed: {e}")

        print("="*70)

        # Continue?
        again = input("\nGenerate more? (y/n) [y]: ").strip().lower()
        if again == 'n':
            break

    print("\n✓ Generation complete!")


def batch_generate():
    """Generate multiple samples at once"""
    print_header("BATCH TEXT GENERATION")

    if not os.path.exists('out/ckpt.pt'):
        print("✗ No trained model found!")
        return

    # Load model
    print("Loading model...")
    checkpoint = torch.load('out/ckpt.pt', map_location='cpu')
    model_config = checkpoint['model_config']
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to('cpu')

    vocab = checkpoint['vocab']
    stoi = vocab['stoi']
    itos = vocab['itos']

    print("✓ Model loaded!")

    # Get prompts
    print("\nEnter prompts (one per line, empty line to finish):")
    prompts = []
    while True:
        prompt = input("> ").strip()
        if not prompt:
            break
        prompts.append(prompt)

    if not prompts:
        print("No prompts entered")
        return

    # Generation parameters
    max_tokens = int(input("\nMax tokens per sample [100]: ").strip() or "100")
    temperature = float(input("Temperature [0.8]: ").strip() or "0.8")

    # Generate for each prompt
    print("\n" + "="*70)
    for i, prompt in enumerate(prompts, 1):
        print(f"\nSample {i}/{len(prompts)}")
        print(f"Prompt: '{prompt}'")
        print("-"*70)

        # Encode
        context = torch.tensor(
            [stoi.get(c, 0) for c in prompt],
            dtype=torch.long,
            device='cpu'
        ).unsqueeze(0)

        # Generate
        with torch.no_grad():
            generated = model.generate(
                context,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=200
            )

        # Decode and print
        text = ''.join([itos[int(i)] for i in generated[0]])
        print(text)
        print()

    print("="*70)
    print(f"✓ Generated {len(prompts)} samples!")


if __name__ == '__main__':
    interactive_generate()
