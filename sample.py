"""
Text generation script for trained GPT model

Usage:
    python sample.py --prompt "Once upon a time" --max_tokens 100
    python sample.py --checkpoint out/ckpt.pt --temperature 0.8
"""

import os
import argparse
import pickle
import torch

from gpt_from_scratch.model import GPT, GPTConfig


def sample(
    checkpoint_path='out/ckpt.pt',
    prompt='',
    max_new_tokens=100,
    temperature=0.8,
    top_k=200,
    top_p=0.9,
    seed=1337,
    device='auto'
):
    """
    Generate text from a trained model

    Args:
        checkpoint_path: Path to model checkpoint
        prompt: Starting text prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        seed: Random seed
        device: Device to use (auto, cuda, cpu, mps)
    """

    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get vocab
    vocab = checkpoint['vocab']
    stoi = vocab['stoi']
    itos = vocab['itos']
    vocab_size = vocab['vocab_size']

    # Create model
    model_config = checkpoint['model_config']
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Context length: {model_config.block_size}")

    # Encode prompt
    if prompt:
        print(f"\nPrompt: {repr(prompt)}")
        # Check if all characters in prompt are in vocabulary
        for ch in prompt:
            if ch not in stoi:
                print(f"Warning: Character '{ch}' not in vocabulary, replacing with space")
                prompt = prompt.replace(ch, ' ')

        start_ids = [stoi[ch] for ch in prompt]
    else:
        # Start with newline character
        start_ids = [stoi.get('\n', 0)]

    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate
    print(f"\nGenerating {max_new_tokens} tokens...")
    print("=" * 80)

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None
        )

    # Decode
    generated_ids = y[0].tolist()
    generated_text = ''.join([itos[i] for i in generated_ids])

    print(generated_text)
    print("=" * 80)
    print(f"Generated {len(generated_ids)} tokens")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text from a trained GPT model')

    # Model
    parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt',
                      help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cuda', 'cpu', 'mps'],
                      help='Device to use for generation')

    # Generation
    parser.add_argument('--prompt', type=str, default='',
                      help='Starting text prompt')
    parser.add_argument('--max_tokens', type=int, default=100,
                      help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                      help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=200,
                      help='Top-k filtering (0 = disabled)')
    parser.add_argument('--top_p', type=float, default=0.9,
                      help='Nucleus sampling threshold (1.0 = disabled)')
    parser.add_argument('--seed', type=int, default=1337,
                      help='Random seed')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode')

    args = parser.parse_args()

    if args.interactive:
        # Interactive mode - keep generating until user exits
        print("Interactive mode - Enter 'quit' or 'exit' to stop")
        print("=" * 80)

        while True:
            prompt = input("\nEnter prompt (or 'quit' to exit): ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not prompt:
                prompt = '\n'

            try:
                sample(
                    checkpoint_path=args.checkpoint,
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    seed=args.seed,
                    device=args.device
                )
            except Exception as e:
                print(f"Error: {e}")
                continue
    else:
        # Single generation
        sample(
            checkpoint_path=args.checkpoint,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            device=args.device
        )


if __name__ == '__main__':
    main()
