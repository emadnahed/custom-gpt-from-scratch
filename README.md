# Custom GPT from Scratch

This project implements a custom GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. The implementation includes the core transformer architecture, training pipeline, and text generation capabilities.

## 🚀 Features

- Pure PyTorch implementation of GPT architecture
- Training pipeline with configurable hyperparameters
- Text generation with temperature sampling
- Support for custom datasets
- Efficient training with mixed precision (FP16) support

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/emadnahed/custom-gpt-from-scratch.git
   cd custom-gpt-from-scratch
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ Project Structure

```
custom-gpt-from-scratch/
│
├── data/                 # Data loading and preprocessing
│   └── prepare.py        # Data preparation scripts
│
├── model/                # Model architecture
│   ├── __init__.py
│   └── transformer.py    # Core transformer implementation
│
├── config/               # Configuration files
│   └── train_default.py  # Default training configuration
│
├── train.py              # Training script
├── sample.py             # Text generation script
├── requirements.txt      # Project dependencies
└── README.md
```

## 🚦 Getting Started

### Training

To train the model with default configuration:

```bash
python train.py
```

For custom training configuration, create a new config file in the `config/` directory and specify it:

```bash
python train.py --config config/your_config.py
```

### Text Generation

To generate text using a trained model:

```bash
python sample.py --prompt "Your prompt here" --model_path path/to/checkpoint.pt
```

## 📊 Dependencies

- Python 3.11+
- PyTorch 2.2.2+
- torchvision
- torchaudio
- NumPy
- tqdm
- Hugging Face Datasets (for data loading)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.