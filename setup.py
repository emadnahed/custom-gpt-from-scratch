from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gpt-from-scratch",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A from-scratch implementation of GPT model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/custom-gpt-from-scratch",
    packages=find_packages(include=['gpt_from_scratch', 'gpt_from_scratch.*']),
    python_requires=">=3.8",
    install_requires=[
        'torch>=1.12.0',
        'numpy>=1.21.0',
        'tqdm>=4.65.0',
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'tokenizers>=0.13.3',
        'sentencepiece>=0.1.99',
        'accelerate>=0.20.0',
        'bitsandbytes>=0.39.0',
        'wandb>=0.15.0',
    ],
    entry_points={
        'console_scripts': [
            'gpt=gpt_from_scratch.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
