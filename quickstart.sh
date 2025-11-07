#!/bin/bash
# Quick Start Script for GPT Training
# This script helps you set up and start training quickly

set -e  # Exit on error

echo "=================================================="
echo "GPT Training Quick Start"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo -e "${BLUE}Step 1: Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi
echo ""

# Step 2: Create virtual environment
echo -e "${BLUE}Step 2: Setting up virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping...${NC}"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Step 3: Activate virtual environment
echo -e "${BLUE}Step 3: Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo "Please manually activate: source venv/bin/activate"
fi
echo ""

# Step 4: Install dependencies
echo -e "${BLUE}Step 4: Installing dependencies...${NC}"
echo "This may take a few minutes..."
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 5: Check hardware
echo -e "${BLUE}Step 5: Detecting hardware...${NC}"
python check_hardware.py
echo ""

# Step 6: Prepare data
echo -e "${BLUE}Step 6: Preparing dataset...${NC}"
if [ -f "data/train.bin" ] && [ -f "data/val.bin" ]; then
    echo -e "${YELLOW}Dataset already prepared. Skipping...${NC}"
else
    echo "Preparing Shakespeare dataset..."
    cd data
    python prepare.py
    cd ..
    echo -e "${GREEN}✓ Dataset prepared${NC}"
fi
echo ""

# Step 7: Ready to train
echo "=================================================="
echo -e "${GREEN}Setup Complete! Ready to train.${NC}"
echo "=================================================="
echo ""
echo "To start training, run:"
echo "  python train.py"
echo ""
echo "Or for interactive hardware selection:"
echo "  python train.py --interactive"
echo ""
echo "For help, see:"
echo "  - GETTING_STARTED.md (comprehensive guide)"
echo "  - HARDWARE_FEATURE_SUMMARY.md (hardware features)"
echo ""
echo "Happy training! 🚀"
