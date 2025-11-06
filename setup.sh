#!/bin/bash
# =============================================================================
# GPT Training - Complete Automated Setup
# This script will:
# 1. Check/install Python
# 2. Set up virtual environment
# 3. Install dependencies
# 4. Prepare dataset
# 5. Get you ready to train!
# =============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}"
echo "======================================================================="
echo "          GPT Training - Automated Setup Script                       "
echo "======================================================================="
echo -e "${NC}"

# =============================================================================
# Step 1: Check/Install Python
# =============================================================================

echo -e "${BOLD}Step 1: Checking Python...${NC}"

# Check for Python 3.11 first (recommended)
if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found (recommended version!)${NC}"
    PYTHON_CMD="python3.11"
# Check if Python 3 is installed
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found (perfect!)${NC}"
        PYTHON_CMD="python3"
    elif [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        echo -e "${YELLOW}⚠ Python $PYTHON_VERSION found (works, but 3.11 recommended)${NC}"
        PYTHON_CMD="python3"
    else
        echo -e "${YELLOW}⚠ Python $PYTHON_VERSION is too old (need 3.8+, recommend 3.11)${NC}"
        NEED_PYTHON=1
    fi
else
    echo -e "${YELLOW}⚠ Python 3 not found${NC}"
    NEED_PYTHON=1
fi

# Install Python if needed
if [ ! -z "$NEED_PYTHON" ]; then
    echo -e "${BOLD}Installing Python...${NC}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS"
        if command -v brew &> /dev/null; then
            echo "Using Homebrew to install Python..."
            brew install python@3.11
            PYTHON_CMD="python3"
        else
            echo -e "${YELLOW}Homebrew not found. Installing Homebrew first...${NC}"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            brew install python@3.11
            PYTHON_CMD="python3"
        fi

    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Detected Linux"
        if command -v apt-get &> /dev/null; then
            # Debian/Ubuntu
            echo "Using apt to install Python..."
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3-pip
            PYTHON_CMD="python3.11"
        elif command -v yum &> /dev/null; then
            # RedHat/CentOS
            echo "Using yum to install Python..."
            sudo yum install -y python3.11
            PYTHON_CMD="python3.11"
        else
            echo -e "${RED}✗ Cannot auto-install Python on this Linux distribution${NC}"
            echo "Please install Python 3.8+ manually from https://python.org"
            exit 1
        fi

    else
        echo -e "${RED}✗ Cannot auto-install Python on this OS${NC}"
        echo "Please install Python 3.8+ manually:"
        echo "  - macOS: Download from https://python.org or use 'brew install python3'"
        echo "  - Windows: Download from https://python.org"
        echo "  - Linux: Use your package manager (apt, yum, etc.)"
        exit 1
    fi

    # Verify installation
    if command -v $PYTHON_CMD &> /dev/null; then
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
        echo -e "${GREEN}✓ Python $PYTHON_VERSION installed successfully${NC}"
    else
        echo -e "${RED}✗ Python installation failed${NC}"
        exit 1
    fi
fi

echo ""

# =============================================================================
# Step 2: Create Virtual Environment
# =============================================================================

echo -e "${BOLD}Step 2: Setting up virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
else
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

echo ""

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================

echo -e "${BOLD}Step 3: Installing dependencies...${NC}"
echo "This may take a few minutes (downloading PyTorch, etc.)..."

pip install --upgrade pip --quiet
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# =============================================================================
# Step 4: Detect Hardware
# =============================================================================

echo -e "${BOLD}Step 4: Detecting hardware...${NC}"
python check_hardware.py
echo ""

# =============================================================================
# Step 5: Prepare Dataset
# =============================================================================

echo -e "${BOLD}Step 5: Preparing dataset...${NC}"

if [ -f "data/train.bin" ] && [ -f "data/val.bin" ]; then
    echo -e "${YELLOW}Dataset already prepared${NC}"
else
    echo "Preparing Shakespeare dataset..."
    cd data
    python prepare.py
    cd ..
    echo -e "${GREEN}✓ Dataset prepared${NC}"
fi

echo ""

# =============================================================================
# All Done!
# =============================================================================

echo -e "${BOLD}${GREEN}"
echo "======================================================================="
echo "                    ✓ SETUP COMPLETE!                                  "
echo "======================================================================="
echo -e "${NC}"

echo ""
echo -e "${BOLD}Quick Start Commands:${NC}"
echo ""
echo "  ${BLUE}python gpt.py info${NC}        # Check your setup"
echo "  ${BLUE}python gpt.py train${NC}       # Start training (interactive)"
echo "  ${BLUE}python gpt.py generate${NC}    # Generate text"
echo "  ${BLUE}python gpt.py hardware${NC}    # Check hardware options"
echo ""
echo -e "${BOLD}Want to train right now?${NC}"
read -p "Start interactive training? (y/n) [y]: " START_TRAIN

if [ -z "$START_TRAIN" ] || [ "$START_TRAIN" == "y" ] || [ "$START_TRAIN" == "Y" ]; then
    python gpt.py train
else
    echo ""
    echo "No problem! When you're ready, run:"
    echo "  ${BLUE}python gpt.py train${NC}"
    echo ""
fi

echo ""
echo -e "${BOLD}📚 Documentation:${NC}"
echo "  - QUICK_REFERENCE.md    - Command cheat sheet"
echo "  - GETTING_STARTED.md    - Beginner's guide"
echo "  - README.md             - Full documentation"
echo ""
echo "Happy training! 🚀"
