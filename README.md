# DR Project

## 🚀 Quick Start

### Installation
```bash
conda create -n icrpm_dr python=3.7 -y
conda activate icrpm_dr

# Install dependencies (mujoco210 required)
pip install -e .
# Install gym-ergojr
pip install git+https://github.com/fgolemo/gym-ergojr.git
# Install openai baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip install tensorflow-gpu==1.14
pip install -e .
```

## Usage
To run the ICRPM training process, execute the following command:
```
python experiments/domainrand/run_all_and_plot.py
```