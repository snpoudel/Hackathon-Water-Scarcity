# Hackathon on Water Scarcity 2025 - Baseline Model Repository

This repository provides a simple toolkit to train baseline models and generate submission files for the [Hackathon on Water Scarcity 2025](https://www.codabench.org/competitions/4335). The baseline models predict water discharge for the 52 stations of eval dataset. You are free to experiment with different modeling approaches or use distinct models per station, as long as your submission file adheres to the required format (see Codabench guidelines).

## Data

- **Download:**  
  Obtain the dataset from [Zenodo](https://zenodo.org/records/14536611).  
- **Setup:**  
  Unzip the dataset and place it in the root directory of the repository.

## Notebook Structure

1. **Preprocessing**
   - *01 - Data Preprocessing*
   - *02 - Feature Engineering*

2. **Training**
   - *03 - Modelisation*

3. **Submission**
   - *01 - Prediction Computation*

## Submission

After running the notebooks, create your submission file (`data/evaluation/predictions.zip`) and upload it to [Codabench](https://www.codabench.org/competitions/4335).

## Setup Local Environment

```shell
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip if needed
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Add the current directory to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PWD"
