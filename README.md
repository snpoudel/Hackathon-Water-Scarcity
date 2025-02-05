The following repo give you a basic kit to train a baseline model and make a submission for the *Hackathon on Water Scarcity 2025* : https://www.codabench.org/competitions/4335 . We created several baseline model that predict water discharged on 52 stations. 
The evaluation dataset has a weekly frequency and we predict at 4 weeks horizon on 6 years period. Among those 52 stations :


* 14 doesn't have history in the training dataset
* 39 have history in the training dataset

You are free to have a different modelisation approach using different models for different water station as long as the submission file is in the right format (see codabench).

Setup local environment
Create virtual env python3 -m venv .venv

Activate virtual env source .venv/bin/activate

Upgrade pip if required pip install --upgrade pip

Install requirements pip install -r requirements.txt

Export working directory in python path export PYTHONPATH="PYTHONPATH:
PYTHONPATH:PWD" Data Zenodo For this project, we use data that we extracted from multiple open sources accesses : - ERA5 : Precipitations & Temperatures - etc.

We collected all the raw an pre-processed data that were important for the creation of the baseline model and made it available for you.

The structure of the notebook is the following :
01 - Preprocessing :
             01 - Data Preprocessing
             02 - Feature engineering
02 - Training :
              03 - Modelisation
03 - Submission :
              01 - Prediction Computation


## Setup local environment

``` shell
# Create virtual env
python3 -m venv .venv

# Activate virtual env
source .venv/bin/activate

# Upgrade pip if required
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Export working directory in python path
export PYTHONPATH="$PYTHONPATH:$PWD"
```
