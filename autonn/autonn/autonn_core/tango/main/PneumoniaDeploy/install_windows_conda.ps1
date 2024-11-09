# Define the name of the Conda environment
$env:PYTHONUTF8 = 1
$conda_env_name = "ETRI_Pneumonia_Diagnostic_Model"
Write-Output "The Conda environment name is: $conda_env_name"

conda update -n base conda -y
conda update --all -y
python -m pip install --upgrade pip

conda create -n $conda_env_name python=3.11 -y
conda activate $conda_env_name
pip install -r requirements.txt