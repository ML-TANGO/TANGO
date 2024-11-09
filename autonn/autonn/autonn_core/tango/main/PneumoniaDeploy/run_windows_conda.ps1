$env:PYTHONUTF8 = 1
$conda_env_name = "ETRI_Pneumonia_Diagnostic_Model"
conda activate $conda_env_name
python app.py
