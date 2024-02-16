# Settings
mvlm_dataset="docile_5k"
token_classif_dataset="docile"
full_pipeline_base_path="/domino/datasets/local/DeepInsuranceDocs/models/layoutlm_full_pipeline"

# 1. Proceed with Unsupervised MVLM on LayoutLM model
mvlm_config_path="/mnt/config/mvlm_$mvlm_dataset.json"

python /mnt/deepinsurancedocs/layoutlm/mvlm/train.py --config_path "$mvlm_config_path"