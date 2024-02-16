# Settings
mvlm_dataset="docile_10k"
token_classif_dataset="docile"
full_pipeline_base_path="/domino/datasets/local/DeepInsuranceDocs/models/layoutlm_full_pipeline"

# 1. Proceed with Unsupervised MVLM on LayoutLM model
mvlm_config_path="/mnt/config/from_scratch_mvlm_$mvlm_dataset.json"

echo "FROM SCRATCH MVLM TRAINING"
mvlm_output=$(python /mnt/deepinsurancedocs/layoutlm/mvlm/train.py --config_path "$mvlm_config_path")|| { echo "Error during MVLM training"; exit 1; }

# 2. Extract the folder where the trained model has been saved
mvlm_model_folder=$(echo "$mvlm_output" | grep -oP "MVLM Model saved in: \K.*")
training_date=$(echo "$mvlm_output" | grep -oP "MVLM Training date: \K.*")
full_pipeline_folder="$full_pipeline_base_path/from_scratch_mvlm_${mvlm_dataset}_tc_${token_classif_dataset}/$training_date"
mkdir -p "$full_pipeline_folder/"

# 2 bis. Save the MVLM config in full_pipeline_folder
#        Save the MVLM Tensorboard graphs in full_pipeline_folder
cp -p "$mvlm_config_path" "$full_pipeline_folder/mvlm_config.json"|| { echo "Error during copying MVLM config file"; exit 1; }
cp -r "$mvlm_model_folder/mvlm_runs" "$full_pipeline_folder/mvlm_runs"
echo "MVLM Model saved in $mvlm_model_folder"
echo "MVLM config saved in $full_pipeline_folder"

# 3. Create a modified config file with the newly trained model and save it in full_pipeline_folder
token_classif_config_file="/mnt/config/token_classif_$token_classif_dataset.json"

jq --arg mvlm_model_folder "$mvlm_model_folder" '.pretrained_model = $mvlm_model_folder' "$token_classif_config_file" > "$full_pipeline_folder/token_classif_config.json"
token_classif_config_file="$full_pipeline_folder/token_classif_config.json"

# 4. Proceed with Token Classification with weights of MVLM trained model
echo "TC TRAINING WITH MVLM UPDATED WEIGHTS"

token_classif_output=$(python /mnt/deepinsurancedocs/layoutlm/token_classif/train.py --config_path "$token_classif_config_file")|| { echo "Error during TC training"; exit 1; }
token_classif_model_folder=$(echo "$token_classif_output" | grep -oP "TC Model saved in: \K.*")

echo "TC Model saved in $token_classif_model_folder"
echo "TC config saved in $full_pipeline_folder"

# 5. Save the paths of the two models in a txt file
cp -r "$token_classif_model_folder/tc_runs" "$full_pipeline_folder/tc_runs"
echo -e "$mvlm_model_folder\n$token_classif_model_folder" > "$full_pipeline_folder/models_path.txt"

# Now we have a folder with the information about which MVLM/TC datasets combination is used, with a subfolder named as the date of training. 
# In this folder, we have the config used for MVLM and TC training + a models_paths.txt file with the paths of the MVLM trained and TC trained models
