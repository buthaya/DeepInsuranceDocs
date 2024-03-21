## Everything needed to run an experiment on its own aws instance

# Parse keyword arguments into environment variables
# derived from https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            MVLM_DATASET)                    MVLM_DATASET=${VALUE} ;;
            TOKEN_CLASSIF_DATASET)           TOKEN_CLASSIF_DATASET=${VALUE} ;;
            SUBSET_INDEX_PATH)               SUBSET_INDEX_PATH=${VALUE} ;;
            HOME_DIR)                        HOME_DIR=${VALUE} ;;

            MODEL_NAME)                      MODEL_NAME=${VALUE} ;;
            PRETRAINED_MODEL)                PRETRAINED_MODEL=${VALUE} ;;
            PREPROCESS_NORMALIZE_TXT)        PREPROCESS_NORMALIZE_TXT=${VALUE} ;;
            PREPROCESS_TAG_SCHEME)           PREPROCESS_TAG_SCHEME=${VALUE} ;;

            EPOCH_NUM)                       EPOCH_NUM=${VALUE} ;;
            BATCH_SIZE)                      BATCH_SIZE=${VALUE} ;;
            LEARNING_RATE)                   LEARNING_RATE=${VALUE} ;;
            GRADIENT_ACCUMULATION_STEPS)     GRADIENT_ACCUMULATION_STEPS=${VALUE} ;;
            MVLM_CONFIG_PATH)               MVLM_CONFIG_PATH=${VALUE} ;;
            *)   
    esac    
done

# Navigate to dir
cd $HOME_DIR/DeepInsuranceDocs
which python

# Setup
EXPERIMENT_DIR=experiments/PRETRAIN_$MVLM_DATASET/TC_$TOKEN_CLASSIF_DATASET
MVLM_SAVE_DIR=experiments/PRETRAIN_$MVLM_DATASET

mkdir -p $EXPERIMENT_DIR
mkdir -p $MVLM_SAVE_DIR

# Some prints
echo GOT THE FOLLOWING PARAMS:
echo EXPERIMENT_DIR=$EXPERIMENT_DIR
echo MVLM_DATASET=$MVLM_DATASET
echo TOKEN_CLASSIF_DATASET=$TOKEN_CLASSIF_DATASET
# echo FULL_PIPELINE_BASE_PATH=$FULL_PIPELINE_BASE_PATH
echo SUBSET_INDEX_PATH=$SUBSET_INDEX_PATH
echo HOME_DIR=$HOME_DIR

echo MODEL_NAME=$MODEL_NAME
echo PRETRAINED_MODEL=$PRETRAINED_MODEL
echo PREPROCESS_NORMALIZE_TXT=$PREPROCESS_NORMALIZE_TXT
echo PREPROCESS_TAG_SCHEME=$PREPROCESS_TAG_SCHEME

echo EPOCH_NUM=$EPOCH_NUM
echo BATCH_SIZE=$BATCH_SIZE
echo LEARNING_RATE=$LEARNING_RATE
echo GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS
echo MVLM_CONFIG_PATH=$MVLM_CONFIG_PATH

# Some exports

export EXPERIMENT_DIR=$EXPERIMENT_DIR
export MVLM_DATASET=$MVLM_DATASET
export TOKEN_CLASSIF_DATASET=$TOKEN_CLASSIF_DATASET
# export FULL_PIPELINE_BASE_PATH=$FULL_PIPELINE_BASE_PATH
export HOME_DIR=$HOME_DIR
export MODEL_NAME=$MODEL_NAME
export PRETRAINED_MODEL=$PRETRAINED_MODEL
export PREPROCESS_NORMALIZE_TXT=$PREPROCESS_NORMALIZE_TXT
export PREPROCESS_TAG_SCHEME=$PREPROCESS_TAG_SCHEME
export EPOCH_NUM=$EPOCH_NUM
export BATCH_SIZE=$BATCH_SIZE
export LEARNING_RATE=$LEARNING_RATE
export GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS
export MVLM_CONFIG_PATH=$MVLM_CONFIG_PATH

# 1. Proceed with Unsupervised MVLM on LayoutLM model. Resulting model will be saved in 
echo Running MVLM, check at logs/PRETRAIN_${MVLM_DATASET}_TC_${TOKEN_CLASSIF_DATASET}.out


# # 2. Extract the folder where the trained model has been saved
#### After MVLM, model will be saved in experiments/PRETRAIN_${MVLM_DATASET}/

mvlm_model_folder=$(echo "$mvlm_output" | grep -oP "MVLM Model saved in: \K.*")

python /deepinsurancedocs/layoutlm/mvlm/train.py \
    --config_path "$MVLM_CONFIG_PATH"\
    --output_dir $MVLM_SAVE_DIR  \
    &> $MVLM_SAVE_DIR/logs/PRETRAIN_${MVLM_DATASET}.out

# training_date=$(echo "$mvlm_output" | grep -oP "MVLM Training date: \K.*")
# full_pipeline_folder="$FULL_PIPELINE_BASE_PATH/from_scratch_mvlm_${mvlm_dataset}_tc_${token_classif_dataset}/$training_date"
# mkdir -p "$full_pipeline_folder/"

# # 2 bis. Save the MVLM config in full_pipeline_folder
# #        Save the MVLM Tensorboard graphs in full_pipeline_folder
# cp -p "$mvlm_config_path" "$full_pipeline_folder/mvlm_config.json"|| { echo "Error during copying MVLM config file"; exit 1; }
# cp -r "$mvlm_model_folder/mvlm_runs" "$full_pipeline_folder/mvlm_runs"
# echo "MVLM Model saved in $mvlm_model_folder"
# echo "MVLM config saved in $full_pipeline_folder"

# # 3. Create a modified config file with the newly trained model and save it in full_pipeline_folder
# token_classif_config_file="/mnt/config/token_classif_$TOKEN_CLASSIF_DATASET.json"

# jq --arg mvlm_model_folder "$mvlm_model_folder" '.pretrained_model = $mvlm_model_folder' "$token_classif_config_file" | tee "$full_pipeline_folder/token_classif_config.json" > /dev/null
# token_classif_config_file="$full_pipeline_folder/token_classif_config.json"

# # 4. Proceed with Token Classification with weights of MVLM trained model
# echo "TC TRAINING WITH MVLM UPDATED WEIGHTS"

# # 2. Extract the folder where the trained model has been saved
# mvlm_model_folder=$(echo "$mvlm_output" | grep -oP "MVLM Model saved in: \K.*")
# training_date=$(echo "$mvlm_output" | grep -oP "MVLM Training date: \K.*")
# full_pipeline_folder="$FULL_PIPELINE_BASE_PATH/from_scratch_mvlm_${mvlm_dataset}_tc_${token_classif_dataset}/$training_date"
# mkdir -p "$full_pipeline_folder/"

# # 2 bis. Save the MVLM config in full_pipeline_folder
# #        Save the MVLM Tensorboard graphs in full_pipeline_folder
# cp -p "$mvlm_config_path" "$full_pipeline_folder/mvlm_config.json"|| { echo "Error during copying MVLM config file"; exit 1; }
# cp -r "$mvlm_model_folder/mvlm_runs" "$full_pipeline_folder/mvlm_runs"
# echo "MVLM Model saved in $mvlm_model_folder"
# echo "MVLM config saved in $full_pipeline_folder"

# # 3. Create a modified config file with the newly trained model and save it in full_pipeline_folder
# token_classif_config_file="/mnt/config/token_classif_$TOKEN_CLASSIF_DATASET.json"

# jq --arg mvlm_model_folder "$mvlm_model_folder" '.pretrained_model = $mvlm_model_folder' "$token_classif_config_file" > "$full_pipeline_folder/token_classif_config.json"
# token_classif_config_file="$full_pipeline_folder/token_classif_config.json"

# # 4. Proceed with Token Classification with weights of MVLM trained model
# echo "TC TRAINING WITH MVLM UPDATED WEIGHTS"

# token_classif_output=$(python /mnt/deepinsurancedocs/layoutlm/token_classif/train.py --config_path "$token_classif_config_file")|| { echo "Error during TC training"; exit 1; }
# token_classif_model_folder=$(echo "$token_classif_output" | grep -oP "TC Model saved in: \K.*")

# echo "TC Model saved in $token_classif_model_folder"
# echo "TC config saved in $full_pipeline_folder"

# # 5. Save the paths of the two models in a txt file
# cp -r "$token_classif_model_folder/tc_runs" "$full_pipeline_folder/tc_runs"
# echo -e "$mvlm_model_folder\n$token_classif_model_folder" > "$full_pipeline_folder/models_path.txt"

# # Now we have a folder with the information about which MVLM/TC datasets combination is used, with a subfolder named as the date of training. 
# # In this folder, we have the config used for MVLM and TC training + a models_paths.txt file with the paths of the MVLM trained and TC trained models
