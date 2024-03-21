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
            TOKEN_CLASSIF_CONFIG_PATH)       TOKEN_CLASSIF_CONFIG_PATH=${VALUE} ;;
            *)   
    esac    
done

# Navigate to dir
cd $HOME_DIR/DeepInsuranceDocs
which python

# Setup
EXPERIMENT_DIR=experiments/PRETRAIN_$MVLM_DATASET/TC_$TOKEN_CLASSIF_DATASET
MVLM_SAVE_DIR=experiments/PRETRAIN_$MVLM_DATASET
TOKEN_CLASSIF_SAVE_DIR=experiments/TC_$TOKEN_CLASSIF_DATASET

mkdir -p $EXPERIMENT_DIR
mkdir -p $TOKEN_CLASSIF_SAVE_DIR

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
echo TOKEN_CLASSIF_CONFIG_PATH=$TOKEN_CLASSIF_CONFIG_PATH

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
export TOKEN_CLASSIF_CONFIG_PATH=$TOKEN_CLASSIF_CONFIG_PATH

# 1. Proceed with Unsupervised MVLM on LayoutLM model. Resulting model will be saved in 
echo Running MVLM, check at $TOKEN_CLASSIF_SAVE_DIR/pretrain_${MVLM_DATASET}_tc_${TOKEN_CLASSIF_SAVE_DIR}.out


# # 2. Extract the folder where the trained model has been saved
#### After MVLM, model will be saved in experiments/PRETRAIN_${MVLM_DATASET}/

mvlm_model_folder=$(echo "$mvlm_output" | grep -oP "MVLM Model saved in: \K.*")

python deepinsurancedocs/layoutlm/mvlm/train.py \
    --config_path "$TOKEN_CLASSIF_CONFIG_PATH"\
    --output_dir $TOKEN_CLASSIF_SAVE_DIR  \
    &> $TOKEN_CLASSIF_SAVE_DIR/pretrain_${MVLM_DATASET}_tc_${TOKEN_CLASSIF_SAVE_DIR}.out

cat $TOKEN_CLASSIF_SAVE_DIR/pretrain_${MVLM_DATASET}_tc_${TOKEN_CLASSIF_SAVE_DIR}.out