## Everything needed to run an experiment on its own aws instance

# Parse keyword arguments into environment variables
# derived from https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            MVLM_DATASET)                    MVLM_DATASET=${VALUE} ;;
            MVLM_VAL_DATASET)                MVLM_VAL_DATASET=${VALUE} ;;
            TOKEN_CLASSIF_DATASET)           TOKEN_CLASSIF_DATASET=${VALUE} ;;
            IS_DOCILE)                       IS_DOCILE=${VALUE} ;;
            SUBSET_INDEX_PATH)               SUBSET_INDEX_PATH=${VALUE} ;;
            HOME_DIR)                        HOME_DIR=${VALUE} ;;

            HF_MODEL_DIR)                      HF_MODEL_DIR=${VALUE} ;;
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
cd $HOME_DIR
echo $(pwd)
which python

# Setup
EXPERIMENT_DIR=experiments/pretrain_$MVLM_DATASET/tc_$TOKEN_CLASSIF_DATASET
MVLM_SAVE_DIR=experiments/pretrain_$MVLM_DATASET
MVLM_TRAIN_DATA_PATH=data/docile/$MVLM_DATASET
MVLM_VAL_DATA_PATH=data/$MVLM_VAL_DATASET
mkdir -p $EXPERIMENT_DIR
mkdir -p $MVLM_SAVE_DIR


# Some prints
# echo "-------------- GOT THE FOLLOWING PARAMS: --------------"
# echo EXPERIMENT_DIR=$EXPERIMENT_DIR
# # echo MVLM_DATASET=$MVLM_DATASET
# echo MVLM_TRAIN_DATA_PATH=$MVLM_TRAIN_DATA_PATH
# # echo MVLM_VAL_DATASET=$MVLM_VAL_DATASET
# echo MVLM_VAL_DATA_PATH=$MVLM_VAL_DATA_PATH
# echo IS_DOCILE=$IS_DOCILE

# echo TOKEN_CLASSIF_DATASET=$TOKEN_CLASSIF_DATASET
# echo SUBSET_INDEX_PATH=$SUBSET_INDEX_PATH
# echo HF_MODEL_DIR=$HF_MODEL_DIR
# echo HOME_DIR=$HOME_DIR

# echo PRETRAINED_MODEL=$PRETRAINED_MODEL
# echo PREPROCESS_NORMALIZE_TXT=$PREPROCESS_NORMALIZE_TXT
# echo PREPROCESS_TAG_SCHEME=$PREPROCESS_TAG_SCHEME

# echo EPOCH_NUM=$EPOCH_NUM
# echo BATCH_SIZE=$BATCH_SIZE
# echo LEARNING_RATE=$LEARNING_RATE
# echo GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS
# echo MVLM_CONFIG_PATH=$MVLM_CONFIG_PATH
# Some exports


export EXPERIMENT_DIR=$EXPERIMENT_DIR
export MVLM_DATASET=$MVLM_DATASET
export MVLM_TRAIN_DATA_PATH=$MVLM_TRAIN_DATA_PATH
export MVLM_VAL_DATA_PATH=$MVLM_VAL_DATA_PATH

export TOKEN_CLASSIF_DATASET=$TOKEN_CLASSIF_DATASET
export HOME_DIR=$HOME_DIR
export HF_MODEL_DIR=$HF_MODEL_DIR
export PRETRAINED_MODEL=$PRETRAINED_MODEL
export PREPROCESS_NORMALIZE_TXT=$PREPROCESS_NORMALIZE_TXT
export PREPROCESS_TAG_SCHEME=$PREPROCESS_TAG_SCHEME
export EPOCH_NUM=$EPOCH_NUM
export BATCH_SIZE=$BATCH_SIZE
export LEARNING_RATE=$LEARNING_RATE
export GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS
export MVLM_CONFIG_PATH=$MVLM_CONFIG_PATH

# 1. Proceed with Unsupervised MVLM on LayoutLM model. Resulting model will be saved in 
echo "-------------- RUNNING MVLM --------------"
echo Running MVLM, check at $MVLM_SAVE_DIR/pretrain_${MVLM_DATASET}.out

# # 2. Extract the folder where the trained model has been saved
#### After MVLM, model will be saved in experiments/PRETRAIN_${MVLM_DATASET}/

python deepinsurancedocs/layoutlm/mvlm/train.py \
    --train_data_dir $MVLM_TRAIN_DATA_PATH \
    --is_docile $IS_DOCILE \
    --validation_data_dir $MVLM_VAL_DATA_PATH \
    --pretrained_model $PRETRAINED_MODEL \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epoch_num $EPOCH_NUM \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --tagging_scheme $PREPROCESS_TAG_SCHEME \
    --model_dir $HF_MODEL_DIR \
    --output_dir $MVLM_SAVE_DIR > $MVLM_SAVE_DIR/pretrain_${MVLM_DATASET}.out 2>&1 

cat $MVLM_SAVE_DIR/pretrain_${MVLM_DATASET}.out