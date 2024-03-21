MVLM_DATASET=docile_5k
TOKEN_CLASSIF_DATASET=docile
# FULL_PIPELINE_BASE_PATH=
SUBSET_INDEX_PATH=data/docile/subsets_index/5k.json
HOME_DIR=$WORK/DeepInsuranceDocs  # To adjust

MODEL_NAME=layoulm-base-uncased
PRETRAINED_MODEL=null
PREPROCESS_NORMALIZE_TXT=true
PREPROCESS_TAG_SCHEME=BIO

EPOCH_NUM=5
BATCH_SIZE=10
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION_STEPS=0
MVLM_CONFIG_PATH=experiments/from_scratch_mvlm_docile_5k.json

# Run MVLM on docile_5k without pretraining
bash scripts/run_pretraining.sh\
    MVLM_DATASET=docile_5k\
    TOKEN_CLASSIF_DATASET=docile\
    SUBSET_INDEX_PATH=data/docile/subsets_index/5k.json\
    HOME_DIR=$WORK/DeepInsuranceDocs\
    MODEL_NAME=layoulm-base-uncased\
    PRETRAINED_MODEL=null\
    PREPROCESS_NORMALIZE_TXT=true\
    PREPROCESS_TAG_SCHEME=BIO\
    EPOCH_NUM=5\
    BATCH_SIZE=10\
    LEARNING_RATE=5e-5\
    GRADIENT_ACCUMULATION_STEPS=0\
    MVLM_CONFIG_PATH=experiments/from_scratch_mvlm_docile_5k.json

# Run Token Classif on docile with model pretrained on docile_5k
