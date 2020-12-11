WEIGHT_DECAY = 0.005

MNLI_MODEL_PATH = None
HANS_MODEL_PATH = None
MNLI2_MODEL_PATH = None
MNLI_IMITATOR_MODEL_PATH = None

# Trained and used in MNLI
MNLI_FAISS_INDEX_PATH = None
# Trained and used in HANS
HANS_FAISS_INDEX_PATH = None
# Trained and used in MNLI-2
MNLI2_FAISS_INDEX_PATH = None
# Trained on MNLI2 and used in HANS
MNLI2_HANS_FAISS_INDEX_PATH = None
# Trained on HANS and used in MNLI2
HANS_MNLI2_FAISS_INDEX_PATH = None

MNLI_TRAIN_INPUT_COLLECTIONS_PATH = None

HANS_DATA_DIR = None
GLUE_DATA_DIR = None
MNLI_TRAIN_FILE_NAME = None
MNLI_EVAL_MATCHED_FILE_NAME = None
MNLI_EVAL_MISMATCHED_FILE_NAME = None
HANS_TRAIN_FILE_NAME = None
HANS_EVAL_FILE_NAME = None


# Experiments specific
MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR = None
MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR2 = None

# Some useful default hparams for influence functions
DEFAULT_INFLUENCE_HPARAMS = {
    # `train_on_task_name`
    "mnli": {
        # `eval_task_name`
        "mnli": {
            "damp": 5e-3,
            "scale": 1e4,
            "num_samples": 1000
        }
    }
}
