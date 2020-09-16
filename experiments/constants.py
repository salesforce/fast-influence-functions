import os

# Note that the paths used in `scripts/run_*.sh` are still
# hard-coded to `/export/home/`
WEIGHT_DECAY = 0.005

MNLI_MODEL_PATH = "/export/share/hguo/Experiments/20200706/"
HANS_MODEL_PATH = "/export/share/hguo/Experiments/20200907/"
MNLI2_MODEL_PATH = "/export/share/hguo/Experiments/20200801/"

# Trained and used in MNLI
MNLI_FAISS_INDEX_PATH = "/export/share/hguo/Experiments/20200713/MNLI.index"
# Trained and used in HANS
HANS_FAISS_INDEX_PATH = "/export/share/hguo/Experiments/20200908/HANS.index"
# Trained and used in MNLI-2
MNLI2_FAISS_INDEX_PATH = "/export/share/hguo/Experiments/20200805/MNLI.index"
# Trained on MNLI2 and used in HANS
MNLI2_HANS_FAISS_INDEX_PATH = "/export/share/hguo/Experiments/20200908/MNLI2-HANS.index"
# Trained on HANS and used in MNLI2
HANS_MNLI2_FAISS_INDEX_PATH = "/export/share/hguo/Experiments/20200908/HANS-MNLI2.index"

MNLI_TRAIN_INPUT_COLLECTIONS_PATH = "/export/share/hguo/Experiments/20200706/train_inputs_collections.tmp.pt.full"

HANS_DATA_DIR = "/export/share/hguo/Data/HANS/"
GLUE_DATA_DIR = "/export/share/hguo/Data/Glue/MNLI/"
MNLI_TRAIN_FILE_NAME = "/export/share/hguo/Data/Glue/MNLI/train.tsv"
MNLI_EVAL_MATCHED_FILE_NAME = "/export/share/hguo/Data/Glue/MNLI/dev_matched.tsv"
MNLI_EVAL_MISMATCHED_FILE_NAME = "/export/share/hguo/Data/Glue/MNLI/dev_mismatched.tsv"
HANS_TRAIN_FILE_NAME = "/export/share/hguo/Data/HANS/heuristics_train_set.txt"
HANS_EVAL_FILE_NAME = "/export/share/hguo/Data/HANS/heuristics_evaluation_set.txt"

# Remote specific
REMOTE_DEFAULT_SENDER_EMAIL = None
REMOTE_DEFAULT_RECIPIENT_EMAIL = None
REMOTE_DEFAULT_SENDER_PASSWORD = None
REMOTE_DEFAULT_SERVER_USERNAME = "ec2-user"
REMOTE_DEFAULT_SERVER_PASSWORD = None
REMOTE_DEFAULT_SERVER_ADDRESS = "ec2-54-172-210-41.compute-1.amazonaws.com"
REMOTE_DEFAULT_SSH_KEY_FILENAME = "./cluster/salesforce-intern-project.pem"
REMOTE_DEFAULT_REMOTE_BASE_DIR = os.getenv("REMOTE_BASE_DIR")

# Experiments specific
MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR = "/export/share/hguo/Experiments/20200904/"

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
