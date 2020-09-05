import os

WEIGHT_DECAY = 0.005
MNLI_FAISS_INDEX_PATH = "/export/share/hguo/Experiments/20200713/MNLI.index"
MNLI2_FAISS_INDEX_PATH = "/export/share/hguo/Experiments/20200805/MNLI.index"
MNLI_TRAIN_INPUT_COLLECTIONS_PATH = "/export/share/hguo/Experiments/20200706/train_inputs_collections.tmp.pt.full"

HANS_DATA_DIR = "/export/share/hguo/Data/HANS/"
GLUE_DATA_DIR = "/export/share/hguo/Data/Glue/MNLI/"
MNLI_MODEL_PATH = "/export/share/hguo/Experiments/20200706/"
MNLI2_MODEL_PATH = "/export/share/hguo/Experiments/20200801/"

REMOTE_DEFAULT_SENDER_EMAIL = None
REMOTE_DEFAULT_RECIPIENT_EMAIL = None
REMOTE_DEFAULT_SENDER_PASSWORD = None
REMOTE_DEFAULT_SERVER_USERNAME = "ec2-user"
REMOTE_DEFAULT_SERVER_PASSWORD = None
REMOTE_DEFAULT_SERVER_ADDRESS = "ec2-54-172-210-41.compute-1.amazonaws.com"
REMOTE_DEFAULT_SSH_KEY_FILENAME = "./cluster/salesforce-intern-project.pem"
REMOTE_DEFAULT_REMOTE_BASE_DIR = os.getenv("REMOTE_BASE_DIR")
