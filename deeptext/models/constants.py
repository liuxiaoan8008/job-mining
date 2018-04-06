
# Base

PARAM_KEY_MODEL_NAME = "model_name"
PARAM_KEY_MODEL_DIR = "model_dir"
PARAM_KEY_MODEL_ERR_DIR = "err_dir"

TENSOR_NAME_PREDICTION  = "deeptext/models/prediction"
TENSOR_NAME_LOSS        = "deeptext/models/loss"

# Sequence Labeling

PARAM_KEY_TOKEN_VOCAB_SIZE = "token_vocab_size"
PARAM_KEY_LABEL_VOCAB_SIZE = "label_vocab_size"
PARAM_KEY_MAX_DOCUMENT_LEN = "max_document_len"
PARAM_KEY_EMBEDDING_SIZE = "embedding_size"
PARAM_KEY_DROPOUT_PROB = "dropout_prob"

FILENAME_TOKEN_VOCAB = "token.vocab"
FILENAME_LABEL_VOCAB = "label.vocab"

TENSOR_NAME_TOKENS      = "deeptext/models/sequence_labeling/tokens"
TENSOR_NAME_LABELS      = "deeptext/models/sequence_labeling/labels"
TENSOR_NAME_LOGITS      = "deeptext/models/sequence_labeling/logits"

SUMMARY_FILE_PATH = "/tmp/deeptext/models/sequence_labeling/base_model"

# word2vec
FILENAME_WORD2VEC_MODEL = 'w2v.model'