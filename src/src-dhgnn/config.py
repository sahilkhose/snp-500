# import transformers

################################################################
### CHANGE:
################################################################
STOCK_NUM = 116  # 481/119/50 #TODO could automate this

MODEL_PATH = f"../../models/models-dhgnn/{STOCK_NUM}/"
CONFUSION_PATH = f"../../models/models-dhgnn/{STOCK_NUM}/CONFUSION/"

NUM = 0
EPOCHS = 500
LOAD_PATH = MODEL_PATH + "3_model_500.bin"
LR = 5e-4
EVAL_EVERY = 5  # epochs
BERT_SIZE = 768

################################################################
### DATA PATH:
################################################################
DATES_PATH = "../../input/DATA/DATES.txt"
ARTICLES = "../../input/DATA/ARTICLES/"
# data_path = f"../input/DATA-{STOCK_NUM}/"
data_path = f"../../input/DATA-N/filter_data/"

HG_PATH = data_path + "FILTER_HYPERGRAPHS/"
NAMES_HG_PATH = data_path + "FILTER_NAMES_HG.txt"
TICKERS_PATH = data_path + "FILTER_TICKERS.txt"
STOCK_EMB_PATH = data_path + "sent_embs.pt"
LABELS_PATH = data_path + "FILTER_LABELS.csv"
PRICE_PATH = data_path + "FILTER_PRICE.csv"


# HG_PATH = data_path + "HYPERGRAPHS/"
# NAMES_HG_PATH = data_path + "NAMES_HG.txt"
# TICKERS_PATH = data_path + "NEW_TICKERS.txt"
# STOCK_EMB_PATH = data_path + "sent_embs.pt"
# LABELS_PATH = data_path + "NEW_LABELS.csv"
# PRICE_PATH = data_path + "PRICE.csv"

################################################################
### FIX VALUES:
################################################################
LOOKBACK_WINDOW = 3  # actual window = LOOKBACK_WINDOW + 1(today's data)
DEVICE = "cuda"
BERT_PATH = "../../input/bert_base_uncased"
TRAIN_BATCH_SIZE = 1
# TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
# zero init for node_emb: Accuracy Score = 0.5422847880474999