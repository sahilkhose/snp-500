import argparse

################################################################
### CHANGE:
################################################################
STOCK_NUM = 116  # 481/119/50 #TODO could automate this

MODEL_PATH = f"../../models/models-dhgnn/{STOCK_NUM}/"
CONFUSION_PATH = f"../../models/models-dhgnn/{STOCK_NUM}/CONFUSION/"
TEST_DATA_SHEET = f"../../models/models-dhgnn/{STOCK_NUM}/test_data_sheet.csv"
TRAIN_DATA_SHEET = f"../../models/models-dhgnn/{STOCK_NUM}/train_data_sheet.csv"
parser = argparse.ArgumentParser("Train  model args")
parser.add_argument(
    "--NUM",
    type=int,
    default=0,
    help="prefix number to save the models eg: 0"
)
# NUM = 0
parser.add_argument(
    "--model_path",
    type=str,
    default="3_model_500.bin",
    help="model path eg: 3_model_500.bin"
)
parser.add_argument(
    "--LR",
    type=float,
    default=5e-4,
    help="learning rate"
)
# LR = 5e-4
parser.add_argument(
    "--EVAL_EVERY",
    type=int,
    default=5,
    help="number of epochs for every eval"
)
# EVAL_EVERY = 5  # epochs
args = parser.parse_args()
LOAD_PATH = MODEL_PATH + args.model_path
EPOCHS = 500
BERT_SIZE = 64

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

################################################################
### FIX VALUES:
################################################################
LOOKBACK_WINDOW = 3  # actual window = LOOKBACK_WINDOW + 1(today's data)
DEVICE = "cuda"
BERT_PATH = "../../input/bert_base_uncased"
TRAIN_BATCH_SIZE = 1