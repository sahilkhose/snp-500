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
    "--BERT_SIZE",
    type=int,
    default=16,
    help="BERT embedding size"
)
parser.add_argument(
    "--EPOCHS",
    type=int,
    default=500,
    help="Number of training epochs"
)
parser.add_argument(
    "--EVAL_EVERY",
    type=int,
    default=5,
    help="number of epochs for every eval"
)
parser.add_argument(
    "--LOOKBACK_WINDOW",
    type=int,
    default=3,  # actual window = LOOKBACK_WINDOW + 1(today's data)
    help="Size of lookback window (default 3)"
)
parser.add_argument(
    "--LR",
    type=float,
    default=5e-4,
    help="learning rate"
)
parser.add_argument(
    "--model_path",
    type=str,
    default="3_model_500.bin",
    help="model path eg: 3_model_500.bin"
)
parser.add_argument(
    "--NUM",
    type=int,
    default=0,
    help="prefix number to save the models eg: 0"
)

args = parser.parse_args()
LOAD_PATH = MODEL_PATH + args.model_path

################################################################
### DATA PATH:
################################################################
DATES_PATH = "../../input/DATA/DATES.txt"
# ARTICLES = "../../input/DATA/ARTICLES/"
ARTICLES = f"../../input/DATA/PCA_ARTICLE_EMB/{args.BERT_SIZE}"
pca_article_emb_tensor = f"../../input/DATA/pca_article_emb_{args.BERT_SIZE}.pt"


data_path = f"../../input/DATA-N/filter_data/"

CON_E_PATH = data_path + "CON_E/"
ADJ_U_PATH = data_path + "ADJ_U/"
NAMES_HG_PATH = data_path + "FILTER_NAMES_HG.txt"
TICKERS_PATH = data_path + "FILTER_TICKERS.txt"
PRICE_PATH = data_path + "FILTER_PRICE.csv"
# PRICE_PATH = data_path + "FILTER_NORM_PRICE.csv"
LABELS_PATH = data_path + "FILTER_LABELS.csv"

################################################################
### FIX VALUES:
################################################################
DEVICE = "cuda"
TRAIN_BATCH_SIZE = 1