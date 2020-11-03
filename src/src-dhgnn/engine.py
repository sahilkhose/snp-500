"""Loss, Train, Eval, Metrics, Save model functions.

Author:
    Sahil Khose (sahilkhose18@gmail.com)
"""
import config

import os 
import pandas as pd 
import torch 
import torch.nn as nn

from sklearn import metrics
from tqdm import tqdm 

def loss_fn(outputs, targets):
    '''CrossEntropyLoss: Combines Softmax and NLL loss.
    @param   outputs (torch.tensor) : Model predictions.    tensor.shape: (num_stocks, 2)
    @param   targets (torch.tensor) : Prediction label.     tensor.shape: (1, num_stocks)

    @returns loss    (torch.float)  : Cross Entropy Loss. 
    '''
    #* Data imbalance correction: [[0.5423, 0.4577]]
    loss = nn.CrossEntropyLoss(weight=torch.tensor([[0.538, 0.462]]).to(device="cuda"))
    return loss(outputs, targets.view(-1).long()) 

def train_fn(data_loader, model, optimizer, device, epoch):
    '''Train function.
    @param   data_loader (DataLoader)
    @param   model       (StockModel)
    @param   optimizer   (Adam)
    @param   device      (torch.device)
    @param   epoch       (int)               : Number of epoch for tqdm

    @returns LOSS        (torch.float)       : Loss
    @returns fin_y       (List[List[float]]) : List of list containing label. (1)
    @returns fin_outputs (List[List[float]]) : List of list containing model predictions through sigmoid. (2)
    '''
    model.train()
    LOSS = 0.
    fin_y = []  # To calculate accuracy
    fin_outputs = []  # To calculate accuracy

    #* To train on one epoch:
    # data_i = iter(data_loader) 
    # next(data_i)
    # data = next(data_i)

    for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Train Epoch {epoch}/{config.EPOCHS}"):
        #* Preparing data:
        con_e_list, adj_u_list, article_embs, y, prices = data 

        for article_emb in article_embs:
            for article in article_emb:
                article = article.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)

        #* Train:
        optimizer.zero_grad()
        outputs = model(con_e_list, adj_u_list, article_embs, prices) # (num_stocks, 2)
        loss = loss_fn(outputs, y)
        LOSS += loss
        loss.backward() 
        optimizer.step()

        fin_y.extend(y.view(-1, 1).cpu().detach().numpy().tolist())  # (num_stocks, 1)
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())  # (num_stocks, 2)
        
    LOSS/=len(data_loader)
    return fin_outputs, fin_y, LOSS
        
def eval_fn(data_loader, model, device, epoch, eval_type):
    '''Eval function.
    @param   data_loader (DataLoader)
    @param   model       (StockModel)
    @param   device      (torch.device)
    @param   epoch       (int)          : Number of epoch for tqdm
    @param   eval_type   (str)          : Valid/Test

    @returns LOSS        (torch.float)       : Loss
    @returns fin_y       (List[List[float]]) : List of list containing label. (1)
    @returns fin_outputs (List[List[float]]) : List of list containing model predictions through sigmoid. (2)
    '''
    model.eval()
    LOSS = 0.
    fin_y = []  # To calculate accuracy
    fin_outputs = []  # To calculate accuracy

    with torch.no_grad():
        #* To train on one epoch:
        # data = next(iter(data_loader))
        for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"{eval_type} Epoch {epoch}"):
            #* Preparing data:
            con_e_list, adj_u_list, article_embs, y, prices = data 

            for article_emb in article_embs:
                for article in article_emb:
                    article = article.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            #* Evaluate:
            outputs = model(con_e_list, adj_u_list, article_embs, prices).view(-1, 2)
            LOSS += loss_fn(outputs, y)
            fin_y.extend(y.view(-1, 1).cpu().detach().numpy().tolist())  # (num_stocks, 1)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())  # (num_stocks, 2)

    LOSS/=len(data_loader) 
    return fin_outputs, fin_y, LOSS

def metrics_fn(outputs, targets, loss, all_ones=False):
    '''Metrics function.
    @param   outputs (List[List[float]]) : List of list containing model predictions through sigmoid. (2)
    @param   targets (List[List[float]]) : List of list containing label. (1)
    @param   loss    (torch.float)

    @returns acc     (float)             : Accuracy
    @returns cm      (List[List[float]]) : MCC matrix
    '''
    if not all_ones:
        outputs = torch.max(torch.tensor(outputs), 1)[1]
    acc = metrics.accuracy_score(targets, outputs)
    mcc = metrics.matthews_corrcoef(targets, outputs)
    cm = metrics.confusion_matrix(targets, outputs)
    f1 = metrics.f1_score(targets, outputs)

    print(f"Loss      : {round(float(loss), 4)}")
    print(f"Accuracy  : {round(acc, 4)}")
    print(f"MCC Score : {round(mcc, 4)}")
    print(f"F1 Score  : {round(f1, 4)}")
    print(f"Confusion Matrix: \n{cm}\n")

    return acc, cm, mcc, f1

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_model(model_type, accuracy_t, all_ones_acc, model, epoch, cm_t, mcc_t, f1_t):
    '''Saves models and confusion matrices.
    @param model_type    (str)               : best/intermediate/last
    @param accuracy_t    (float)             : Test Accuracy
    @param all_ones_acc  (float)             : All ones accuracy 
    @param model         (model.StockModel)  : Model to save its parameters
    @param epoch         (int)               : Epoch number
    @param cm_t          (List[List[float]]) : Test Confusion Matrix 
    @param mc_t                                MCC score
    @param f1_t                                F1 score
    '''
    #* Saving model:
    print(f"Saving the {model_type} model! Test Accuracy: {accuracy_t}, All ones: {all_ones_acc}")
    print("Saving model: ", config.MODEL_PATH + f"{config.args.NUM}_model_{epoch}.bin")
    torch.save(model.state_dict(), config.MODEL_PATH + f"{config.args.NUM}_model_{epoch}.bin")

    #* Saving confusion matrix:
    print("Saving Confusion Matrix: ", config.CONFUSION_PATH + f"{config.args.NUM}_model_{epoch}.txt")
    cm_file = open(config.CONFUSION_PATH + f"{config.args.NUM}_model_{epoch}.txt", "w")
    for ele in cm_t:
        cm_file.write(str(ele) + "\n")

    #* Saving metrics to test_data_sheet.csv
    print(f"Saving metrics to: {config.TEST_DATA_SHEET}")
    df = pd.read_csv(config.TEST_DATA_SHEET, index_col=[0])
    df = df.append({
    'model' : f"{config.args.NUM}_model_{epoch}.bin", 
    'all_ones' : round(all_ones_acc, 4), 
    'acc' : round(accuracy_t, 4), 
    'mcc' : round(mcc_t, 4),
    'f1' : round(f1_t, 4), 
    'lr' : config.args.LR},  
    ignore_index = True
    ) 
    df.to_csv(config.TEST_DATA_SHEET)