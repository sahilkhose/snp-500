"""Training script.

Author:
    Sahil Khose (sahilkhose18@gmail.com)
"""
import config 
import dataset 
import engine 


import numpy as np 
import os 
import pandas as pd 
import torch 

from model import StockModel
from sklearn import metrics
from sklearn import model_selection
print("__"*80)
print("Imports Done...")

def run():
    #* Loading data. x: dates(str) list, y: labels dataframe
    x = sorted(open(config.DATES_PATH, "r").read().split())
    y = pd.read_csv(config.LABELS_PATH, index_col=0)

    '''
    Spliting dataset:
    60 - 20 - 20      split of total 1784 days
    1070 - 357 - 357  split
    1067 - 354 - 354  split after lookback_window slicing
    '''
    x_train, x_test = model_selection.train_test_split(x, test_size=0.2, shuffle=False, stratify=None)
    y_train, y_test = model_selection.train_test_split(y, test_size=0.2, shuffle=False, stratify=None)

    x_train, x_valid = model_selection.train_test_split(x_train, test_size=0.25, shuffle=False, stratify=None)
    y_train, y_valid = model_selection.train_test_split(y_train, test_size=0.25, shuffle=False, stratify=None)

    x_train, y_train = x_train[config.LOOKBACK_WINDOW:], y_train[config.LOOKBACK_WINDOW:]
    x_valid, y_valid = x_valid[config.LOOKBACK_WINDOW:], y_valid[config.LOOKBACK_WINDOW:]
    x_test, y_test = x_test[config.LOOKBACK_WINDOW:], y_test[config.LOOKBACK_WINDOW:]

    #* Data Loaders:
    training_set = dataset.StockDataset(x_train, y_train)
    train_data_loader = torch.utils.data.DataLoader(training_set, batch_size=1, num_workers=1)

    validation_set = dataset.StockDataset(x_valid, y_valid)
    valid_data_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, num_workers=1)

    testing_set = dataset.StockDataset(x_test, y_test)
    test_data_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, num_workers=1)

    #* Loading model, and train init
    device = torch.device(config.DEVICE)

    if not os.path.exists(config.MODEL_PATH):
        os.mkdir(config.MODEL_PATH)
        os.mkdir(config.CONFUSION_PATH)

    # load_file = config.MODEL_PATH + "3_model_500.bin"
    load_file = config.LOAD_PATH
    confusion_file = config.CONFUSION_PATH 

    model = StockModel()
    if os.path.exists(load_file):
        print("Model loading", load_file)
        model.load_state_dict(torch.load(load_file))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    num_train_steps = int(len(x_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    best_accuracy = 0

    for epoch in range(1, config.EPOCHS+1): #TODO change this to while loop for inf train
        #* Train metrics:
        print("__"*80)
        outputs_tr, targets_tr, loss_tr = engine.train_fn(train_data_loader, model, optimizer, device, epoch)
        print(f"\nEpoch {epoch} Train metrics:")
        engine.metrics_fn(outputs_tr, targets_tr, loss_tr)
       
       #* All ones metrics:
        if epoch == 1:
            print("\nALL ONES")
            outputs_t, targets_t, loss_t = engine.eval_fn(test_data_loader, model, device, epoch, "Test")
            # engine.metrics_fn(list(np.ones((len(targets_t)))), targets_t, 0, all_ones=True)
            all_ones_acc = metrics.accuracy_score(targets_t, list(np.ones((len(targets_t)))))
            all_ones_cm = metrics.confusion_matrix(targets_t, list(np.ones((len(targets_t)))))
            print(f"Accuracy  : {round(all_ones_acc, 4)}")
            print(f"Confusion Matrix: \n{all_ones_cm}\n")

        #* Validation, Testing and saving models:
        # if (epoch % config.EVAL_EVERY == 0) or (epoch % 50 == 0) or (epoch == config.EPOCHS):
        #     #* Validation:
        #     outputs_v, targets_v, loss_v = engine.eval_fn(valid_data_loader, model, device, epoch, "Valid")
        #     print(f"\nEpoch {epoch} Valid metrics:")
        #     engine.metrics_fn(outputs_v, targets_v, loss_v)

        #     #* Testing:
        #     outputs_t, targets_t, loss_t = engine.eval_fn(test_data_loader, model, device, epoch, "Test")
        #     print(f"\nEpoch {epoch} Test metrics:")
        #     accuracy_t, cm_t, _ = engine.metrics_fn(outputs_t, targets_t, loss_t)
            
        #     #* Saving models and Confusion Matrix:
        #     if accuracy_t > best_accuracy:
        #         engine.save_model("best", accuracy_t, all_ones_acc, model, epoch, cm_t)
        #         best_accuracy = accuracy_t

        #     elif epoch % 50 == 0:
        #         engine.save_model("intermediate", accuracy_t, all_ones_acc, model, epoch, cm_t)

        #     elif epoch == config.EPOCHS:
        #         engine.save_model("last", accuracy_t, all_ones_acc, model, epoch, cm_t)

if __name__ == "__main__":
    run()