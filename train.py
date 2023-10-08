import collections
import os

import numpy as np
import pandas as pd
import torch
import yaml

from dataset.deepfakes_dataloader import get_dataloader
from dataset.deepfakes_dataset import DeepFakesDataset
from models.efficient_vit import EfficientViT
from utils import (apply_threshold, get_n_params, list_subfolder,
                   print_information, separation_frame_video)

if __name__ == '__main__':
    with open("./params.yaml", 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if config['training']['efficient_net'] == 0:
        channels = 1280
    else:
        channels = 2560

    model = EfficientViT(config=config, channels=channels,
                            selected_efficient_net=config['training']['efficient_net'])
    model = model.train()


    optimizer = eval(config['training']['optimizer'])
    scheduler = eval(config['training']['scheduler'])
    starting_epoch = 0
    if config['training']['resume']:
        model.load_state_dict(torch.load(config['training']['resume']))
        # The checkpoint's file name format should be "checkpoint_EPOCH"
        starting_epoch = int(config['training']['resume'].split('/')[-1].split('_')[0].replace('epoch',''))
        print(f"Checkpoint loaded at {starting_epoch+1} epoch")
    else:
        print("No checkpoint loaded.")

    print("Model Parameters:", get_n_params(model))


    train_paths = list_subfolder(config['dataset_config']['train_dir'])
    val_paths = list_subfolder(config['dataset_config']['val_dir'])
    test_paths = list_subfolder(config['dataset_config']['test_dir'])

    df_metadata = pd.read_csv(config['dataset_config']['labels_dataframe'])


    x_train, y_train = separation_frame_video(train_paths,df_metadata,config)
    x_val, y_val = separation_frame_video(val_paths,df_metadata,config)
    x_test, y_test = separation_frame_video(test_paths,df_metadata,config)



    # Print some useful statistics
    print("Train images:", len(x_train),
            "Validation images:", len(x_val))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(y_train)
    print(train_counters)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(y_val)
    print(val_counters)
    print("___________________")


    loss_fn = eval(config['training']['loss'])
    print(f"Loss function: {loss_fn}")

    train_dataset = DeepFakesDataset(x_train,y_train,config['dataset_config']['train_transform_function'])
    val_dataset = DeepFakesDataset(x_val,y_val,config['dataset_config']['val_transform_function'])
    test_dataset = DeepFakesDataset(x_test,y_test,config['dataset_config']['val_transform_function'])

    train_dataloader = get_dataloader(train_dataset,config['dataset_config']['batch_size'],config['dataset_config']['workers'])
    val_dataloader = get_dataloader(val_dataset,config['dataset_config']['batch_size'],config['dataset_config']['workers'])
    test_dataloader = get_dataloader(test_dataset,config['dataset_config']['batch_size'],config['dataset_config']['workers'])     


    model = model.train()
    model = model.cuda()
    not_improved_loss = 0
    previous_loss = np.inf

    MODELS_PATH = config['training']['dir_checkpoint']
    CHECKPOINT_MODELS_PATH = f"{MODELS_PATH}/{config['training']['name_checkpoint']}"

    for folder_ in [MODELS_PATH,CHECKPOINT_MODELS_PATH]:
        if not os.path.exists(folder_):
            os.makedirs(folder_)

    for epoch in range(starting_epoch+1,config['training']['num_epochs']+1):
        if not_improved_loss == config['training']['patience']:
            print("Loss did not improved, stoping training")
            break

        train_loss = 0

        ground_true = []
        preds = []

        for index, (images_,labels_) in enumerate(train_dataloader):
            images = np.transpose(images_, (0, 3, 1, 2))
            labels = labels_.unsqueeze(1)
            images = images.cuda()

            y_pred = model(images)
            y_pred = y_pred.cpu()
            
            loss = loss_fn(y_pred, labels.type(torch.float32))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = torch.sigmoid(y_pred)
            y_pred = apply_threshold(y_pred,config['training']['threshold'])

            preds.extend([x[0] for x in y_pred.tolist()])
            ground_true.extend([x[0] for x in labels.tolist()])

            
            print_information(epoch, train_loss, ground_true, preds, index)
            
        print("")

        with torch.no_grad():
            model = model.eval()
            val_loss = 0

            ground_true = []
            preds = []
            index_ = 0

            for index, (images_,labels_) in enumerate(val_dataloader):
                index_ = index
                images = np.transpose(images_, (0, 3, 1, 2))
                labels = labels_.unsqueeze(1)
                images = images.cuda()

                y_pred = model(images)
                y_pred = y_pred.cpu()

                loss = loss_fn(y_pred, labels.type(torch.float32))
                val_loss += loss.item()

                y_pred = torch.sigmoid(y_pred)
                y_pred = apply_threshold(y_pred,config['training']['threshold'])

                preds.extend([x[0] for x in y_pred.tolist()])
                ground_true.extend([x[0] for x in labels.tolist()])

            recall_fake, recall_real, precision_fake, precision_real = print_information(epoch, val_loss, ground_true, preds, index_,val=True)
            print("\n")

        scheduler.step()

        val_loss /= index_ + 1

        if previous_loss <= val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
            #pt_files = [x for x in os.listdir(CHECKPOINT_MODELS_PATH) if x.endswith('.pt')]
        else:
            not_improved_loss = 0

            torch.save(
                model.state_dict(), 
                f"{CHECKPOINT_MODELS_PATH}/epoch{epoch}_recall_fake{recall_fake}_recall_real{recall_real}_precision_fake{precision_fake}_precision_real{precision_real}.pt"    
            )

        previous_loss = val_loss

        
            