import collections
import os

import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    


def list_subfolder(dir_name:str,max_videos=-1)->list:
    """Give a train, val or test dir return a list of subfolders in dir (videos)

    Args:
        dir_name (str): path to sets
        max_videos (int, optional): max videos. Defaults to -1.

    Returns:
        list: a list of cropped faces videos
    """
    paths = []
    for index, video_folder_name in enumerate(os.listdir(dir_name)):
        if index == max_videos:
            break

        if os.path.isdir(os.path.join(dir_name, video_folder_name)):
            paths.append(os.path.join(dir_name, video_folder_name))

    return paths

def separation_frame_video(video_paths:list, dataframe:pd.DataFrame,config:dict,val=False)->list:
    """Separete frames based on params file

    Args:
        video_path (list): List of video folder paths
        dataframe (pd.DataFrame): dataframe with labels
        config (dict): params dict
        val (bool,optinonal): Defaults to False

    Returns:
        list: List of image paths and labels
    """
    paths = []
    labels = []
    for video_path in tqdm(video_paths):

        name_video = video_path.split('/')[-1]
        aux_ = dataframe[dataframe['name'] == name_video]

        if(aux_.shape[0] == 0):
            print(f"Video not found:{name_video}")
            continue
        else:
            label = aux_['label'].tolist()[0]
            frames_number = len(os.listdir(video_path))
            if label == 0:
                min_video_frames = max(int(config['dataset_config']['frames-per-video'] *
                                    config['dataset_config']['rebalancing_real']), 1)  # Compensate unbalancing
            else:
                min_video_frames = max(int(
                    config['dataset_config']['frames-per-video'] * config['dataset_config']['rebalancing_fake']), 1)
            
            if val:
                min_video_frames = int(max(min_video_frames/8, 2))
            frames_interval = int(frames_number / min_video_frames)
            frames_paths = os.listdir(video_path)
            frames_paths_dict = {}

            # Group the faces with the same index, reduce probabiity to skip some faces in the same video
            for path in frames_paths:
                for i in range(0, 1):
                    if "_" + str(i) in path:
                        if i not in frames_paths_dict.keys():
                            frames_paths_dict[i] = [path]
                        else:
                            frames_paths_dict[i].append(path)

            # Select only the frames at a certain interval
            if frames_interval > 0:
                for key in frames_paths_dict.keys():
                    if len(frames_paths_dict) > frames_interval:
                        frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]

                    frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
            
            #frames_paths_labels = []

            for key in frames_paths_dict.keys():
                for frame_image in frames_paths_dict[key]:
                    #frames_paths_labels.append([os.path.join(video_path, frame_image),label])
                    paths.append(os.path.join(video_path, frame_image))
                    labels.append(label)

            #return frames_paths_labels
    return [paths, labels]


def apply_threshold(output:torch.Tensor,threshold:float)->torch.Tensor:
    return torch.tensor([[1] if x[0] > threshold else [0] for x in output])


def print_information(epoch, train_loss, ground_true, preds, index,val=False):
    recall_fake = round(recall_score(ground_true,preds,pos_label=1,zero_division=0),4)
    recall_real = round(recall_score(ground_true,preds,pos_label=0,zero_division=0),4)

    precision_fake = round(precision_score(ground_true,preds,pos_label=1,zero_division=0),4)
    precision_real = round(precision_score(ground_true,preds,pos_label=0,zero_division=0),4)

    pred_count = collections.Counter(preds)
    true_count = collections.Counter(ground_true)

    print("", end="\r{}Epoch: {} - Loss: {:.8f} - Fake/Real Recall: {:.4f}/{:.4f} - Fake/Real Precision: {:.4f}/{:.4f} - Fake: {}/{} - Real: {}/{}".format(
            'Validation ' if val else '',
            epoch,
            train_loss/(index+1),
            recall_fake,
            recall_real,
            precision_fake,
            precision_real,
            pred_count[1],
            true_count[1],
            pred_count[0],
            true_count[0]
            ))
    
    if val:
        return recall_fake, recall_real, precision_fake, precision_real


def print_images_grid():
    # num = 4

    # col = 2
    # row = num//col if num%col == 0 else ((num//col) + 1)

    # fig, axs = plt.subplots(row,col,figsize=(20,20))

    # i,j,soma = 0,0,0

    # while(soma < num):
    #     if(j == col):
    #         j = 0
    #         i += 1
    #     #print(i,j)
    #     axs[i,j].imshow(images_[soma].numpy().astype(np.uint8))

    #     j+=1
    #     soma+=1
    pass
    
