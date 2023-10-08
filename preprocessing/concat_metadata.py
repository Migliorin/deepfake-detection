import argparse
import json
import os

import pandas as pd


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str,
                        help='Path with json metadatas')
    parser.add_argument('--output_path', type=str,
                        help='Output path to save dataframe')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':

    opt = get_argparse()
    path = opt.metadata_path
    output_path = opt.output_path

    aux = {
        "name":[],
        "label":[]
    }
    for path_ in os.listdir(path):
        if(not path_.endswith('json')):
            continue
        with open(f"{path}/{path_}","r") as infile:
            load = json.load(infile)
            infile.close()

        for key_ in load.keys():
            if(key_ not in aux['name']):
                aux['name'].append(key_.replace(".mp4",""))
                aux['label'].append(1 if load[key_]['label'] == 'FAKE' else 0)
            else:
                raise Exception('Video already in list')

    pd.DataFrame(aux).to_csv(f"{output_path}/metadata.csv",index=False)