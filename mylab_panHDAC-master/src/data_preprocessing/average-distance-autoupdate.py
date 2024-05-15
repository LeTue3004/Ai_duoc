import pandas as pd
import numpy as np
from tabulate import tabulate

import os
import psycopg2

from dotenv import load_dotenv
import logging
from tqdm import tqdm
import multiprocessing
import time

from pharmacy_common import PharmacyCommon
#Rdkit ultis
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys

import sys
sys.path.append("/home/mylab-pharma/Code/tuele/pan_HDAC/mylab_panHDAC-master/src/common")

#class to encode smiles
common = PharmacyCommon()
"""Notification: This file is used to calculate the average distance between a smiles and its 5 nearest-neighbor in the database."""

def import_train_fpts(bits):
    train_test_path = "/home/mylab-pharma/Code/tuele/pan_HDAC/mylab_panHDAC-master/data/train_test_data/NoCL/20240321_pan_HDAC_train_test_data.xlsx"
    train_dataset = pd.read_excel(train_test_path, sheet_name='train_dataset')
    print("Train data imported:")
    print(len(train_dataset))
    #Fingerprint ECFP4, ECFP6
    X_Train = common.gen_ecfp6_fpts(train_dataset['SMILES'], bits= bits)
    return X_Train

def cal_tc(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("The arrays must have the same length.")
    
    # Calculate the Tanimoto coefficient
    intersection = sum(a and b for a, b in zip(array1, array2))
    union = sum(a or b for a, b in zip(array1, array2))
    
    if union == 0:  # Handle the case when both arrays are all zeros
        return 0.0
    else:
        tanimoto_coefficient = intersection / union
        return tanimoto_coefficient

def calculate_distances(identifier, screening_vectors, X_Train):
    distances = []
    for screening_vector in screening_vectors:
        tc_dist = np.sum(screening_vector & X_Train, axis=1) / np.sum(screening_vector | X_Train, axis=1)
        distances.append(tc_dist)
    return identifier, distances

def find_nearest_neighbors_distance(X_Screening, X_Train, n_neighbors):
    nearest_neighbors_distances = []
    nearest_neighbors_indices = []
    X_Screening = np.array(X_Screening)
    X_Train = np.array(X_Train)
    if(np.size(X_Screening) == 0):
        return nearest_neighbors_distances, nearest_neighbors_indices
     
    if X_Screening.shape[1] != X_Train.shape[1]:
        raise ValueError("X_Screening bit vectors must have the same size as X_Train bit vectors: " + str(X_Train.shape[1]))

    num_processes = multiprocessing.cpu_count()
    
    if len(X_Screening) <= num_processes:
        screening_chunks = [(i, X_Screening[i:i + 1]) for i in range(len(X_Screening))]
    else:
        chunk_size = len(X_Screening) // num_processes
        screening_chunks = [(i, X_Screening[i:i + chunk_size]) for i in range(0, len(X_Screening), chunk_size)]

    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.starmap(calculate_distances, [(i, chunk, X_Train) for i, chunk in screening_chunks])
    pool.close()
    pool.join()

    # Sort the results by identifier to ensure the correct order
    results.sort(key=lambda x: x[0])

    # Extract the distances and indices
    for _, distances in results:
        for distance in distances:
            # Get the indices of the first n_neighbors elements with the largest Tanimoto coefficients
            nearest_neighbor_indices = np.argsort(distance)[::-1][:n_neighbors]

            # Extract the distances to the nearest neighbors
            nearest_neighbors_distances.append([distance[i] for i in nearest_neighbor_indices])
            nearest_neighbors_indices.append(nearest_neighbor_indices)

    return nearest_neighbors_distances, nearest_neighbors_indices

def update_average_distance(screening_data, n_neighbors, X_train, bits):
    logging.info(f"[+] Update average distance for screening dataset")
    working_dataset = screening_data
    if(len(working_dataset)>0):
        X_Screening = common.gen_ecfp6_fpts(working_dataset["SMILES"], bits)
        logging.info(f"[-] Start finding nearest neighbor for: {len(X_Screening)}")
        #Find nearest neighbor
        dist_array, nn_idx = find_nearest_neighbors_distance(X_Screening=X_Screening, n_neighbors=n_neighbors, X_Train=X_train)
        result_df = pd.DataFrame({'SMILES': working_dataset['SMILES'], 
                                  'AVG_DISTANCE': np.average(dist_array, axis=1)})
        for idx, row in working_dataset.iterrows():
            smiles = row['SMILES']
            nn_dist = dist_array[idx]
            #Calculate average distance
            avg_distance = np.average(nn_dist)
            result_df = pd.concat([result_df, pd.DataFrame({'SMILES': [smiles], 'AVG_DISTANCE': [avg_distance]})])
    else:
        logging.info("[-] Empty data, skip this batch!")
        print("[-] Empty data, skip this batch!")    

    # Lưu DataFrame vào tệp Excel
    result_df.to_excel(f'/home/mylab-pharma/Code/tuele/pan_HDAC/mylab_panHDAC-master/results/avg_distance/20240415_average_distance_ecfp6_{bits}.xlsx', index=False)
    logging.info("[-] Finished finding, saved average distance to Excel!")
    logging.info("\n")

def main():
    #Import the X_Train data
    fpt_bits = 2048
    X_Train = import_train_fpts(bits = fpt_bits)
    # nn_nums: numbers of nearest-neighbors
    ## AD for ECFP4 2048bits, ECFP6 2048bits, ECFP4 1024bits
    nn_nums = 5
    screening_data_path = "/home/mylab-pharma/Code/tuele/pan_HDAC/mylab_panHDAC-master/data/screening_data/all_screen_data.xlsx"
    screening_data = pd.read_excel(screening_data_path, sheet_name='final_screen_data')
    update_average_distance(screening_data=screening_data, n_neighbors=nn_nums,X_train = X_Train, bits= fpt_bits)

if __name__=="__main__":
    main()
