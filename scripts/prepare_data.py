"""
Created by: Anshuman Dewangan
Date: 2021

Description: Generates .npy files for all the labels and metadata.pkl file for easy dataloading.
"""

# Other package imports
import pickle
import os
import cv2
import numpy as np
from pathlib import Path

# File imports
import util_fns


# TODO: Make sure these have correct paths
raw_data_path = 'data/dataset_artigo/raw_images'
labels_path = 'data/dataset_artigo/labels'
labels_output_path = 'data/dataset_artigo/drive_clone_numpy_new'


print("Preparing Data...")

### Initialize metadata.pkl ###
metadata = {}

metadata['fire_to_images'] = util_fns.generate_fire_to_images(raw_data_path)

metadata['omit_no_xml'] = []
metadata['omit_no_contour'] = []
metadata['omit_no_contour_or_bbox'] = []
metadata['omit_mislabeled'] = np.loadtxt('./data/omit_mislabeled.txt', dtype=str)

metadata['labeled_fires'] = [folder.stem for folder in filter(Path.is_dir, Path(labels_path).iterdir())]
metadata['unlabeled_fires'] = []
metadata['train_only_fires'] = []
metadata['eligible_fires'] = []
metadata['monochrome_fires'] = []
metadata['night_fires'] = np.loadtxt('./data/night_fires.txt', dtype=str)
metadata['mislabeled_fires'] = np.loadtxt('./data/mislabeled_fires.txt', dtype=str)

metadata['bbox_labels'] = {}

# Loop through all fires
for i, fire in enumerate(metadata['fire_to_images']):
    if 'mobo-m' in fire:
        metadata['monochrome_fires'].append(fire)
        continue
    if fire not in metadata['labeled_fires']:
        metadata['unlabeled_fires'].append(fire)
    elif 'mobo-c' not in fire:
        metadata['train_only_fires'].append(fire)
    else:
        metadata['eligible_fires'].append(fire)

    print('Preparing Folder ', i, fire)
    
    # Skip the next steps if the fire has not been labeled
    if fire in metadata['labeled_fires']:
        # print(fire)
        # Loop through each image in the fire

        for image in metadata['fire_to_images'][fire]:
            label_path = labels_path+'/'+\
                    util_fns.get_fire_name(image)+'/'+\
                    util_fns.get_only_image_name(image)+'.jpg.xml'
            has_xml_label = os.path.isfile(label_path)

            # If a positive image does not have an XML file associated with it, add it to omit_images_list
            if util_fns.get_ground_truth_label(image) != has_xml_label:
                metadata['omit_no_xml'].append(image)

            # If a label file exists...
            if has_xml_label:
                # Convert XML file into poly arrays
                poly_contour = None
                poly_bbox=None


                poly_contour = util_fns.xml_to_contour(label_path)

                if (len(poly_contour)==0):
                    poly_contour = None
                    poly_bbox = util_fns.xml_to_bbox(label_path)



                if poly_contour is not None:
                    # If there is a contour, use that to fill labels
                    x = cv2.imread(raw_data_path + '/' + image + '.jpg')
                    labels = np.zeros(x.shape[:2], dtype=np.uint8) 
                    cv2.fillPoly(labels, poly_contour, 1)
                    os.makedirs(labels_output_path + '/' + fire, exist_ok=True)
                    np.save(labels_output_path + '/' + image + '.npy', labels.astype(np.uint8))

                    # print(labels)
                    # cv2.imwrite('frame.jpg',labels)
                    # exit()

                else:
                    metadata['omit_no_contour'].append(image)

                    if poly_bbox is not None:
                        # If there isn't a contour but there is a bbox, use that to fill labels
                        x = cv2.imread(raw_data_path + '/' + image + '.jpg')
                        labels = np.zeros(x.shape[:2], dtype=np.uint8) 

                        for rect in poly_bbox:
                            cv2.rectangle(labels, rect[0],rect[1], 1, -1)

                        #     cv2.rectangle(labels, rect[0],rect[1], 255, -1)
                            

                        # if(len(poly_bbox)>1):
                        #     print(poly_bbox)
                        #     cv2.imwrite('frame.jpg',labels)
                        #     exit()


                        os.makedirs(labels_output_path + '/' + fire, exist_ok=True)
                        np.save(labels_output_path + '/' + image + '.npy', labels.astype(np.uint8))
                    else:
                        metadata['omit_no_contour_or_bbox'].append(image)

                # If there is a bbox, save the array to 'bbox_labels'
                if poly_bbox is not None:
                    metadata['bbox_labels'][image] = list(np.array(poly_bbox).flatten())

    with open(f'./data/metadata_dataset_artigo.pkl', 'wb') as pkl_file:
        pickle.dump(metadata, pkl_file)

print("Preparing Data Complete.")