import pandas as pd
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

def xml_to_contour(xml_file):
    """
    Description: Takes an XML label file and converts it to Numpy array
    Args:
        - xml_file (str): Path to XML file
    Returns:
        - all_polys (Numpy array): Numpy array with labels
    """
    tree = ET.parse(xml_file)
    for cur_object in tree.findall('object'):

        if cur_object.find('name').text=="fire":
            all_polys = []
            
            for cur_poly in cur_object.findall('polygon'):
                cur_poly_pts = []
                for cur_pt in cur_poly.findall('pt'):
                    cur_poly_pts.append([int(cur_pt.find('x').text), int(cur_pt.find('y').text)])
                all_polys.append(cur_poly_pts)
            
            all_polys = np.array(all_polys, dtype=np.int32)
            return all_polys
    return None

def xml_to_bbox(xml_file):
    """
    Description: Takes an XML label file and converts it to Numpy array
    Args:
        - xml_file (str): Path to XML file
    Returns:
        - all_polys (Numpy array): Numpy array with labels
    """
    tree = ET.parse(xml_file,parser = ET.XMLParser(encoding = 'iso-8859-5'))
    all_polys = []
    
    for cur_object in tree.findall('object'):
        # if cur_object.find('deleted').text=="1":
        #     continue
        
        if cur_object.find('name').text=="fire":
            x_s = []
            y_s = []
            for cur_pt in cur_object.find('bndbox'):
                if 'xm' in cur_pt.tag:
                    x_s.append(int(round(float(cur_pt.text))))
                if 'ym' in cur_pt.tag:
                    y_s.append(int(round(float(cur_pt.text))))

            polys = [[min(x_s), min(y_s)], [max(x_s), max(y_s)]]
            all_polys.append(polys)


    all_polys = np.array(all_polys, dtype=np.int32)
    return all_polys
    

# def read_box_annotation(xml_path):
#     if not os.path.isfile(xml_path):
#         return []
        
#     tree = ET.parse(xml_path) 
#     root = tree.getroot() # get root object

#     height = int(root.find("size")[0].text)
#     width = int(root.find("size")[1].text)
#     channels = int(root.find("size")[2].text)


#     bbox_coordinates = []
#     # for member in root.findall('object'):
#         # class_name = member[0].text # class name
#         # print(member.tag)
#         # # bbox coordinates
#         # # xmin = int(member[4][0].text)
#         # # ymin = int(member[4][1].text)
#         # # xmax = int(member[4][2].text)
#         # # ymax = int(member[4][3].text)
#         # ymin = int(member.find("bndbox/ymin").text)
#         # xmin = int(member.find("bndbox/xmin").text)
#         # ymax = int(member.find("bndbox/ymax").text)
#         # xmax = int(member.find("bndbox/xmax").text)
#         # # store data in list
#         # bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])

#     return(bbox_coordinates)


def draw_image(image_name, 
                label_path,
                image_pred=None, 
                image_prob=None,
                tile_preds=None, 
                tile_probs=None, 
                tile_labels=None,
                grid=True):
    img = cv2.imread(image_name)
    h, w, _ = img.shape
    rows, cols = 5,9
    dy, dx = h / rows, w / cols
    color = (0,255,255)
    thickness = 2
    fontScale=1
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    i=0
    for x in np.linspace(start=0, stop=w-dx, num=cols):
        j=0
        for y in np.linspace(start=0, stop=h-dy, num=rows):
            # print(int(x),int(y))
            # print(j*cols+i)
            prob=float(tile_probs[j*cols+i])
            label=float(tile_labels[j*cols+i])
            prob=round(float(prob),4)
            if prob>0.5:
                if label>0:
                    cv2.putText(img, str(prob), org=(int(x)+3,int(y)+30),fontFace=font, fontScale = fontScale, color=(0,255,0), thickness=thickness)
                else:
                    cv2.putText(img, str(prob), org=(int(x)+3,int(y)+30),fontFace=font, fontScale = fontScale, color=(0,0,255), thickness=thickness)
            else:
                if label>0:
                    cv2.putText(img, str(prob), org=(int(x)+3,int(y)+30),fontFace=font, fontScale = fontScale, color=(255,0,0), thickness=thickness)
                else:
                    cv2.putText(img, str(prob), org=(int(x)+3,int(y)+30),fontFace=font, fontScale = fontScale, color=(0,0,0), thickness=thickness)
            j+=1
        i+=1

    if os.path.exists(label_path):
        poly_contour = xml_to_contour(label_path)
        if (len(poly_contour)==0):
            poly_contour = xml_to_bbox(label_path)

        if poly_contour is not None:
            cv2.polylines(img, poly_contour, True, (255,0,255),2)
    return img
        
df = pd.read_csv(
        '/home/natalia/pytorch-lightning-smoke-detection/lightning_logs/train_contour/version_3/image_preds.csv',
        names = ['image_name', 'image_pred', 'image_prob', 'image_loss', 'tile_probs', 'tile_pred', 'tile_labels'])

path = './data/figlib/raw_images/'
label_path = './data/figlib/labels/'

height = 480
width = 640
out = cv2.VideoWriter('output_video_figlib.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))
n = 60000

for i,image_name in enumerate(df['image_name']):

    tile_probs=df['tile_probs'][i].split('tensor(')[1].split(']')[0].split('[')[2].replace("\n", "").split(',')
    tile_labels = df['tile_labels'][i].split('tensor(')[1].split(']')[0].split('[')[2].replace("\n", "").split(',')
    print(tile_labels)
    img = draw_image(path+image_name+'.jpg', label_path+image_name+'.jpg.xml', tile_probs=tile_probs, tile_labels=tile_labels)
    img = cv2.resize(img, (width, height))
    out.write(img)
    cv2.imwrite('out.jpg',img)
    if i == n:
        break

out.release()