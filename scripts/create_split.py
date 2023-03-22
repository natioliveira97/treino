import os

root = 'data/dataset_artigo/raw_images'
labels= 'data/dataset_artigo/drive_clone_numpy_new/'
f1 = open('/home/natalia/pytorch-lightning-smoke-detection/data/dataset_artigo/train_files.txt','w')
f2 = open('/home/natalia/pytorch-lightning-smoke-detection/data/dataset_artigo/val_files.txt','w')
f3 = open('/home/natalia/pytorch-lightning-smoke-detection/data/dataset_artigo/test_files.txt','w')

b = len(os.listdir(root))
print(int(0.5*b))
i=0

for path, subdirs, files in os.walk(root):
    # print(i)
    for name in sorted(files):
        label_filename = labels+path.split('/')[-1]+'/'+name.split('.jpg')[0]+'.npy'
            f1.write(os.path.join(path,name))
            f1.write('\n')
        elif i<int(0.75*b):
            f2.write(os.path.join(path,name))
            f2.write('\n')
        else:
            f3.write(os.path.join(path,name))
            f3.write('\n')
    i+=1


f1.close()
f2.close()
f3.close()