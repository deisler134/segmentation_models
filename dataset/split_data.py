'''
Created on Apr. 23, 2019

    build data to train val and test.
    note: image and mask have same name in diferent directory
    
@author: deisler
'''

import os
import glob

root = '/media/'
image_path = 'image/'
mask_path = 'mask/'

train_path = 'u-train/train/'
val_path = 'u-train/val/'
test_path = 'u-train/test/'

def split_data(root, image_path, mask_path, train_path, val_path, test_path):
    
    train_path = os.path.join(root, train_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        
    val_path = os.path.join(root, val_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        
    test_path = os.path.join(root, test_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
        
    imagelist = glob.glob(os.path.join(root,image_path, '*'))
    
    trainlist = imagelist[:int((len(imagelist)-100)*0.85)]
    vallist = imagelist[int((len(imagelist)-100)*0.85):-100]  
    testlist = imagelist[-100:]
    
    data_copy(trainlist, root + mask_path, train_path + 'image/', train_path + 'mask/')
    data_copy(vallist, root + mask_path, val_path + 'image/', val_path + 'mask/')
    data_copy(testlist, root + mask_path, test_path + 'image/', test_path + 'mask/')
              


def data_copy(imagelist, maskpath, target_imgpath, target_maskpath):
    if not os.path.exists(target_imgpath):
        os.makedirs(target_imgpath)
    if not os.path.exists(target_maskpath):
        os.makedirs(target_maskpath)
        
    for image in imagelist:
        imgname = image.split('/')[-1]
        mask_path = maskpath + imgname
        os.system('cp {} {}'.format(image, target_imgpath))
        os.system('cp {} {}'.format(mask_path, target_maskpath))
            
if __name__ == '__main__':
    split_data(root, image_path, mask_path, train_path, val_path, test_path)

        
    
