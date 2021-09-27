
import glob
import matplotlib.image as mpimg
import cv2
from patchify import patchify
import numpy as np

original = glob.glob('C:/Users/phant/Desktop/To_Pradyuman_Sriniketan/leftImg8bit/train/**/*.png')
mask = glob.glob('C:/Users/phant/Desktop/To_Pradyuman_Sriniketan/gtFine/train/**/*.png')

print(len(original))
print(len(mask))
'''
image_dataset = []

for i in range(0,10000):
    single_patch = mpimg.imread(original[i])
    print(single_patch.shape)
    image_dataset.append(single_patch)
    #print(image_dataset.shape)

image_dataset = np.array(image_dataset)
print(image_dataset.shape)

'''

patch_num_sem = 1
patch_num = 1

image_color = mpimg.imread(original[0])


print(image_color.shape)

image_patch = patchify(image_color,(512,512,3),step=512)

print(image_patch.shape)
new_image_patch = image_patch.reshape(2,4,512,512,3)

for i in range(image_patch.shape[0]):
    for j in range(image_patch.shape[1]):
        
        single_patch = new_image_patch[i,j,:,:]
        print(single_patch.shape)
        
        #cv2.imshow('image_patch',single_patch)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows
           
        
        cv2.imwrite('C:/patches/'+'image_original_'+str(i)+str(j)+'.png', single_patch*255)
        
print(single_patch)






image_color_sem = mpimg.imread(mask[0])
print(image_color_sem.shape)

image_patch_sem = patchify(image_color_sem,(256,256),step=256)

print(image_patch_sem.shape)


for m in range(image_patch_sem.shape[0]):
   for n in range(image_patch_sem.shape[1]):
        single_patch_sem = image_patch_sem[m,n,:,:]
        print(single_patch_sem.shape)
        cv2.imwrite('C:/patches/sem/'+'image_mask_'+str(i)+str(j)+'.png', single_patch_sem*255)
        
        #cv2.imshow('image_patch',single_patch)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows
           
print(single_patch_sem.shape)        
         

