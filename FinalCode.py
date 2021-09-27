#import patchify
import glob
import matplotlib.image as mpimg
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU




original = glob.glob('C:/SelfDrivingMaterials/Data/train_images/train/*.png')
mask = glob.glob('C:/SelfDrivingMaterials/Data/train_masks/train/*.png')



print(len(original))
print(len(mask))

'''
img_num = random.randint(0, 23799)


img_for_plot = mpimg.imread(original[img_num])
mask_for_plot = mpimg.imread(mask[img_num])



plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot)
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask')
plt.show()


patch_num_sem = 1
patch_num = 1

'''



seed=24
batch_size= 16
num_class = 34
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


scaler = MinMaxScaler()


def preprocess_data(img, mask, num_class):
    
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)
    mask = to_categorical(mask, num_class)
    
      
    return (img,mask)

def trainGenerator(train_img_path, train_mask_path, num_class):
    
    img_data_gen_args = dict(horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')
    
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)



train_img_path = "C:/SelfDrivingMaterials/Data/train_images/"
train_mask_path = "C:/SelfDrivingMaterials/Data/train_masks/"




train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=34)

x, y = train_img_gen.__next__()

print(x.shape)
print(y.shape)



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


def multi_unet_model(n_classes=34, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
   
    s = inputs

    #encoder
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #decoder
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    

    
    return model


import numpy as np
from tensorflow.keras.utils import normalize
from matplotlib import pyplot as plt



#model = multi_unet_model(34,256,256,3)
model = sm.Unet(BACKBONE,encoder_weights='imagenet',input_shape=(256, 256, 3),
                classes=34, activation='softmax')

model.compile(optimizer='adam',loss=sm.losses.categorical_focal_jaccard_loss, metrics=[sm.metrics.iou_score])
model.summary()

num_train_imgs = len(glob.glob('C:/SelfDrivingMaterials/Data/train_images/train/*.png'))
steps_per_epoch = num_train_imgs //batch_size
print(num_train_imgs)
print(steps_per_epoch)


history = model.fit(train_img_gen, 
                    steps_per_epoch=steps_per_epoch, epochs=50)


model.save('C:/SelfDrivingmaterials/sem_seg_50epochs.hdf5')




val_img_path = "C:/SelfDrivingMaterials/Data/val_images/"
val_mask_path = "C:/SelfDrivingMaterials/Data/val_masks/"
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=34)



from tensorflow.keras.models import load_model

model = load_model("C:/SelfDrivingmaterials/sem_seg_50epochs.hdf5", compile=False)


test_image_batch, test_mask_batch = val_img_gen.__next__()


test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3) 
test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)

n_classes = 34
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


img_num = random.randint(0, test_image_batch.shape[0]-1)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_image_batch[img_num])
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_batch_argmax[img_num])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_pred_batch_argmax[img_num])
plt.show()



'''
def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    for i in range(0, image.shape[0], 256):   #Steps of 256
        for j in range(0, image.shape[1], 256):  #Steps of 256
            #print(i, j)
            single_patch = image[i:i+patch_size, j:j+patch_size]
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_shape = single_patch_norm.shape[:2]
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
          
            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img

##########
#Load model and predict
#model = get_model()

model.load_weights('mitochondria_50_plus_100_epochs.hdf5')

#Large image
large_image = cv2.imread('data/01-1.tif', 0)
segmented_image = prediction(model, large_image, patch_size)
plt.hist(segmented_image.flatten())  #Threshold everything above 0

plt.imsave('data/results/segm.jpg', segmented_image, cmap='gray')

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(segmented_image, cmap='gray')
plt.show()

##################################
#Watershed to convert semantic to instance
#########################

'''
'''


from skimage import measure, color, io

#Watershed
img = cv2.imread('data/results/segm.jpg')  #Read as color (3 channels)
img_grey = img[:,:,0]

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(img_grey,cv2.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv2.dilate(opening,kernel,iterations=10)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

ret2, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret3, markers = cv2.connectedComponents(sure_fg)
markers = markers+10

markers[unknown==255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

cv2.imshow('Overlay on original image', large_image)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)


props = measure.regionprops_table(markers, intensity_image=img_grey, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity'])
    
import pandas as pd
df = pd.DataFrame(props)
df = df[df.mean_intensity > 100]  #Remove background or other regions that may be counted as objects
   
print(df.head())


'''