
# Aerial Wind Turbine Detection - YOLOV5 with Slicing Aided Hyper Inference (SAHI)

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/1.jpg?raw=true)

### Why Oject Detection is difficult in small objects ...??

- Normally datasets which generally consist of low-resolution photos (640 x 480 pixels) with huge objects and high pixel coverage (on average, 60 percent of the image height). 

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/19.jpg?raw=true)

- While the trained models perform well on those sorts of input data, they exhibit much poorer accuracy on small item detection tasks in high-resolution photos captured by high-end drones and surveillance cameras.
- Small object detection is thus a difficult task in computer vision because, in addition to the small representations of objects, the diversity of input images makes the task more difficult. 
- For example, an image can have different resolutions; if the resolution is low, the detector may have difficulty detecting small objects.

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/18.png?raw=true)



## Business Scenario

- The Client is looking for an Effective Wind Turbine detection System on aerial images taken from drones.

## Challenges
- Pixel size of wind turbines are very small when comparing the size image. We will see training images sample later on this repo.
- Limited resource
- Amount data is small (1266 images for training)


## Approach
- To address the small object detection difficulty, `Fatih Akyon et al.` presented `Slicing Aided Hyper Inference (SAHI)`, an open-source solution that provides a generic slicing aided inference and fine-tuning process for small object recognition problems. 
- During the fine-tuning and inference stages, a slicing-based architecture is used.

## What is Slicing Aided Hyper Inference (SAHI)
- This method works by Splitting the input photos into overlapping slices results in smaller pixel regions when compared to the images fed into the network. 
- The proposed technique is generic in that it can be used on any existing object detector without needing to be fine-tuned. 
- This strategy is currently available and tested for these models `Detectron2, MMDetection, and YOLOv5`.

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/5.gif?raw=true)

- Each image is sliced into overlapping patches of varying dimensions that are chosen within predefined ranges known as hyper-parameters. 
- Then, during fine-tuning, patches are resized while maintaining the aspect ratio, so that the image width is between 800 and 1333 pixels, resulting in augmentation images with larger relative object sizes than the original image. 
- These images, along with the original images, are utilized during fine-tuning.
- During the inference step, the slicing method is also used. 
- In this case, the original query image is cut into a number of overlapping patches. 
- The patches are then resized while the aspect ratio is kept. 
- Following that, each overlapping patch receives its own object detection forward pass. 
- To detect larger objects, an optional full-inference (FI) using the original image can be used. 
- Finally, the overlapping prediction results and, if applicable, the FI results are merged back into their original size.

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/6.gif?raw=true)



## Data Understanding

- The available dataset have total 1266 images for Training.
- 319 images are provided for validation and testing.
- I have used Image augmentation technique using roboflow and created 4261 images for training, by applying below augmentations

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/2.jpg?raw=true)


Let's see some sample from training data

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/3.jpg?raw=true)


## Data Labeling

- Labeling was done with `LabelImg` tool and labels are saved on `.txt` format

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/4.jpg?raw=true)


## Model Building and Evaluation

- Used Yolo-Small-V5 model and YoloX-V5 for detection to understand the performance on lite model(Yolo small) and heavy model(YoloX).
- Yolo-Small-V5 model trained from scratch to 700 epochs on Google Colab.
- YoloX-V5 model trained from scratch to 1100 epochs on Google Colab.
- YoloV5 does image augmentation internally on training images which helps in better predictions and reduce overfitting.

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/6.jpg?raw=true)

Let's visualize some of our `training image batch` and `validation image batch`

#### Training image batch with Mosaic augmentation applied

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/7.jpg?raw=true)
#### Validation image batch

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/8.jpg?raw=true)


- The mAP for Yolo-Small-V5 model on `700 epochs` reached to `mAP 0.89` at `threshold 0.567`. 
- I've tried to run the model for another 100 epochs but the model didn't show any improvement in mAP beyond 700 epochs.
- The mAP for YoloX-V5 model on `1100 epochs` reached to `0.91` at `threshold 0.417`. 
- I've tried to run the model for another 50 epochs but the model didn't show any improvement in mAP beyond 1100 epochs.

#### Precision-Recall Curve

- YOLO Small 

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/9.png?raw=true)

- YOLOX 

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/12.png?raw=true)

#### F1 Curve

- YOLO Small 
- 
![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/10.png?raw=true)

- The F1 curve shows that any threshold (confidence) value between 0.45 to almost 0.7 gives better results from the model

- YOLOX

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/13.png?raw=true)

- The F1 curve shows that any threshold (confidence) value between 0.4 to almost 0.5 gives better results from the model

#### Confusion Matrix
- YOLO Small 

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/11.png?raw=true)

- YOLOX

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/14.png?raw=true)


## Prediction Images

- The UI is used for individual predictions by uploading the image

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/15.jpg?raw=true)

- The UI consist of drop down option for selecting different models 

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/16.jpg?raw=true)

- For applying SAHI with different slice sizes on Yolo Small and YoloX we can use this option

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/17.jpg?raw=true)


- Let's see some prediction done by the model through UI

- Let's see the prediction on a `small Image (size 600x450)` with `standard YOLO Small and YOLOX`  

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/pred_1.jpg?raw=true)

- Let's see the prediction on a `large Image (size 1560x1038)` with `standard YOLO Small and YOLOX`

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/pred_3.jpg?raw=true)

- Prediction with `SAHI+YOLO Small` and `SAHI+YOLOX` with different slicing sizes

- Let's see the prediction on a `small Image (size 600x450)` with `SAHI+YOLO Small and SAHI+YOLOX`

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/pred_2.jpg?raw=true)

- Let's see the prediction on a `large Image (size 1560x1038)` with `SAHI+YOLO Small and SAHI+YOLOX`

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/pred_4.jpg?raw=true)

### Note: 
- If the input image size is small the slicing size should keep to smaller values eg: 64x64
- If the input image size is large the slicing size should keep to larger values eg: 512x512

## Video Demo

[click here](https://youtu.be/MTM80cvyMUA)


## References:
#### How to do training and inferencing 
[click here](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/How_to_run.txt)

#### YoloV5 custom training helper repo
[click here](https://github.com/sudheeshe/YoloV5_Custom_training_template)

#### Blogs
- https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80
- https://github.com/obss/sahi
- https://analyticsindiamag.com/small-object-detection-by-slicing-aided-hyper-inference-sahi/
- https://soorajsknair.medium.com/an-overview-of-small-object-detection-by-slicing-aided-hyper-inference-sahi-3eed8791c21
- https://www.kaggle.com/code/remekkinas/sahi-slicing-aided-hyper-inference-yv5-and-yx/notebook
