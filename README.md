
# Aerial Wind Turbine Detection - YOLOV5 with Slicing Aided Hyper Inference (SAHI)

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/1.jpg?raw=true)

### Why Oject Detection is difficult in small objects ...??

- These datasets generally consist of low-resolution photos (640 480 pixels) with huge objects and high pixel coverage (on average, 60 percent of the image height). 
- While the trained models perform well on those sorts of input data, they exhibit much poorer accuracy on small item detection tasks in high-resolution photos captured by high-end drones and surveillance cameras.
- Small object detection is thus a difficult task in computer vision because, in addition to the small representations of objects, the diversity of input images makes the task more difficult. 
- For example, an image can have different resolutions; if the resolution is low, the detector may have difficulty detecting small objects.


- Compared to traditional wired circuits, PCBs offer a number of advantages. Their small and lightweight design is appropriate for use in many modern devices, while their reliability and ease of maintenance suit them for integration in complex systems. 
- Additionally, their low cost of production makes them a highly cost-effective option.

## Business Scenario

- The Client is looking for an Effective Wind Turbine detection System on aerial images taken from drones.

## Challenges
- Pixel size of wind turbines are very small when comparing the size image.
- Limited resource.

## Approach
- To address the small object detection difficulty, Fatih Akyon et al. presented `Slicing Aided Hyper Inference (SAHI)`, an open-source solution that provides a generic slicing aided inference and fine-tuning process for small object recognition. 
- During the fine-tuning and inference stages, a slicing-based architecture is used.

## What is Slicing Aided Hyper Inference (SAHI)
- This method works by Splitting the input photos into overlapping slices results in smaller pixel regions when compared to the images fed into the network. 
- The proposed technique is generic in that it can be used on any existing object detector without needing to be fine-tuned. 
- The suggested strategy was tested using the models `Detectron2, MMDetection, and YOLOv5`.

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

## Data Understanding

- The available dataset have total 1266 images for Training.
- 139 images for Validation and 319 images are provided for validation and testing
- I have used Imgae augmantation techique using roboflow and created 4261 images for training by applying below augmentations

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/2.jpg?raw=true)


Let's see some sample from training data

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/3.jpg?raw=true)


## Data Labeling

- Labeling was done with `LabelImg` tool and labels are saved on `.txt` format

![alt text](https://github.com/sudheeshe/Turbine_Detection_YOLO_SAHI/blob/main/readme_imgs/4.jpg?raw=true)


## Model Building and Evaluation

- Used YoloV5 model for detection.
- YoloV5 trained from scratch to 1500 epochs on Paperspace P4000 GPU.
- YoloV5 does image augmentation internally on training images which helps in better predictions and reduce overfitting.

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/6.jpg?raw=true)

Let's visualize some of our `training image batch` and `validation image batch`

#### Training image batch with Mosaic augmentation applied

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/train_tile_batch.jpg?raw=true)

#### Validation image batch

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/val_tile_batch.jpg?raw=true)


- Let's see the `mAP` for first `100 epochs`. The mAP was `0.0086`

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/Reports/100_epoch_eport.png?raw=true)


- The mAP for `1500 epochs` reached to `0.811` at mAP@0.5 (means mAP at threshold of 0.5). 
- I've tried to run the model for another 200 epochs but the model didn't show any improvement in mAP beyond 1500 epochs.

#### Precision-Recall Curve

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/YoloV5_Training/yolov5/runs/train/yolov5s_results8/PR_curve.png?raw=true)

#### F1 Curve

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/YoloV5_Training/yolov5/runs/train/yolov5s_results8/F1_curve.png?raw=true)

- The F1 curve shows that any threshold (confidence) value between 0.2 to almost 0.6 gives better results from the model

#### Confusion Matrix

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/YoloV5_Training/yolov5/runs/train/yolov5s_results8/confusion_matrix.png?raw=true)

- The model has less prediction power on `spurious_copper` class and very high confidence on `missing_hole` and other classes have decent prediction capability.

## Prediction Images

- In order to do prediction on new images please find my another repository

[Github link](https://github.com/sudheeshe/PCB_Defect_Detection)

- The UI is used for individual predictions by uploading the image

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/UI.jpg?raw=true)

- Let's see some prediction done by the model through UI

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/pred_collage.jpg?raw=true)


## Video Demo

[click here](https://www.youtube.com/watch?v=oRXxbZ7rxrI&ab_channel=SudheeshE)


## References:
#### How to do training and inferencing 
[click here](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/How_to_run.txt)

#### YoloV5 custom training helper repo
[click here](https://github.com/sudheeshe/YoloV5_Custom_training_template)
