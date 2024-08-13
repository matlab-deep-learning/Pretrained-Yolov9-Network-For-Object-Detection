# Pretrained-YOLOv9-Network-For-Object-Detection

This repository provides multiple pretrained YOLO v9[1] object detection networks for MATLAB®, trained on the COCO 2017[2] dataset. These object detectors can detect 80 different object categories including [person, car, traffic light, etc](/src/%2Bhelper/getCOCOClasess.m).
The input should be a single image.

**Creator**: MathWorks Development

**Includes Codegen support**: ✔

**Includes Simulink support script**: ✔

**Includes Import support script**: ✔

**Includes transfer learning script**: ❌  

Refer to [Pretrained YOLOv8 Network For Object Detection](https://github.com/matlab-deep-learning/Pretrained-YOLOv8-Network-For-Object-Detection) (or) [trainYOLOXObjectDetector](https://in.mathworks.com/help/vision/ref/trainyoloxobjectdetector.html) for training latest YOLOs on custom dataset.


## License
The software and model weights are released under the [GNU Affero General Public License v3.0](https://github.com/ultralytics/ultralytics?tab=AGPL-3.0-1-ov-file#readme). For alternative licensing, contact [Ultralytics Licensing](https://www.ultralytics.com/license).

## Requirements
- MATLAB® R2024a or later
- Computer Vision Toolbox™
- Deep Learning Toolbox™
- Deep Learning Toolbox Converter for ONNX Model Format
- (optional) MATLAB® Coder for code generation
- (optional) GPU Coder for code generation

## Getting Started
Download or clone this repository to your machine and open it in MATLAB®.

### Setup
Add path to the cloned directory.

```matlab
addpath(genpath(pwd));
```

### Download the pretrained network
Use the code below to download the pretrained network.

```matlab
% Load YOLO v9 medium model
modelName = 'Yolov9m';
model = helper.downloadPretrainedYOLOv9(modelName);
net = model.net;
```

modelName of the pretrained YOLO v9 deep learning model, specified as one of these:
- Yolov9t
- Yolov9s
- Yolov9m
- Yolov9c
- Yolov9e

Following is the description of various YOLO v9 models available in this repo:

| Model         |                                      Description                                                                                                                   |
|-------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Yolov9t       |   Tiny pretrained YOLO v9 model optimized for speed and efficiency.                                                                                                |
| Yolov9s       |   Small pretrained YOLO v9 model balances speed and accuracy, suitable for applications requiring real-time performance with good detection quality.               |
| Yolov9m       |   Medium pretrained YOLO v9 model offers higher accuracy with moderate computational demands.                                                                      |
| Yolov9c       |   Compact pretrained YOLO v9 model prioritizes maximum detection accuracy for high-end systems, at the cost of computational intensity.                              |
| Yolov9e       |   Extensive YOLOv9 model is the most accurate but requires significant computational resources, ideal for high-end systems prioritizing detection performance.   |

### Detect Objects Using Pretrained YOLO v9
To perform object detection on an example image using the pretrained model, you can execute the `runInference.m` script. Alternatively, utilize the code provided below for the same purpose.

```matlab
% Read test image.
I = imread(fullfile('data/Input','inputTeam.jpg'));

% Load pretrained medium variant of YOLOv9 object detector.
det = yolov9ObjectDetector('Yolov9m');

% Perform detection using pretrained model.
[bboxes, scores, labelIds] = detect(det, I);

% Visualize detection results.
helper.plotObjectDetections(I,bboxes,scores,labelIds);
```
![Results](/data/Output/inputTeamResults.jpeg)

## Metrics and Evaluation

### Size and Accuracy Metrics

| Model         | Input image resolution | Size (MB) | mAP  |
|-------------- |:----------------------:|:---------:|:----:|
| Yolov9t       |       640 x 640        |  7.5      | 38.3 |
| Yolov9s       |       640 x 640        |  25       | 46.8 |
| Yolov9m       |       640 x 640        |  67.2     | 51.4 |
| Yolov9c       |       640 x 640        |  85       | 53.0 |
| Yolov9e       |       640 x 640        |  190      | 55.6 |

mAP for models trained on the COCO dataset is computed as average over IoU of .5:.95.

## Deployment
Code generation enables you to generate code and deploy YOLO v9 on multiple embedded platforms. The list of supported platforms is shown below:

| Target                             |  Support  |   Notes                     |
|------------------------------------|:---------:|:---------------------------:|
| GPU Coder                          |     ✔     |    run `gpuCodegenYOLOv9.m` |
| MATLAB Coder                       |     ✔     |    run `codegenYOLOv9.m`    |

To deploy YOLO v9 to GPU Coder, run `gpuCodegenYOLOv9.m`. This script calls the `yolov9Predict.m` entry point function and generate CUDA code for it. It will run the generated MEX and give an output.
For more information about codegen, see [Deep Learning with GPU Coder](https://in.mathworks.com/help/gpucoder/gpucoder-deep-learning.html).

## Simulink
Simulink is a block diagram environment used to design systems with multidomain models, simulate before moving to hardware, and deploy without writing code. For more information about Simulink, see [Get Started with Simulink](https://in.mathworks.com/help/simulink/getting-started-with-simulink.html)

```matlab
% Read test image.
im = imread(fullfile('data/Input','inputTeam.jpg'));

% Open Simulink model.
open('yolov9SimulinkSupport.slx')
```
To run the simulation, click `Run` from the `Simulation` tab.

The output will be logged to the workspace variable `out` from the Simulink model.

## Network Overview
YOLO v9 is one of the best performing object detectors and is considered as an improvement to the existing YOLO variants such as YOLO v5, YOLOX and YOLO v8.

Following are the key features of the YOLO v9 object detector compared to its predecessors:
- Improved Accuracy: YOLO v9 is expected to offer enhanced accuracy in object detection compared to its previous versions. This improvement can lead to more precise and reliable detection results.
- Better Speed and Efficiency: YOLO v9 may have optimizations that allow it to achieve faster processing speeds while maintaining high accuracy. This can be crucial for real-time applications or scenarios with limited computational resources.
- Enhanced Object Classification: YOLO v9 may introduce improvements in object classification capabilities, allowing for more accurate and detailed classification of detected objects. 


## References
[1] https://github.com/ultralytics/ultralytics

[2] Lin, T., et al. "Microsoft COCO: Common objects in context. arXiv 2014." arXiv preprint arXiv:1405.0312 (2014).


Copyright 2024 The MathWorks, Inc.


