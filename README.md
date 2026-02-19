# the404society
Training an AI model with synthetic data to perform semantic segmentation for off-road autonomous navigation in desert environments.
Project Overview

This project implements a semantic segmentation model using DeepLabV3-ResNet50 to classify offroad terrain and environmental objects into 10 categories.
The objective was to maximize the Mean Intersection over Union (IoU) score while ensuring reproducibility, clarity, and efficient optimization.
Final Achieved IoU: 0.561

Model Architecture
 Backbone: ResNet-50 (Pretrained)
 Segmentation Head: DeepLabV3
 Modified final classifier layer for 10 classes

python model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)


Project Overview

This project implements a semantic segmentation model using **DeepLabV3-ResNet50** to classify offroad terrain and environmental objects into 10 categories.
The objective was to maximize the **Mean Intersection over Union (IoU)** score while ensuring reproducibility, clarity, and efficient optimization.
"DEFAULT")
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)


 Training Configuration
Parameter	Value
Optimizer	Adam
Learning Rate	1e-4
Epochs	40
Loss Function	CrossEntropyLoss
Input Resolution	256x256
Device	CUDA (RTX 4050)
Batch Size	4
 Key Improvements
Initial Problem
Model plateaued at IoU ≈ 0.50.
Optimization Strategy
Introduced data augmentation:
Random Horizontal Flip
Random Rotation (±10°)

Result
IoU improved to 0.561.

Performance Metrics
Epoch	Mean IoU
10	0.550
20	0.556
30	0.563
40	0.561

Training loss showed stable convergence without instability.

Project Structure
Offroad-Semantic-Segmentation/
│
├── train_segmentation.py
├── model.pth
├── requirements.txt
├── generate_report_graphs.py
├── IoU_Performance_Graph.png
├── Loss_Graph.png
└── dataset/
    ├── images/
    └── masks/
 Installation
1. Create Environment
 conda create -n EDU python=3.10
 conda activate EDU
2. Install Dependencies
 pip install -r requirements.txt

If using GPU:
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
 

Training
Run:
python train_segmentation.py
This will:
Train the model for 40 epochs
Print IoU per epoch
Save trained weights as model.pth


 Generating Performance Graphs
Run:
python generate_report_graphs.py
This creates:
IoU_Performance_Graph.png
Loss_Graph.png


 Reproducibility
To reproduce results:
Clone repository
Install dependencies
Place dataset inside dataset/
Run training script
All hyperparameters are defined at the top of train_segmentation.py.


 Challenges Faced
Training plateau at 0.50 IoU
Slow CPU training
Augmentation integration errors
CUDA installation issues
All resolved through systematic debugging and optimization.


Future Improvements
Dice + CrossEntropy combined loss
Higher input resolution (320x320)
Class-balanced loss
Advanced augmentation techniques
Confusion matrix evaluation


Conclusion
This project demonstrates a complete semantic segmentation pipeline including:
Model customization
Data preprocessing
Optimization
GPU acceleration
Performance evaluation
Professional documentation
Final IoU achieved: 0.561
