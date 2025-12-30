# CPE-Assessment
This project builds an end-to-end fine-grained image classification system to identify flower species using the Oxford 102 Flowers dataset. It applies transfer learning with a pre-trained model, data augmentation for robustness, and explainability techniques to ensure accurate and interpretable predictions.
git clone https://github.com/psyahmi/CPE-Assessment.git
cd CPE-Assessment

conda create -n CPE_env python=3.12
conda activate CPE_env

python -m venv CPE_env
source CPE_env/bin/activate  
CPE_env\Scripts\activate  

# Download Dataset
from torchvision.datasets import Flowers102
train_dataset = Flowers102(root="./data", split="train", download=True)

##Training/Evaluation##
# Example: Training script
python train.py

# Example: Evaluate model on validation/test set
python evaluate.py

#Run Inference on single image
from inference import predict_image

result = predict_image("path/to/your/image.jpg")
print(result)

# optional grad cam
from gradcam import GradCAM, show_cam_on_image

gradcam = GradCAM(model, target_layer)
show_cam_on_image(input_tensor, cam)
