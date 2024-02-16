Image_colorization

Description

This project utilizes a deep learning model to colorize grayscale images automatically. It employs a Convolutional Neural Network (CNN) trained to understand the nuances of color and apply this knowledge to transform grayscale images into colored ones. The repository includes a pre-trained model, scripts for capturing grayscale images and colorizing them, and a custom data loader for model training.

Getting Started

Dependencies
Ensure you have Python 3.x installed on your system.
The project requires PyTorch, Torchvision, NumPy, and OpenCV-python. Install these packages using pip:
bash
Copy code
pip install torch torchvision numpy opencv-python
Installation
Clone the repository to your local machine:
bash
Copy code
git clone [your-repo-link]
cd [repo-directory]
Download the pre-trained model final_pretrained_model_30.pth and place it in the project directory.
Files
camera.py: Captures a photo in grayscale to be colorized.
dataloader.py: Prepares and loads the dataset for training.
main.py: The main script for training the model or colorizing images.
final_pretrained_model_30.pth: The pre-trained model file.
image_colorization_project.pdf: Contains detailed documentation about the project.
Usage

Colorizing an Image
To colorize a grayscale image using the pre-trained model:

Capture a grayscale image:
bash
Copy code
python camera.py
Colorize the captured image:
bash
Copy code
python main.py --mode colorize
Training the Model
To train the model from scratch:

bash
Copy code
python main.py --mode train
