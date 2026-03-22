# Machine Learning-Based Fingerprint Dermatoglyphic Analysis for Non-Invasive Human Blood Group Prediction

This project presents a non-invasive approach to predict human blood groups based on fingerprint dermatoglyphic analysis. It uses a Deep Learning Ensemble Model combining the architectures of **AlexNet**, **LeNet**, and **ResNet50** to achieve high accuracy in blood group classification from fingerprint images.

## Features
- **Ensemble Deep Learning Model**: Integrates AlexNet, LeNet, and a pre-trained ResNet50 to accurately classify blood groups into 8 classes: `A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, and `O-`.
- **High Accuracy**: Reaches over 90% validation accuracy and is optimized with categorical cross-entropy and the Adam optimizer.
- **Web Application**: Provides an easy-to-use web interface built with Flask for users to upload fingerprint images and get real-time blood group predictions.
- **Performance Metrics**: The project includes detailed evaluation metrics including confusion matrices, precision, recall, f1-scores, and training/validation loss and accuracy graphs.

## Project Structure
- `code/ensemble_model/Ensemble_Model.ipynb`: A comprehensive Jupyter Notebook containing the code for data preprocessing, building the ensemble models, model training, and evaluation.
- `code/ensemble_model/app.py`: A Flask web application script that serves the frontend, loads the trained `ensemble_best_model.h5` model, and processes image uploads to predict the blood group.
- `code/ensemble_model/templates/` & `static/`: Contains the HTML and CSS for the web application UI.
- `Dataset_Blood_Group/`: Contains the training, validation, and testing dataset of fingerprint images grouped by blood type.
- `performance metrix/`: Directory containing screenshots of the training graphs and confusion matrix.
- `Requirements.txt`: List of all Python dependencies required to run the project.

## Installation & Setup

1. **Clone the repository** (if applicable) or navigate to the project directory.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r Requirements.txt
   ```

## Usage

### 1. Training the Model (Optional)
If you wish to retrain the model on the fingerprint datasets, you can run the Jupyter Notebook:
- Make sure the dataset is located at the correct path specified in the notebook (`../../dataset_new`).
- Open `code/ensemble_model/Ensemble_Model.ipynb` and run the cells. The best model will be saved as `ensemble_best_model.h5`.

### 2. Running the Web Application
To run the prediction web interface, navigate to the `code/ensemble_model/` directory and execute the Flask app:
```bash
cd code/ensemble_model
python app.py
```
- Open your browser and go to `http://127.0.0.1:5000/`.
- Upload a fingerprint image through the interface to see the predicted blood group and its confidence score.

## Technologies Used
- **Deep Learning Framework**: TensorFlow / Keras
- **Web Framework**: Flask
- **Image Processing**: Pillow (PIL), OpenCV
- **Data Manipulation & Visualization**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn

## Evaluation Results
The ensemble model combined the spatial hierarchies learned by ResNet50, the mid-level features from AlexNet, and basic structural features from LeNet. It achieved an outstanding test accuracy of **~91.33%** with well-balanced precision and recall across all blood group classes.
