DS-Generator: Deep Learning Architecture for Generating Next Data Sequence

This repository contains the implementation of DS-Generator, a novel deep learning architecture designed to generate the next data sequence using its previous sequences. Our method is demonstrated on eight PROMISE repository projects and aims to reduce software testing and development costs by predicting future sequences of bug counts and various software metrics.

1)  Key Features
- **Data Augmentation Phase**: Enhances the input data for better model performance.
- **Next Version Prediction Phase**: Generates successive sequence data with high accuracy.
- **High Performance**: Achieves over 60% accuracy on 5 out of 8 datasets, significantly outperforming baseline methods.

2)  Directory Structure

- `dataset/`: Contains all versions of the PROMISE dataset used in the experiments.
- `sample_predictions/`: Contains sample predictions for the "ant" dataset.
- `scripts/`: Python scripts for data preprocessing, model definition, and testing.
  - `get_data.py`: Generates sequence data for the next version.
  - `model.py`: Contains the deep learning model information.
  - `new_model_test.py`: Tests the model.
- `config/`: Configuration files containing model hyperparameters.
  - `LSTM.txt`: Contains information about the model hyperparameters.

3) Prerequisites

- Python 3.x
- Required Python packages listed in `requirements.txt`

4)  Installation

1. Clone the repository:
    ```bash
    git clone [repository_url]
    cd Replication_Package
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

Usage
a)  Data Preprocessing

1. Run the `get_data.py` script to generate sequence data for the next version:
    ```bash
    python scripts/get_data.py --input_dir dataset/ --output_dir processed_data/
    ```
b) Model Training

2. Train the model using the `model.py` script:
    ```bash
    python scripts/model.py --config config/LSTM.txt --data_dir processed_data/
    ```

c) Model Testing

3. Test the model using the file new_model_test.py
