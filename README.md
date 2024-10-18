# EEG HealthCare Sector Project

## Project Description
The EEG Care Sector Project aims to utilize EEG (electroencephalogram) signals for multiple applications, including emotional state classification, epilepsy diagnosis, and activity tracking. By analyzing brainwave patterns, the project seeks to develop robust machine learning models that leverage Random Forest for epilepsy diagnosis and Gated Recurrent Units (GRU) for emotion analysis. This multi-faceted approach enhances our understanding of human emotional responses and contributes to advancements in mental health assessments and therapies.

## Features
- Preprocessing of EEG data for improved model training.
- Training and evaluation of machine learning models for emotion classification and epilepsy diagnosis.
- Visualization of training and validation performance through learning curves.

## Requirements
- Python 3.x
- TensorFlow
- Pandas
- Matplotlib
- Scikit-learn

## Setup Instructions

1. **Clone the Repository**
   Start by cloning the repository to your local machine:
   ```bash
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory**
   Change into the project directory:
   ```bash
   cd project-directory/backend
   ```

3. **Create a Virtual Environment**
   Set up a virtual environment to manage dependencies:
   ```bash
   python -m venv venv
   ```

4. **Activate the Virtual Environment**
   Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. **Install Required Packages**
   Install the necessary packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project Locally

1. **Prepare the Data**
   Ensure that your EEG datasets (e.g., `emotions.csv`, `mental-state.csv`) are in the appropriate directory.

2. **Preprocess the EEG Data**
   Run the preprocessing script to prepare the data for model training:
   ```bash
   python plot.py
   ```

3. **Train the Model**
   The training script will train the emotion recognition model and generate learning curves. 

4. **Evaluate the Model**
   After training, the model’s performance will be evaluated, and learning curves will be saved for analysis.

## Contributions
Contributions to improve the project are welcome! If you have suggestions for enhancements, new features, or optimizations, feel free to open an issue or submit a pull request.

### Contributors
- [Your Name](https://github.com/your-github-username)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [TensorFlow](https://www.tensorflow.org/) for machine learning and deep learning framework.
- [Pandas](https://pandas.pydata.org/) for data manipulation and analysis.
- [Matplotlib](https://matplotlib.org/) for data visualization.
