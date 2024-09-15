
# AI Wizards: Machine Learning Pipeline for Image Processing and Entity Prediction

Welcome to the **AI Wizards** project! This repository contains an end-to-end machine learning pipeline designed to process images, extract text using Optical Character Recognition (OCR), and predict key entity values such as dimensions, weights, and electrical values. 

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Tools & Technologies](#tools--technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to handle a real-world challenge of extracting and predicting important product attributes such as width, depth, voltage, weight, and more from images. We utilize image links, apply OCR to extract meaningful text, and feed the extracted data into a machine learning model for entity prediction.

The pipeline is designed to be scalable, leveraging multi-processing techniques to download and process images, while incorporating both OCR and machine learning components.

## Dataset
We worked with the following datasets:
- **train.csv**: Contains columns for `image_link`, `group_id`, `entity_name`, and `entity_value`.
- **test.csv**: Includes `index`, `image_link`, `group_id`, and `entity_name`.
- **sample_test.csv**: A small test dataset with similar columns as the `test.csv`.
- **sample_test_out.csv**: Contains predictions with entity values and their units.
- **sample_test_out_fail.csv**: Contains predictions with entity values and units that could not be processed accurately.

### Entity Types:
- `entity_name`: width, depth, maximum weight recommendation, voltage, wattage, item weight.
- `entity_value`: Values such as foot, kilogram, volt, kilowatt, etc.

## Key Features
1. **Image Downloading**: Images are downloaded from URLs and saved for further processing.
2. **OCR**: Applied Optical Character Recognition (OCR) using **Pytesseract** to extract text from product images.
3. **Preprocessing**: Cleaned and standardized entity values (handling units like kg, cm, ft, etc.).
4. **ML Model**: A machine learning model trained to predict entity values based on extracted text and additional features.
5. **Error Handling**: Placeholder images are created for invalid or missing image URLs to ensure pipeline continuity.

## Tools & Technologies
- **Python**: Core programming language.
- **Pandas**: For data manipulation and processing.
- **Pytesseract**: For text extraction from images using OCR.
- **OpenCV**: Image preprocessing for better OCR accuracy.
- **Scikit-learn**: Machine learning model training and evaluation.
- **Matplotlib**: For visualizing and debugging image downloads and OCR results.
- **TQDM**: For progress visualization during image downloading.

## Installation

### Clone the Repository:
```bash
git clone https://github.com/YourUsername/AIWizards-MLPipeline.git
cd AIWizards-MLPipeline
```

### Install Dependencies:
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

### Set Up OCR:
Ensure **Tesseract OCR** is installed on your machine. Instructions for installation can be found [here](https://github.com/tesseract-ocr/tesseract).

### Download Datasets:
Make sure your dataset files (`train.csv`, `test.csv`, etc.) are placed in the appropriate folder within the project directory.

## Usage

### Step 1: Image Downloading
The `download_images.py` script will download all product images based on the links provided in the dataset.

```bash
python download_images.py
```

### Step 2: OCR Processing
Apply Optical Character Recognition (OCR) on the downloaded images to extract text.

```bash
python ocr_processing.py
```

### Step 3: Preprocessing and Model Training
Run the notebook to preprocess the entity values, train the machine learning model, and evaluate its performance.

### Step 4: Generate Predictions
Generate predictions for the test dataset using the trained model.

```bash
python generate_predictions.py
```

## Results
- **OCR Accuracy**: Extracted text from images with reasonable accuracy, though some adjustments were needed for edge cases.
- **Model Performance**: The machine learning model was able to predict the entity values with notable accuracy across most entity types.

### Visualizations:
Example images downloaded, along with extracted text, can be visualized using the provided `visualize_images.py` script.

## Contributing
We welcome contributions to improve the performance of OCR or the machine learning model. Please feel free to submit a pull request or open an issue to discuss improvements or report bugs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

