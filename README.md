# AI-Based Traffic Severity Prediction

This is a complete, modern, and responsive web application built with Python (Flask) and Machine Learning (Decision Tree Classifier) to predict the severity of traffic accidents based on weather and road features. 

## Design Theme
It features a **Neo-Light Glassmorphism** design using pure CSS for a stunning, modern look that's fully responsive across desktop, tablet, and mobile.

## Project Structure
```text
C:\bvvs project\
 ┣ 📂 dataset
 ┃ ┗ 📜 US_Accidents_March23.csv (Required here!)
 ┣ 📂 model
 ┃ ┣ 📜 model.pkl (Generated after training)
 ┃ ┗ 📜 features.pkl (Generated after training)
 ┣ 📂 static
 ┃ ┣ 📂 css
 ┃ ┃ ┗ 📜 style.css
 ┃ ┗ 📂 js
 ┃   ┗ 📜 script.js
 ┣ 📂 templates
 ┃ ┣ 📜 about.html
 ┃ ┣ 📜 base.html
 ┃ ┣ 📜 dashboard.html
 ┃ ┣ 📜 index.html
 ┃ ┗ 📜 predict.html
 ┣ 📜 app.py
 ┣ 📜 train.py
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

## How to Run the Project

### 1. Install Dependencies
Open your command prompt or terminal inside the `C:\bvvs project` folder and install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (First Time Only)
To ensure the model performs accurately and learns from your dataset, run the training script. This script processes up to 20000 rows automatically, selects the best features, and exports the `.pkl` files.
```bash
python train.py
```
*Note: This might take a minute depending on your computer's speed.*

### 3. Run the Flask Web Server
Start the backend application:
```bash
python app.py
```

### 4. Open in Browser
Visit the following URL in your web browser:
**http://127.0.0.1:5000**

Enjoy your beautiful, fully-functional Machine Learning web application!
