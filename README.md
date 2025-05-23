# 🌐 Anomaly Recognition and Fault Detection in SolarSonic TrafficGuard Sensor Systems

## 📜 Mentored By
**Dr. Indu Dohare**

## 📝 Presented By
- **Shivam Kumar Gupta** (20214082)
- **Sneha Kumari** (20214102)
- **Snehashish Datta** (20214153)

---

## 📘 Broad Area
This project combines **IoT** and **Machine Learning** for real-time anomaly detection in environmental monitoring systems, focusing on the SolarSonic TrafficGuard at traffic signals to ensure accurate and reliable environmental readings.

---

## ❓ Problem Statement
The SolarSonic TrafficGuard, designed for real-time environmental monitoring, faces challenges such as:

1. **Environmental Anomalies**: Unexpected changes in temperature, humidity, or air quality.
2. **Hardware Malfunctions**: Sensor issues, calibration errors, or physical damage.
3. **Software Glitches**: Data transfer issues, connectivity problems, and potential software bugs.

---

## 📂 Project Structure

```plaintext
Anomaly-Recognition-and-Fault-Detection/
├── notebooks/                       # Jupyter notebooks for interactive analysis
│   ├── data_preprocessing.ipynb      # Data cleaning and handling of missing values
│   ├── feature_engineering.ipynb     # Feature extraction and transformation
│   ├── anomaly_detection.ipynb       # Techniques for detecting various anomalies
│   └── model_evaluation.ipynb        # Model testing and performance assessment
├── src/                              # Source code structured by functionality
│   ├── preprocessing/
│   │   └── preprocess.py             # Data preparation and cleaning scripts
│   ├── feature_engineering.py        # Code for creating new features from raw data
│   ├── anomaly_detection/
│   │   └── anomaly_detection.py      # Anomaly detection model implementations
│   └── evaluation/
│       └── evaluation.py             # Model evaluation and performance metrics
└── README.md                         # Documentation of project structure and details


🔑 Key Features
Anomaly Detection
Utilize Isolation Forest and LSTM models for identifying anomalies in sensor data.

Dynamic Thresholding
Employ weighted scoring and correlation-based behavior analysis for refined classifications.

Visualization Tools
Generate insightful graphs and plots to understand sensor performance and anomalies visually.

Real-Time Monitoring
Integrate IoT pipelines for seamless real-time data processing.

📊 Anomaly Classification Categories
Category	Description
Normal	Data is within expected limits with no anomalies detected.
Environmental Change	Sudden but valid changes caused by external environmental factors.
Fault	Sensor failures or inconsistencies indicating hardware issues.
⚙️ Technologies Used
Programming: Python
Frameworks: TensorFlow, Scikit-learn
Data Visualization: Matplotlib, Seaborn
IoT Integration: ESP8266, MQTT
Tools: Jupyter Notebooks, Git

📂 Notebooks Overview
Data Preprocessing

Handle missing values and prepare data for analysis.
Feature Engineering

Extract meaningful features like correlations and weighted scores.
Anomaly Detection

Implement algorithms for detecting anomalies in sensor data.
Model Evaluation

Test and validate models for performance accuracy.
🛠️ Setup Instructions
Clone the repository:

bash
Copy code
git clone https://github.com/Shivam20214082/Major-Project.git
cd Anomaly-Recognition-and-Fault-Detection
Install dependencies:

bash
Copy code
pip install -r src/requirements.txt
Run the main script:

bash
Copy code
python src/main.py
Access visualizations in the notebooks/plots/ directory.

🌟 Future Work
Integrate self-healing mechanisms for automatic fault correction.
Deploy the solution on edge devices for faster anomaly detection.
Expand the system for multimodal sensor networks in other domains.
🤝 Contributing
We welcome contributions! Feel free to submit pull requests or report issues in the GitHub repository.

📧 Contact Us
For inquiries, reach out to us via email:

Shivam Kumar Gupta - shivam64jnp@gmail.com
Sneha Kumari - snehakr654@gmail.com
Snehashish Datta - snehashish@example.com
