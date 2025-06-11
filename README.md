<p align="center">
  <img src="assets/logo.png" alt="App Logo" width="200"/>
</p>

# DiaCausal: A Causal Diabetes Decision Support System

## 📋 Project Description

Non-communicable diseases (NCDs) such as cardiovascular conditions, cancer, and diabetes are the leading causes of global mortality, highlighting the need for effective prevention and management strategies. While Artificial Intelligence (AI) holds great promise in healthcare for tasks like diagnosis and prediction, its lack of transparency and causal understanding limits its utility in clinical decision-making. 

Integrating causal inference into AI models addresses this gap by enabling interpretable, intervention-driven insights that support more informed and effective healthcare outcomes.

This Streamlit application serves as an **interactive decision support system**, developed based on the research presented in:

> Zahoor, S., Constantinou, A. C., Curtis, T. M., & Hasanuzzaman, M. (2024b). Investigating the validity of structure learning
> algorithms in identifying risk factors for intervention in patients with diabetes. *arXiv (Cornell University)*.
> https://doi.org/10.48550/arxiv.2403.14327

### 🔍 Purpose

This tool is designed to provide doctors, researchers, and medical students with the ability to:

- Explore relationships between various risk factors and diabetes using the **BRFSS 2015** dataset.
- Define and evaluate **causal graphical models** using domain expertise.
- Compare **expert knowledge** with causal structures learned from data via various algorithms.
- Simulate **intervention effects** on key risk factors.
- Analyze the **sensitivity** of diabetes risk to different causal factors.
- Foster a dialogue between **expert-driven knowledge** and **data-driven insights** for improved decision-making.

---

## 📁 Repository Structure

📦 root/

├── 01_About.py # Entry point for Streamlit app

├── [Other Streamlit Modules and Pages]

├── app_demo_video/ # Contains a demo video of the app in action

├── knowledge_graphs_ref_paper/ # Expert graphs in CSV (from referenced paper)

├── report/ # Detailed methodology, results, app discussion

├── requirements.txt # Python dependencies

---

## 📊 Data and Expert Knowledge Sources

- Expert knowledge graphs and datasets are sourced from:
  > Constantinou, A. (2019). *The Bayesys User Manual*.  
  > http://bayesianai.eecs.qmul.ac.uk/bayesys/

---

## 🚀 Running the App Locally (Recommended)

Due to resource constraints on the free tier of Streamlit Cloud (especially with caching and computational needs on the "Model Fitness" page), we **highly recommend running the app locally**.

### 🔧 Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/hase3b/DiaCausal.git
   cd DiaCausal
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
  
   ```bash
   streamlit run 01_About.py
   ```

---

## 📽️ App Demo

A video walkthrough of the application is available in the app_demo_video/ directory.

---

## 📚 Report

The report contains a comprehensive breakdown of the:

* Methodology
* Causal inference algorithms used
* Experimental results
* Discussion of findings
* App interface description

---

## 🤝 Authors and Acknowledgments

This application was developed as part of the term project for:

> CSE655: Probabilistic Reasoning – Spring 2025
> Institute of Business Administration (IBA), Karachi
> Instructor: Dr. Sajjad Haider

**Team Members:**

* Abdul Haseeb
* Annayah Usman

---

## 🩺 Disclaimer
This tool is intended for research and educational purposes only and should not be used for actual medical decision-making without consultation from qualified healthcare professionals.
