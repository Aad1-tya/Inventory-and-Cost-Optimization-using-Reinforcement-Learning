# 📦 RL Inventory Optimization Agent

This is a Streamlit-based web app that trains a **Deep Q-Network (DQN)** agent to manage inventory using historical demand data. The agent learns to optimize ordering decisions to **minimize holding, ordering, and stockout costs**.

---

## 🚀 Features

- 📊 **Loads real sales data** (`sales_train_validation.csv`)
- 🧠 **Trains a DQN agent** on historical demand
- 📉 **Visualizes training rewards**
- 🔍 **Compares learned policy vs baseline**
- 📦 **Plots inventory dynamics** across days

---

## 🧪 Demo Screenshot

![Demo Screenshot](demo.png) <!-- optional, if you add a screenshot -->

---

## 📁 File Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit application file |
| `sales_train_validation.csv` | Demand data for training |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation |

---

## ⚙️ Requirements

Make sure you have Python 3.8+ installed.

Install required packages:

```bash
pip install -r requirements.txt

streamlit run app.py

---

## 📦 `requirements.txt`

Here's what you'll need based on your `app.py`:

```txt
streamlit
numpy
pandas
torch
matplotlib
scipy
tqdm
