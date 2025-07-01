
# 🧠 Federated Learning for Grinch Bot Detection  
**Collaborative Defense Against Evolving Online Threats**

This project demonstrates a simulated **Federated Learning (FL)** system designed to detect **"Grinch Bots"** – automated programs that flood online stores to snatch up limited items. The core idea is to enable multiple retailers to collaboratively train a powerful bot detection model **without ever sharing their sensitive customer data**.

Retailers face a **common, evolving enemy** but fight the battle alone, often struggling to keep up.
o mimic a real-world multi-retailer scenario, this project simulates three separate retailers, each acting as an independent Federated Learning client:

Client 1: Represents Retailer A 

Client 2: Represents Retailer B 

Client 3: Represents Retailer C 

## 🧠 The Solution: Collective Learning Without Sharing Data
Federated Learning enables retailers to share what they **learn** (model updates) rather than raw customer data.


## 📚 Dataset
Web bot detection dataset includes:
- Humans
- Moderate bots (browser fingerprints)
- Advanced bots (human-like behavior)

##  Extract the dataset and place it under:  
`bot-detector/dataset

---

## 📁 Project Structure

```
bot-detector/
├── client.py                  # Client-side FL logic
├── server.py                  # Server-side FL logic
├── run.py                     # FL runner script
├── run_federated_learning.py  # Master orchestration script
├── README.md                  # Project documentation
├── venv/                      # Virtual environment (ignored by Git)
│
├── dataset/
│   ├── partition/
│   │   ├── client_1/
│   │   │   └── phase1/
│   │   │       ├── annotations/
│   │   │       └── data/
│   │   ├── client_2/
│   │   │   └── phase1/
│   │   │       ├── annotations/
│   │   │       └── data/
│   │   └── client_3/
│   │       └── phase1/
│   │           ├── annotations/
│   │           └── data/
│   └── phase2/                # Download separately
│       ├── annotations/
│       └── data/
│
├── client_updates/            # Encrypted updates from clients
│   ├── client_update_client_1_round_1.enc
│   └── ...
│
├── global_models/             # Aggregated global model files
│   ├── global_model_params_round_1.pkl
│   └── ...
│
├── scripts/
│   ├── fed_split.py           # Optional partitioning script
│   ├── key.py                 # Encryption key generator
│   └── run.py                 # Simulation control script



## 🚀 Setup Instructions

### Clone the Repo
```bash
git clone https://github.com/ibhoomi16/bot.git
cd bot
```

### Create and Activate Virtual Environment
```bash
python -m venv venv
# Windows
.env\Scriptsctivate
# macOS/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn tqdm cryptography
```

---

## 🔐 Set Up Encryption

### Generate Key
```bash
python scripts/key.py
```
Copy the generated key (starts with `b'...`).

### Paste Key in `client.py` and `server.py`
Replace:
```python
ENCRYPTION_KEY = b'YOUR_GENERATED_KEY_HERE'
```

---

## ▶️ Run the Simulation
Make sure `(venv)` is active and run:
```bash
python scripts/run.py
```

---

## 📊 Output Breakdown

- **Encrypted client updates** → `client_updates/client_update_client_X_round_Y.enc`
- **Global models** → `global_models/global_model_params_round_Y.pkl`


---

## ⚙️ Customization

You can modify in `run_federated_learning.py`:
- `NUM_FL_ROUNDS`
- `CLIENT_IDS`

In `client.py`:
- `DP_NOISE_SCALE` – Tune for privacy-accuracy trade-off
