bot-detector/
├── dataset/                  # Contains the raw data, split into client partitions
│   ├── partition/            # Data partitioned for each client
│   │   ├── client_1/
│   │   │   └── phase1/
│   │   │       ├── annotations/
│   │   │       ├── data/
│   │   │       └── ...
│   │   ├── client_2/
│   │   └── client_3/
│   └── ... (original dataset files, if any)
├── scripts/                  # Utility scripts
│   ├── fed_split.py          # (Optional) Script to initially partition your data
│   └── key.py                # Script to generate the encryption key
├── client.py                 # Client-side FL logic (local training, privacy, encryption)
├── server.py                 # Server-side FL logic (decryption, aggregation)
├── run_.py # Orchestrates the FL simulation rounds
├── README.md                 # This file
└── venv/                     # Python Virtual Environment (recommended)
#### ⚙️ Configuration / Customization
You can modify the following variables in run_federated_learning.py to customize the simulation:

## NUM_FL_ROUNDS:
 Change this integer to simulate more or fewer Federated Learning rounds (default is 20).

 
## In client.py, you can adjust:

DP_NOISE_SCALE: Controls the amount of noise added for Differential Privacy. A higher value increases privacy but might decrease model accuracy. Experiment to find a trade-off.


######  How to Run the Simulation
## Ensure your terminal is in the bot-detector directory
python scripts/run.py
