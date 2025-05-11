# NLP Preprocessing Pipeline (Community Notes)

This repository contains only the **pre-processing** steps for the Community Notes NLP project.  
Subsequent steps—topic modeling (BERTopic) and knowledge graph construction—are maintained in separate repositories:

- **BERTopic repo:** <BERTopic_REPO_URL>
- **Knowledge Graph repo:** <KG_REPO_URL>

---

## 1. Environment Setup

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Register Jupyter kernel**
   ```bash
   pip install jupyter
   python -m ipykernel install --user --name=preproc-venv --display-name "Python - Community Notes (.venv)"
   ```

---

## 2. Directory Structure

```
project/
├── code/                         # Jupyter notebooks for each preprocessing step
│   ├── 1_Filter_English_Community_Notes_English.ipynb
│   ├── 2_Filter_Related_Data.ipynb
│   └── 3_Random_Sampling.ipynb
├── scripts/
│   ├── fetch_data.sh             # Download fixed snapshot (2025-03-08)
│   └── runout.sh                 # One-click run: fetch + execute notebooks
├── data/
│   ├── raw/                      # Raw TSV files (snapshot 2025-03-08)
│   ├── english_only/             # Filtered English TSV outputs
│   ├── samples/                  # Random sampling outputs (seed fixed)
│   └── KG/                       # Placeholder for knowledge graph outputs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 3. One-Click Installation & Execution

To run the entire preprocessing pipeline in one command:

```bash
bash scripts/runout.sh
```

### What Happens Under the Hood

1. **Download Data Snapshot**
   - `scripts/fetch_data.sh 2025/03/08` fetches the archived **2025-03-08** data snapshot.
2. **Execute Notebooks**
   - Runs `code/1_Filter_English_Community_Notes_English.ipynb`
   - Runs `code/2_Filter_Related_Data.ipynb`
   - Runs `code/3_Random_Sampling.ipynb`
3. **Generate Outputs**
   - Filtered data → `data/english_only/`
   - Sampled data → `data/samples/`

---

## 4. Reproducibility Details

### 4.1 Fixed Data Snapshot

- Command:
  ```bash
  bash scripts/fetch_data.sh 2025/03/08
  ```
- This ensures everyone uses the same archived snapshot.
- Download URL pattern:
  ```
  https://ton.twimg.com/birdwatch-public-data/2025/03/08/notes-00000.zip
  ```

### 4.2 Deterministic Random Sampling

- All sampling notebooks initialize the random seed to `42`.
- The resulting sample CSV files are also committed under `data/samples/` for quick verification.

---

## 5. Next Steps

After completing preprocessing:

- **Topic Modeling (BERTopic):** see the BERTopic repository at `<BERTopic_REPO_URL>`.
- **Knowledge Graph:** see the Knowledge Graph repository at `<KG_REPO_URL>`.

If you encounter any issues or need help with integration, please open an issue or a pull request.
