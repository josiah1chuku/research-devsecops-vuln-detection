# Transformer-Driven Shift-Left Security
## Integrating Transformer Encoders into DevSecOps Pipelines

**Author:** Josiah Chuku | **Supervisor:** Dr. Liu, Jinwei | **FAMU 2026**

[![CI](https://github.com/josiah1chuku/research-devsecops-vuln-detection/actions/workflows/evaluate.yml/badge.svg)](https://github.com/josiah1chuku/research-devsecops-vuln-detection/actions)

## Results (DiverseVul, 66,098 test functions)

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.7677** |
| Best Val F1 | **0.7794** |
| MCC | 0.2117 |
| Recall | 0.3774 |
| Precision | 0.1948 |
| Accuracy | 87.0% |

## Run End to End

```bash
git clone https://github.com/josiah1chuku/research-devsecops-vuln-detection.git
cd research-devsecops-vuln-detection
pip install -r requirements.txt
python step5_train/prepare_data.py     # 1. Split dataset
python step5_train/build_dfg_cache.py  # 2. Extract DFGs
python step5_train/train.py            # 3. Train model
python step6_eval/evaluate.py          # 4. Evaluate
```

## Colab Notebook (A100 GPU)
https://colab.research.google.com/drive/17-HJnTymfBtEFXbRL1ZMkQtUEyzx-4EL
