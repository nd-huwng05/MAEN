# 🧠 Medical Anomaly Detection using Residual Autoencoder

This repository contains the official implementation of the paper **"Unsupervised Medical Anomaly Detection using Residual Autoencoders"**, applied on VinCXR and RSNA Chest X-ray datasets. The model is trained to learn the distribution of normal chest X-rays and detect anomalies based on reconstruction errors.

---

## 📌 Abstract

We propose an unsupervised learning framework based on a deep residual autoencoder to detect medical anomalies in chest X-ray images. The model is trained only on normal samples and flags anomalies based on high reconstruction error. Experiments on VinCXR and RSNA datasets show promising performance in identifying pathological regions without any manual annotation.

---

## 📊 Results

| Dataset | AUC   | Accuracy |
|---------|-------|----------|
| VinCXR  | 0.912 | 84.7%    |
| RSNA    | 0.894 | 82.3%    |

Visualization of anomaly maps and reconstruction errors can be found in the `results/` folder.

---

## 📂 Project Structure

```bash
├── data/                 # Raw or processed datasets
├── models/               # Network architecture and loss functions
├── utils/                # Utility functions and helpers
├── configs/              # YAML configuration files
├── train.py              # Script to train the model
├── test.py               # Script to evaluate the model
├── infer.py              # Script for single image inference
├── checkpoints/          # Pretrained model weights
├── requirements.txt      # Required Python libraries
└── README.md             # Project description
```

---

## 📁 Dataset

We use two public chest X-ray datasets:

### 1. VinCXR  
- Vietnamese Chest X-ray dataset  
- Contains both normal and abnormal chest X-rays  
- Only normal images are used for training  

### 2. RSNA Pneumonia Detection Challenge Dataset  
- Includes labeled CXR images  
- Used only for evaluation  

👉 **Download datasets here**:  
[📥 VinCXR & RSNA Datasets](https://drive.google.com/drive/folders/1Bqzutteh3mgW3UKY-tXHLU208iktC4L-?usp=drive_link)

> After downloading, extract the folders and organize them as follows:

```bash
data/
├── VinCXR/
│   ├── images/
│   └── data.json/
└── RSNA/
    ├── images/
    └── data.json/
```

---

## 🔧 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda create -n anomaly-detection python=3.9
conda activate anomaly-detection
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the model:

```bash
python train.py --config configs/vincxr.yaml
```

### 2. Evaluate the model:

```bash
python test.py --config configs/vincxr.yaml --checkpoint checkpoints/vincxr_best.pth
```

### 3. Inference on a single image:

```bash
python infer.py --image-path example.jpg --checkpoint checkpoints/vincxr_best.pth
```

---

## 🧠 Pretrained Models

We provide pretrained model checkpoints for both VinCXR and RSNA datasets:

[📥 Download Pretrained Checkpoints](https://drive.google.com/drive/folders/1EjM9GBLt7TSGHhK41dgtqKXSnw5FvYVd?usp=sharing)

> Place the downloaded `.pth` files into the `checkpoints/` directory.

---

## 📷 Visualization

Example output includes:

- Original image  
- Reconstructed image  
- Anomaly heatmap

<p align="center">
  <img src="results/example_input.png" width="250"/>
  <img src="results/example_recon.png" width="250"/>
  <img src="results/example_error.png" width="250"/>
</p>

---

## 📖 Citation

If you use this code or dataset, please cite:

```bibtex
@misc{yourpaper2025,
  title={Unsupervised Medical Anomaly Detection using Residual Autoencoders},
  author={Your Name and Coauthors},
  year={2025},
  note={Preprint},
  url={https://github.com/yourusername/medical-anomaly-detector}
}
```

---

## 📬 Contact

For questions, feedback, or contributions, feel free to contact me via email:

- 📧 [nguyendinhhung290605@gmail.com](mailto:nguyendinhhung290605@gmail.com)

---

## 📝 License

This project is licensed under the MIT License.
