# Retinal Vessel Segmentation: A Comparative Study

A comprehensive evaluation of different approaches to retinal blood vessel segmentation on the DRIVE dataset, comparing classical CNN architectures, modern transformer-based models, and zero-shot segmentation capabilities.

## Project Overview

This project investigates the effectiveness of different segmentation approaches for retinal blood vessel detection, addressing the critical question: **Can modern zero-shot models like SAM compete with task-specific trained models in medical image segmentation?**

### Key Research Questions
- How does the zero-shot Segment Anything Model (SAM) perform on retinal vessel segmentation without additional training?
- What are the performance differences between classical CNN architectures (UNet) and modern transformer-based models (SegFormer)?
- Are SAM's zero-shot capabilities sufficient for medical imaging applications where training data is limited?

## Architecture & Models

### 1. UNet (Implemented ✅)
- **From-scratch implementation** without pre-trained encoders
- Symmetric encoder-decoder architecture with skip connections
- Dice loss with FOV masking
- **Current Results**: Dice Score: 0.5755, IoU: 0.5278

### 2. SegFormer (In Progress 🔄)
- Transformer-based hierarchical architecture
- Efficient attention mechanism for dense prediction
- Comparison with classical CNN approach

### 3. Segment Anything Model (SAM) (Planned 📋)
- Zero-shot evaluation without additional training
- Multiple prompting strategies (bounding box, point, grid sampling)
- Performance comparison with trained models

## Dataset

**DRIVE Dataset** - Standard benchmark for retinal vessel segmentation
- 40 high-resolution retinal images (30\5\5)
- Manual vessel annotations and Field of View (FOV) masks
- Images scaled to 256×256 for training
- Evaluation performed only within FOV regions

## Project Structure

```
sam-fsft/
├── configs/                              # Model configurations
│   ├── unet_bs1_lr0.0001_ep40.yaml
│   ├── segformer_bs1_lr5e-5_ep40.yaml
│   └── ...
├── runs/                                 # Auto-generated experiments
│   ├── unet_bs1_lr0.0001_ep40_20250708_122926/
│   │   ├── checkpoints/
│   │   ├── test_predictions/
│   │   ├── test_outputs/
│   │   └── tensorboard/
│   └── segformer_bs1_lr5e-5_ep40_20250710_105500/
├── datasets.py                           # Data loading and preprocessing
├── metrics.py                            # Evaluation metrics (IoU, Dice, precision and recall scores)
├── utils.py                              # Utility functions
├── unetmodel.py                          # UNet architecture
├── segformer_model.py                    # SegFormer wrapper
├── unettrain.py                          # UNet training script
├── segformertrain.py                     # SegFormer training script
├── unetbest.py                           # Model testing script
└── requirements.txt
```

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Setup for Google Colab
This project is designed to run on Google Colab due to computational requirements. Training scripts support both bash+YAML configuration files and direct Colab notebook calls with kwargs for flexibility.

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/retinal-segmentation
cd retinal-segmentation
```

2. **Download DRIVE dataset**:
   - Download from [DRIVE dataset official page](https://drive.grand-challenge.org/)
   - Upload to Google Drive in folder structure: `/content/drive/MyDrive/DRIVE/`

3. **Mount Google Drive in Colab**:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Training

**UNet Training**:

*Option 1: Using bash + YAML config*
```bash
python unettrain.py --config configs/unet_bs1_lr0.0001_ep40.yaml
```

*Option 2: Direct Colab notebook call*
```python
from unettrain import train_unet
train_unet(
    batch_size=1,
    learning_rate=0.0001,
    epochs=40,
    model_name='unet_experiment'
)
```

**SegFormer Training** (Coming Soon):

*Option 1: Using bash + YAML config*
```bash
python segformertrain.py --config configs/segformer_bs1_lr5e-5_ep40.yaml
```

*Option 2: Direct Colab notebook call*
```python
from segformertrain import train_segformer
train_segformer(
    batch_size=1,
    learning_rate=5e-5,
    epochs=40,
    model_name='segformer_experiment'
)
```

### Evaluation
```bash
python unetbest.py --model_path runs/your_experiment/checkpoints/best_model.pth
```

## Results

### Current Performance (UNet)
| Model | Dice Score | IoU Score | Training Time |
|-------|------------|-----------|---------------|
| UNet  | 0.5755     | 0.5278    | ~40 epochs    |
| SegFormer | TBD    | TBD       | TBD           |
| SAM (zero-shot) | TBD | TBD    | No training   |

### Experiment Tracking
- **TensorBoard**: Real-time training monitoring
- **Automatic logging**: All experiments saved with timestamps
- **Reproducible configs**: YAML-based configuration management

### Planned Comparisons
- **Quantitative**: IoU, Dice coefficient, precision/recall
- **Qualitative**: Visual inspection of segmentation quality
- **Computational**: Training time, inference speed, memory usage

## Academic Context

This project serves dual purposes:
- **Neural Networks Course**: Deep dive into architecture design, training strategies, and performance optimization
- **Technologies and Tools in ML Course**: Focus on experiment tracking, model evaluation pipelines, and deployment considerations

## Future Work

- [ ] Add data augmentation for improved UNet performance
- [ ] Complete SegFormer implementation and evaluation
- [ ] Implement SAM zero-shot evaluation with multiple prompting strategies
- [ ] Comprehensive error analysis and failure case studies
- [ ] Model deployment pipeline (FastAPI + Docker)

## References

- **DRIVE Dataset**: Staal, J., et al. "Ridge-based vessel segmentation in color images of the retina." *IEEE TMI* (2004)
- **UNet**: Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI* (2015)
- **SegFormer**: Xie, E., et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." *NeurIPS* (2021)
- **SAM**: Kirillov, A., et al. "Segment Anything." *ICCV* (2023)

## Contributing

This is an academic project, but suggestions and discussions are welcome! Feel free to open issues or reach out with questions.

## License

This project is for educational purposes. Please respect the original dataset licenses and model usage terms.

---

*Developed as part of Neural Networks and MLOps coursework, focusing on the intersection of classical and modern approaches to medical image segmentation.*
