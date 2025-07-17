# Retinal Vessel Segmentation: A Comparative Study

A comprehensive evaluation of different approaches to retinal blood vessel segmentation on the DRIVE dataset, comparing classical CNN architectures, modern transformer-based models, and zero-shot segmentation capabilities.

## Project overview

This project investigates the effectiveness of different segmentation approaches for retinal blood vessel detection, addressing the critical question: **Can modern zero-shot models like SAM compete with task-specific trained models in medical image segmentation?**

## Project structure

```
sam-fsft/
├── configs/                              # Model configurations (for running from CLI)
│   ├── unet_bs1_lr0.0001_ep40.yaml
│   ├── segformer_bs1_lr5e-5_ep40.yaml
│   └── ...
├── runs/                                 # Auto-generated experiments
│   ├── unet_bs1_lr0.0001_ep40_20250708_122926/
│   │   ├── checkpoints/
│   │   ├── test_predictions/
│   │   └── tensorboard/
│   └── segformer_bs1_lr5e-5_ep40_20250710_105500/
├── dataset.py                           # Data loading and preprocessing
├── metrics.py                            # Evaluation metrics (IoU, Dice, precision and recall scores)
├── utils.py                              # Utility functions
├── unetmodel.py                          # UNet architecture
├── segformer_model.py                    # SegFormer wrapper
├── unitrain.py                           # universal training script (accepts model as kwarg)
├── evaluate.py                           # Model testing script
└── requirements.txt
```

### Key research questions
- How does the zero-shot Segment Anything Model (SAM) perform on retinal vessel segmentation without additional training?
- What are the performance differences between classical CNN architectures (UNet) and modern transformer-based models (SegFormer)?
- Are SAM's zero-shot capabilities sufficient for medical imaging applications where training data is limited?

## Architecture and models

### 1. UNet (Implemented)
- **From-scratch implementation** without pre-trained encoders
- Symmetric encoder-decoder architecture with skip connections
- Dice loss with FOV masking
- **Current Results**: Dice Score: 0.3585, IoU: 0.5787

### 2. SegFormer (Implemented)
- Transformer-based encoder (SegFormer-B0) with lightweight MLP decoder
- Efficient attention mechanism for dense prediction
- Fine-tuned with frozen encoder warmup phase (3 epochs).
- - **Current Results**: Dice Score: 0.3669, IoU: 0.5972

### 3. Segment Anything Model (SAM) (In progress)
- Zero-shot evaluation without additional training
- Multiple prompting strategies (bounding box, point, grid sampling)
- Performance comparison with trained models

## Dataset

**DRIVE Dataset** - Standard benchmark for retinal vessel segmentation
- 40 high-resolution retinal images (30\5\5)
- Manual vessel annotations and Field of View (FOV) masks
- Images scaled accordingly for training
- Evaluation performed only within FOV regions


## Performance Summary (15 Trials, FOV-Masked)

| Model     | Dice (Mean±SD)   | IoU (Mean±SD)    | Precision (Mean±SD) | Recall (Mean±SD)  | Best Dice | Best IoU |
|-----------|------------------|------------------|----------------------|-------------------|-----------|----------|
| UNet      | 0.2922 ± 0.0491  | 0.4900 ± 0.0516  | 0.6314 ± 0.0533      | 0.6856 ± 0.0436   | 0.3585    | 0.5787   |
| SegFormer | 0.3379 ± 0.0406  | 0.5540 ± 0.0638  | 0.7335 ± 0.0404      | 0.6953 ± 0.0764   | 0.3669    | 0.5972   |

## Best Hyperparameters from Optuna

| Model     | LR        | Weight Decay | Batch Size | Acc. Steps | Final Dice | Final IoU |
|-----------|-----------|--------------|------------|-------------|-------------|------------|
| UNet      | 6.34e-4   | 2.18e-4      | 1          | 1           | 0.3585      | 0.5736     |
| SegFormer | 4.84e-4   | 1.15e-6      | 1          | 1           | 0.3669      | 0.5972     |

---

## Training & Evaluation Details

### Hybrid Loss
Loss = 0.6 × BCE + 0.4 × Dice

- Encourages both pixel-wise precision and region overlap  
- Computed **only within FOV region** to exclude background

### Gradient Accumulation

- Simulates large batch sizes under memory constraints  
- Effective batch size = 4 via accumulation

### Mixed Precision

- Enabled via `torch.cuda.amp` + `GradScaler`  
- Faster training and reduced VRAM usage

### Test-Time Augmentation

- Horizontal flip → inference averaged across original + flipped input

---

## Model-Specific Design Choices

### UNet

- `InstanceNorm` used for small batch sizes  
- Bilinear interpolation for upsampling (no transposed conv)  
- `Dropout2d` in decoder: linearly decreased (0.3 → 0.0)  
- Manual `F.pad` for skip-connection alignment

### SegFormer

- Encoder: **SegFormer-B0 (pretrained)**  
- **Warmup**: Encoder frozen for first 3 epochs  
- Bilinear upsampling to match output size  
- Optimizer: **AdamW + linear LR scheduler**

---

## Evaluation Protocol

- FOV-masked per-pixel metrics: *Dice, IoU, Precision, Recall*
- Logged to TensorBoard:
  - Loss curves
  - Prediction samples
  - Weight/grad histograms

### Reproducibility

```python
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

### Experiment tracking
- **TensorBoard**: Real-time training monitoring
- **Automatic logging**: All experiments saved with timestamps
- **Reproducible configs**: YAML-based configuration management

### Comparisons
- **Quantitative**: IoU, Dice coefficient, precision/recall
- **Qualitative**: Visual inspection of segmentation quality
- **Computational**: Training time, inference speed, memory usage

## Academic context

This project serves dual purposes:
- **Neural Networks Course**: Deep dive into architecture design, training strategies, and performance optimization
- **Technologies and Tools in ML Course**: Focus on experiment tracking, model evaluation pipelines, and deployment considerations

## Future Work

- [ ] Implement SAM zero-shot evaluation with multiple prompting strategies
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
