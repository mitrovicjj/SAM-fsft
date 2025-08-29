# Retinal Vessel Segmentation: A Comparative Study

A comprehensive evaluation of different approaches to retinal blood vessel segmentation on the DRIVE dataset, comparing classical CNN architectures and modern transformer-based models under limited data and resource constraints.

## Project Overview

This project investigates the effectiveness of **UNet** and **SegFormer-B0** for retinal blood vessel segmentation, with special focus on:

- Training under **limited GPU and dataset constraints** (Google Colab Pro, 16GB T4 GPU, DRIVE dataset of 40 images).  
- Impact of hyperparameter optimization, data augmentations and FOV masking.  
- Quantitative vs. qualitative trade-offs: metric performance vs. morphological continuity of vessels.  


## Key Research Questions

- How do CNN-based (UNet) and transformer-based (SegFormer-B0) models behave under limited resources?  
- What is the impact of augmentations and FOV masking on segmentation quality?  
- Do models that score higher on Dice/IoU always preserve clinically relevant morphology?  

## Architectures and Models

### 1. UNet
- Implementation from `segmentation_models_pytorch` with modifications:
  - Initial filters reduced (32 → 64 standard).
  - `InstanceNorm2d` instead of BatchNorm (stability with small batches).
  - `Dropout2d` in decoder with linear decay (0.3 → 0.0).
- **Training**: Hybrid loss (0.6 × BCE + 0.4 × Dice), Adam optimizer, LR scheduler.  
- **Results**: Lower Dice/IoU than SegFormer, but better at preserving vascular continuity.

### 2. SegFormer-B0
- Transformer encoder + lightweight MLP decoder (HuggingFace pretrained `nvidia/segformer-b0`).  
- Adaptations:
  - Replaced classification head with binary segmentation head (Xavier initialization).  
  - Encoder frozen for first 3 epochs (warm-up).  
  - Auxiliary head included for improved gradient flow.  
- **Results**: Higher Dice/IoU and more stable across trials, but tends to miss thin vessels and lose morphological continuity.  

---

## Dataset

**DRIVE Dataset**  
- 40 RGB images (584×565), including 7 abnormal pathology cases.  
- Train/val/test split: 30 / 5 / 5.  
- High class imbalance (~10% vessel pixels).  
- Evaluation restricted to **FOV regions**.  

Augmentations (Albumentations): flips, rotations, brightness/contrast shifts, elastic distortions.  

---

## Performance Summary (15 Trials, FOV-Masked)

| Model     | Dice (Mean±SD)   | IoU (Mean±SD)    | Precision (Mean) | Recall (Mean)  | Best Dice | Best IoU |
|-----------|------------------|------------------|------------------|----------------|-----------|----------|
| UNet      | 0.292 ± 0.049    | 0.490 ± 0.052    | 0.631            | 0.686          | 0.359     | 0.579    |
| SegFormer | 0.338 ± 0.041    | 0.554 ± 0.064    | 0.734            | 0.695          | 0.367     | 0.597    |

**Findings**:  
- **SegFormer-B0** outperforms UNet in Dice, IoU, and stability.  
- **UNet** maintains vascular continuity better (thin vessels, bifurcations).  
- **Augmentations** did **not** improve performance (sometimes decreased it).  
- **FOV masking** had a positive impact by reducing noise.  

---

## Best Hyperparameters from Optuna

| Model     | LR        | Weight Decay | Batch Size | Acc. Steps | Final Dice | Final IoU |
|-----------|-----------|--------------|------------|-------------|------------|-----------|
| UNet      | 6.34e-4   | 2.18e-4      | 1          | 1           | 0.359      | 0.574     |
| SegFormer | 4.84e-4   | 1.15e-6      | 1          | 1           | 0.367      | 0.597     |

---

## Training & Evaluation Details

- **Hybrid loss**: `0.6 × BCE + 0.4 × Dice`, restricted to FOV.  
- **Gradient accumulation**: Effective batch size = 4.  
- **Mixed precision**: `torch.cuda.amp` for VRAM efficiency.  
- **Early stopping**: Patience = 10 epochs.  
- **LR scheduler**: ReduceLROnPlateau.  
- **Evaluation**: Dice, IoU, Precision, Recall + qualitative visual inspection.  

---

## Key Insights

- **SegFormer-B0**: better numeric performance, lower variance, robust to weight decay.  
- **UNet**: worse metrics, but better **morphological integrity** → relevant for clinical use.  
- **FOV masking**: essential.  
- **Augmentations**: neutral or negative effect under small batch sizes.  

---

## Future Work

- [ ] Integrate **SAM** for zero-shot evaluation and compare with trained models.  
- [ ] Explore **topology-aware loss functions** (Tversky, focal, unified focal loss).  
- [ ] Use **larger pretrained backbones** (Swin Transformer, medical-domain pretrained).  
- [ ] Ensemble methods for improved robustness.  
- [ ] Cross-validation to mitigate dependence on a single split.  

---

## References

- Staal et al., DRIVE dataset.  
- Ronneberger et al., UNet.  
- Xie et al., SegFormer.  
- Additional references from the experimental study (Albumentations, Optuna, topology-aware losses).  

---

*Developed as part of coursework in Neural Networks and MLOps, focusing on efficient model training and evaluation under constrained conditions.*  
