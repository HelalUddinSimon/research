# Hessian-Based Merge Pruning

This repository implements a modular Hessian-vector based neural network pruning method that reduces model complexity while maintaining performance. The method pairs less significant weights with more significant ones using the dominant Hessian eigenvectors, effectively merging them before pruning.

## 📁 Directory Structure

```
project/
├── main.py                  # Entry point: loads model, prunes, fine-tunes, and evaluates
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # CIFAR-10 loader and preprocessing
│   ├── flops_estimator.py   # Estimate FLOPs per layer
│   ├── weight_counter.py    # Count zeros and non-zeros in model
│   ├── hessian.py           # Hessian-vector approximation
│   ├── pruning.py           # Merge-prune weights using eigenvector-based importance
│   ├── training.py          # Fine-tuning utilities
│   ├── plot_utils.py        # Generate plots for FLOPs, metrics, eigenvector stats
│   └── model_utils.py       # Load and save models
```

## 🧪 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install tensorflow numpy matplotlib
```

## 🚀 Running the Code

1. Place your pretrained ResNet56 model in the `base_model/` directory as `ResNet56.h5`.
2. Run the pruning and fine-tuning process:

```bash
python main.py
```

## 📊 Outputs

After execution, the following files are saved in `saved_models/`:
- `pruned_model_50.h5` — model after merge pruning
- `finetune_model_50.h5` — model after fine-tuning
- `flops_comparison.png` — per-layer FLOPs comparison (before vs. after)
- `metrics_comparison.png` — accuracy/loss before and after pruning
- `eigenvector_distribution.png` — histogram of eigenvector magnitudes

## 📌 Notes

- Change `percentile = 50` in `main.py` to adjust pruning severity.
- Supports CIFAR-10 dataset out of the box.
- FLOPs are estimated using layer-wise analysis and sparsity-aware computation.

## 📚 Citation

If you use this code for your research or work, please consider citing the original author or repository you adapted the base model from.
