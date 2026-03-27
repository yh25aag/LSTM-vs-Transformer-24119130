# LSTM vs Transformer: Why Attention Won

A tutorial on sequential memory, self-attention, and why the Transformer replaced the LSTM — with full training code and reproducible experiments on the AG News dataset.

---

## Overview

For most of the 2010s, the Long Short-Term Memory network (LSTM) dominated sequential modelling. In 2017, Vaswani et al. published *Attention Is All You Need*, and within two years LSTMs had been largely displaced across NLP. This tutorial explains why — mathematically, architecturally, and empirically.

We cover:
- The **vanishing gradient problem** and why it limits LSTMs
- How the **LSTM cell state** partially mitigates it (but not enough)
- How **self-attention** reduces the gradient path length from O(n) to O(1)
- A **head-to-head experiment** training both models on AG News (4-class news classification)
- **Attention heatmap visualisation** showing what each head learns

---

## Repository Structure

```
lstm-vs-transformer/
├── README.md
├── LICENSE
├── tutorial/
│   └── LSTM_vs_Transformer_Tutorial.pdf
├── figures/
│   ├── fig1_vanishing_gradient.png
│   ├── fig2_training_curves.png
│   ├── fig3_attention_heatmap.png
│   ├── fig4_epoch_time.png
│   └── fig5_complexity.png
└── code/
    ├── train.py                  # Main training script (BiLSTM + Transformer)
    ├── models/
    │   ├── bilstm.py             # BiLSTM classifier
    │   └── transformer.py        # Transformer encoder classifier
    ├── data/
    │   └── ag_news.py            # Dataset loading and preprocessing
    └── visualise/
        ├── gradient_flow.py      # Figure 1 — vanishing gradient plot
        ├── complexity.py         # Figure 5 — O(n) vs O(1) complexity plot
        └── attention_heatmap.py  # Figure 3 — attention weight visualisation
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/lstm-vs-transformer.git
cd lstm-vs-transformer
```

### 2. Install dependencies

Python 3.12 and PyTorch are required. Install all dependencies with:

```bash
pip install torch torchtext matplotlib numpy
```

### 3. Train both models

```bash
python code/train.py
```

This will download AG News automatically via `torchtext`, train both a BiLSTM and a Transformer encoder for 5 epochs, and print accuracy and epoch time for each.

### 4. Reproduce figures

```bash
python code/visualise/gradient_flow.py      # Figure 1
python code/visualise/complexity.py         # Figure 5
python code/visualise/attention_heatmap.py  # Figure 3
```

---

## Dataset

**AG News** (Zhang et al., 2015) — 120,000 training and 7,600 test news headlines across four categories:

| Label | Topic |
|-------|-------|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Science / Technology |

AG News is loaded automatically by `torchtext.datasets`. No manual download is required.

---

## Models

Both models use identical hyperparameters wherever possible to ensure a fair comparison.

| Hyperparameter | BiLSTM | Transformer |
|----------------|--------|-------------|
| Embedding dim | 128 | 128 (d_model) |
| Hidden / FF dim | 128 bi-dir (256 total) | FF dim = 256 |
| Layers | 2 stacked BiLSTM | 2 encoder layers |
| Attention heads | — | 4 heads (32 dim each) |
| Dropout | 0.3 | 0.3 |
| Optimiser | Adam lr=0.001 | Adam lr=0.001 |
| Batch size | 64 | 64 |
| Epochs | 5 | 5 |

---

## Results

Trained on CPU, same parameter count, same data, same hardware.

| Metric | BiLSTM | Transformer |
|--------|--------|-------------|
| Final test accuracy | 87.1% | 91.3% |
| Avg epoch time | 88.1s | 42.2s (2.1× faster) |

The Transformer converges faster, reaches higher accuracy, and trains more quickly due to parallelisable attention operations.

---

## Figures

| Figure | Description |
|--------|-------------|
| Fig 1 | Vanishing gradients: Vanilla RNN vs LSTM vs Transformer across backpropagation depth |
| Fig 2 | Training curves: cross-entropy loss and test accuracy over 5 epochs |
| Fig 3 | Self-attention heatmap (Head 1, Layer 1) on the headline "Oil prices fall after OPEC deal" |
| Fig 4 | Training speed: average epoch time in seconds (CPU) |
| Fig 5 | Theoretical complexity: gradient path length and sequential operations vs sequence length |

All figures use the **Wong (2011) 8-colour colourblind-safe palette**, with solid vs dashed line styles for greyscale readability. Orange = BiLSTM, Blue = Transformer throughout.

---

## Accessibility

- Colourblind-safe palette (distinguishable under deuteranopia, protanopia, and tritanopia)
- Distinct line styles (solid vs dashed) so plots are readable in greyscale
- Semantic heading levels in the PDF for screen-reader compatibility
- Attention heatmaps use a sequential colourmap rather than red-green

---

## References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017. https://arxiv.org/abs/1706.03762
2. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
3. Bengio, Y., Simard, P. & Frasconi, P. (1994). *Learning long-term dependencies with gradient descent is difficult.* IEEE Trans. Neural Networks. https://doi.org/10.1109/72.279181
4. Zhang, X., Zhao, J. & LeCun, Y. (2015). *Character-level CNNs for Text Classification.* NeurIPS 2015. https://arxiv.org/abs/1509.01626
5. Alammar, J. (2018). *The Illustrated Transformer.* https://jalammar.github.io/illustrated-transformer/
6. Wong, B. (2011). *Color blindness.* Nature Methods, 8(6), 441. https://doi.org/10.1038/nmeth.1618
7. PyTorch Documentation. https://pytorch.org/docs/stable/index.html

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
