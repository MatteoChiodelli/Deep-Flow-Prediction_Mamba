# Intro
This is an expansion on the research done by the Deep-Flow-Prediction repository.
The objective is to expand the U-net proposed in that paper, in particular using various integration of the MAMBA architecture

# Research Questions & Architectural Exploration Matrix

This document outlines the Research Questions (RQ), the configuration matrix, and the methodological directions for evaluating **State Space Models (Mamba)** and **Self-Attention** blocks integrated into the **TurbNet** architecture for 2D aerodynamic field estimation (CFD).

---

## 1. Research Questions (RQ)

* **RQ1 (Full Hybrid - Multiscale Context):** Does coupling high-resolution Mamba (early layers, linear $\mathcal{O}(N)$ complexity to capture long dependencies) with low-resolution Self-Attention (bottleneck for condensed global correlations) outperform both the purely convolutional baseline and single-module hybrid models?
* **RQ2 (High-Resolution SSM):** Does leveraging Mamba's linear scalability on high-resolution layers ($64 \times 64$ or $128 \times 128$) improve the propagation of wake and pressure gradient information compared to the purely local receptive field of convolutions?
* **RQ3 (Bottleneck SSM):** Does a Mamba block positioned exclusively in the deep bottleneck (short sequence, strong abstraction) offer computational or qualitative advantages over the original convolutional block?
* **RQ4 (Bottleneck Attention Benchmark):** Does the standard Multi-Head Self-Attention block at the bottleneck represent an upper or lower bound compared to an SSM-based bottleneck at parity of resolution and parameters?
* **RQ5 (Inductive Bias Trade-off: Conv vs Sequence):** Does the progressive replacement of convolutional layers in favor of purely sequential blocks degrade the reconstruction of the boundary layer (airfoil surface boundary) or preserve geometric coherence?
* **RQ6 (Non-Raster & Physics-Informed 2D Scanning Strategies):** *[Exploratory / High-Impact]* The standard raster scan (row-by-row / 2D bidirectional scan) imposes an artificial discontinuity in 2D dimensionality. Does an alternative spatial linearization strategy — space-filling curves (e.g., **Hilbert/Peano curves**) or a physics-informed scan along the prevalent flow direction (streamline/aerodynamic-aligned scan) — better preserve local spatial correlation and wake dynamics compared to the standard raster?

---

## 2. Experimental Configurations

| Config ID | Architecture | High-Res Stages ($128^2 / 64^2$) | Bottleneck ($4^2 / 2^2$) | Conv Layers | Mamba Scanning Mode | Main Objective |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C0** | **Baseline TurbNet** | Standard Convolutions | Standard Convolutions | Standard (All) | N/A | Reference benchmark |
| **C1** | **Hybrid Mamba + Attention** | **Mamba 2D** | **Self-Attention** | Mixed | Standard 2D Bidirectional | Extended multiscale model (RQ1) |
| **C2** | **Early Mamba Only** | **Mamba 2D** | Standard Convolutions | Standard Conv | Standard 2D Bidirectional | High-res $\mathcal{O}(N)$ scalability (RQ2) |
| **C3** | **Bottleneck Mamba Only** | Standard Convolutions | **Mamba 2D** | Standard Conv | Standard 2D Bidirectional | SSM on compact sequences (RQ3) |
| **C4** | **Bottleneck Attention Only**| Standard Convolutions | **Self-Attention** | Standard Conv | N/A | Classic Transformer benchmark (RQ4) |
| **C5** | **Conv-Free / Minimal Conv**| Mamba / Attention | Mamba / Attention | Minimized | Standard 2D Bidirectional | Local inductive bias study (RQ5) |
| **C6** | **Mamba Geometry/Flow-Aware**| **Mamba 2D (Advanced)** | Best Setup Choice | Standard / Mixed | **Hilbert Curve / Streamwise Scan** | Anisotropic 2D linearization for CFD (RQ6) |

---

## 3. 2D Mamba Scanning Focus (RQ6 Detail)

1. **Standard Raster Scan (2D Baseline):**
   * *Mechanism:* Horizontal scan $\rightarrow$ Vertical scan (and inverse bidirectional).
   * *Limitation:* Two vertically adjacent pixels $(x, y)$ and $(x, y+1)$ are separated by a sequential distance equal to $W$ (channel width).
2. **Hilbert Space-Filling Curve:**
   * *Mechanism:* Fractal $2\text{D} \rightarrow 1\text{D}$ mapping that preserves topological proximity (two close points in space remain almost always close along the curve).
   * *CFD Advantage:* Keeps local isotropic turbulent correlation intact on vortices and rapid pressure variations.
3. **Streamwise / Directional Scan:**
   * *Mechanism:* Sequential ordering oriented along the main flow axis (left-to-right, from upstream to downstream of the airfoil).
   * *CFD Advantage:* Reflects the hyperbolic/parabolic nature of mass and momentum transport in the fluid flow field.

---

## 4. Training & Comparison Protocol

* **Dataset:** Subset of ~6.4k samples for exploration and configuration screening; full dataset reserved for the final model.
* **Fixed Hyperparameters (Iso-hyperparameters):**
  * $\text{Batch Size} = 10$.
  * Adam optimizer with original schedule and learning rate ($0.0006$).
  * Identical maximum number of epochs with best model saving based on validation loss (Early Stopping checkpoint).
* **Parametric Constraint (Iso-capacity):**
  * Trainable parameters for each variant must remain comparable to the baseline ($\sim 150\text{k} - 250\text{k}$).
* **Multi-Seed Validation:**
  * 3 fixed seeds (`42`, `1955`, `2003`) for the final configurations, reporting metrics in $\text{Mean} \pm \text{Std}$ format.
* **Target Metrics:**
  * **Relative $L_1$ Error** for $u_x$, $u_y$, $p$ and cumulative.
  * Peak VRAM footprint (MB/GB).
  * Temporal throughput (seconds/epoch).

---

## 5. Current Status (Baseline Replication)

Currently, the pipeline is set up to replicate the benchmark results of the original model (**Baseline C0**) with 3 different seedings. 
* **Parameters:** Using the exact hyperparameters provided by the original repository.
* **Dataset:** Training is executed on the local reduced dataset (`train_small`, containing approximately 5,000-6,000 samples).
* **Metrics Output:** Epoch-by-epoch results, relative percentage errors, and loss function plots are automatically saved in the `metrics/` directory.
