
## 🎯 Task Description & Requirements
The objective is to design and train a neural network that takes in $n$ data points as $(x_i, y_i)$ coordinates and produces a fixed-size representation to predict **7 specific outputs**:
1. **Circle Existence (1 value):** Binary output indicating whether a circle exists in the point cloud.
2. **Thickness Class (3 values):** Three scores corresponding to the circle's thickness: *thin*, *medium*, or *thick*.
3. **Circle Parameters (3 values):** Center coordinates and radius $(x, y, r)$.

**Key Constraints:**
- Handle variable-length point clouds.
- Thickness and circle parameters are only meaningful when a circle exists, requiring a masked multi-task loss.
- **Strict Architecture Limitation:** Advanced architectures like CNNs, Transformers, or ViTs are strictly prohibited. The solution must rely on pure Multilayer Perceptrons (MLPs).

## 🧠 Technical Implementation & Methodology

To overcome the challenge of variable-length inputs without using CNNs or Transformers, this project relies on **robust geometric feature engineering** to map the point cloud into a fixed **201-dimensional** feature vector.

### 1. Geometric Feature Extraction
- **Pairwise Distance Histogram:** Computes pairwise distances between points to create a unique distribution signature. A circle yields a specific chord-length peak, while lines or blobs produce different histogram shapes.
- **Hough-like Center Search:** Tests a 5×5 grid of candidate centers and computes a radial distance histogram for each. A sharp peak in this histogram strongly indicates the presence of a circle.
- **Radial Spread Statistics (For Thickness):** Extracts Interquartile Range (IQR), Median Absolute Deviation (MAD), and Coefficient of Variation (CV) from the best candidate center. Thin circles have a tight spread (low variance), while medium/thick circles have wider spreads.
- **Angular & Marginal Statistics:** Analyzes the angular coverage/entropy and projects points onto X and Y marginal histograms.
- **RANSAC-like Stats:** Computes best inlier ratios and mean residuals for circle fitting.

### 2. Data Processing & Augmentation
- **Feature Normalizer:** Because feature scales vary wildly (probabilities vs. pixel coordinates vs. sharpness ratios), a Z-score normalizer is fit strictly on the training set and applied to validation/test sets to prevent gradient explosion.
- **Augmentation:** Employs geometric augmentations like random rotations and random flips. Circle parameters (center $(x,y)$) are mathematically transformed (using sine/cosine rotation matrices) to match the augmented points.

### 3. Model Architecture (`CircleNet`)
The architecture is a **Pure MLP** with a shared trunk and three task-specific heads:
- **Shared Trunk:** Linear layers (512 → 384 → 256) with `BatchNorm1d`, `ReLU`, and `Dropout`.
- **Existence Head:** Linear layers (128 → 64 → 1) predicting the binary presence of a circle.
- **Thickness Head:** Linear layers (128 → 3) predicting the thickness class.
- **Parameters Head:** Linear layers (64 → 3) predicting $(x, y, r)$.

### 4. Masked Multi-Task Loss (`MultiTaskLoss`)
A custom loss function calculates the total error by combining three different losses:
- `BCEWithLogitsLoss` for existence (with a positive weight penalty to handle class imbalance).
- `CrossEntropyLoss` for thickness.
- `SmoothL1Loss` for circle parameters.

*Crucially, the thickness and parameter losses are **masked** (multiplied by the ground truth existence label) so the network is not penalized for incorrect radius predictions when no circle actually exists.* 

### 5. Training Strategy
- **Optimizer:** `AdamW` (Learning Rate: `3e-4`, Weight Decay: `1e-4`).
- **Scheduler:** `CosineAnnealingLR` decaying over 300 epochs.
- **Early Stopping:** Monitors the validation total loss and halts training if no improvement is seen for 35 epochs.
- **Optimal Threshold Search:** Automatically iterates over probability thresholds `[0.1, 0.9]` on the validation set to find the perfect cutoff for binary circle existence.

## 📂 Dataset
The synthetic dataset required to run this notebook can be found here:
👉 **[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1HBh94fSn-WegZiBCud2nOyAnyeOddPv7?usp=drive_link)**

*Please download the dataset and place it in the appropriate directory as expected by the notebook before running.*

## ⚙️ Setup & Execution
1. Clone the repository:
   ```bash
   git clone <YOUR_GITHUB_REPO_LINK>
   cd <REPO_FOLDER>
   ```

2. Create a virtual environment (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook and open the assignment:
   ```bash
   jupyter notebook 23050111037_MustafaSefaSoysal_HW1.ipynb
   ```
