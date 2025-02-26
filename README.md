# Reliable uncertainty quantification for 2D/3D automatic anatomical landmark localization using multi-output conformal prediction
This repository contains the code related to Jonkers et al.'s paper "Reliable uncertainty quantification for 2D/3D automatic anatomical landmark localization using multi-output conformal prediction" (2025).

## Abstract
Automatic anatomical landmark localization in medical imaging requires not just accurate predictions but reliable uncertainty quantification for effective clinical decision support. Current uncertainty quantification approaches often fall short, particularly when combined with normality assumptions, systematically underestimating total predictive uncertainty. This paper introduces conformal prediction as a framework for reliable uncertainty quantification in anatomical landmark localization, addressing a critical gap in automatic landmark localization. We present two novel approaches guaranteeing finite-sample validity for multi-output prediction: Multi-output Regression-as-Classification Conformal Prediction (M-R2CCP) and its variant Multi-output Regression to Classification Conformal Prediction set to Region (M-R2C2R). Unlike conventional methods that produce axis-aligned hyperrectangular or ellipsoidal regions, our approaches generate flexible, non-convex prediction regions that better capture the underlying uncertainty structure of landmark predictions. Through extensive empirical evaluation across multiple 2D and 3D datasets, we demonstrate that our methods consistently outperform existing multi-output conformal prediction approaches in both validity and efficiency. This work represents a significant advancement in reliable uncertainty estimation for anatomical landmark localization, providing clinicians with trustworthy confidence measures for their diagnoses. While developed for medical imaging, these methods show promise for broader applications in multi-output regression problems.

## Repository Structure
```
📂 data        # Necessary files required for processing and evaluation
📂 notebooks   # Jupyter notebooks for result analysis and UQ approach examples
📂 results     # Stored results from experiments
📂 scripts     # Evaluation scripts for different models
📂 src         # Implementation of uncertainty quantification methods (uncertainty.py)
📂 training    # Code for training deterministic landmark localization models using PyTorch Lightning.
```

## Installation
1. Clone the repository
```bash
git clone https://github.com/predict-idlab/landmark-uq.git
cd landmark-uq
```
2. Install the required packages
```bash
pip install -r requirements.txt
```
3. Ensure that all necessary data files are placed in the data/ directory before running the scripts.

## Citation
If you use this code or find it helpful, please cite our paper:


```
UNDER REVIEW
```

## License
This project is licensed under the MIT [license](LICENSE).

---

<p align="center">
👤 <i>Jef Jonkers</i>
</p>