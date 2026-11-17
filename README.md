ECG Morphology-Aware Data Augmentation



Research project | ECG Signal Processing | Multilabel Classification\*\*



Overview



This project investigates how ECG data augmentation affects signal morphology and multilabel classification performance.



The objective is to understand whether augmentation techniques that preserve clinically relevant ECG features can also support effective ECG classification.



Research Questions



\- How do different augmentation techniques affect ECG morphology?

\- How does augmentation intensity influence signal preservation?

\- Is morphology preservation associated with classification performance?

\- Can morphology-informed augmentation selection improve model generalisation?



Dataset



PTB-XL — 12-lead ECG dataset



\- 21,799 ECG records

\- 10-second recordings

\- Sampling frequencies: 100 Hz and 500 Hz

\- Five multilabel superclass categories:

&#x20; - NORM

&#x20; - MI

&#x20; - STTC

&#x20; - CD

&#x20; - HYP



Dataset source: \[PTB-XL on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)



The dataset is not included in this repository. Download it separately from PhysioNet.



Augmentation Techniques



1\. Gaussian Noise

2\. Amplitude Scaling

3\. Temporal Shifting

4\. Temporal Warping



Each technique is evaluated at mild, moderate and strong intensities.



Morphology Evaluation



The current evaluation investigates:



\- R-peak matching

\- QRS duration differences

\- Signal correlation

\- QRS coverage

\- Alignment-aware signal comparison



Morphology analysis uses 500 Hz ECG recordings.



Classification



A 1D CNN is implemented for five-label ECG classification.



Planned evaluation metrics:



\- Macro-F1

\- Macro-AUROC

\- Macro Precision

\- Macro Recall



Repository Structure



ECG\_Research/

├── augmentations.py

├── data\_loader.py

├── models.py

├── train.py

├── evaluate.py

├── classification\_experiments.py

├── morphology\_500hz.py

├── alignment\_evaluation\_500hz.py

├── generate\_morphology\_report.py

├── results/

│   └── github\_demo/

├── requirements.txt

└── README.md





Current Status



\- PTB-XL data preprocessing

\- Multilabel label preparation

\- Four augmentation techniques implemented

\- Initial morphology evaluation completed

\- Alignment-aware evaluation implemented

\- CNN classification pipeline implemented

\- Controlled augmentation experiments in progress



Note: Current morphology results are preliminary and based on an initial 100-record sample. They should not be interpreted as final clinical validation or definitive research conclusions.



Technology Stack



Python, PyTorch, NumPy, Pandas, SciPy, Matplotlib, Scikit-learn, WFDB and NeuroKit2.



Future Work



\- Improve morphology evaluation reliability.

\- Complete controlled classification experiments.

\- Compare baseline and morphology-informed augmentation.

\- Evaluate generalisation on an independent ECG dataset.

\- Compare additional model architectures.



Disclaimer



This is an academic research project. It is not a clinical diagnostic system.

Preliminary Baseline Results



The initial 1D CNN baseline was trained on PTB-XL without data augmentation.



| Metric               | Result |

| -------------------- | -----: |

| Macro-F1             | 0.6825 |

| Macro Precision      | 0.7724 |

| Macro Recall         | 0.6306 |

| Macro-AUROC          | 0.9077 |

| Best Validation Loss | 0.2989 |



\*\*Experimental setup:\*\*



\- Training records: 17,084

\- Validation records: 2,146

\- Test records: 2,158

\- Epochs: 5

\- Random seed: 42



These are preliminary baseline results. Further controlled experiments are required before drawing conclusions about augmentation effectiveness.





