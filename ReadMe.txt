*** Instructions for code understanding ***

1. We extracted features set by GAAC.py, GDPC.py, and DDE.py from HemoPI-1, HemoPI-2, and HemoPI-3 datasets.
2. Raw PSSMs are computed by online link: http://possum.erc.monash.edu/.
3. Each PSSM is then passed S-CS-PSSM.m and S-PSSM-AT.m 
4. Before running S-CS-PSSM.m and S-PSSM-AT.m, please make sure you have installed Matlab 2018b version.
5. All extracted features are fused and provided to SVM-RFE.py for selection of 200 features set.
6. The selected features set then input into MERCNN, XGB, and ERT for model training and prediction.

Note: The following major libraries and their versions are required, ensuring the reproducibility of the computational environment:
•	biopython 1.85
•	numpy 1.26.4
•	pandas 2.2.3
•	scikit-learn 1.6.1
•	scipy 1.15.1
•	tensorflow 2.18.0
•	torch 2.5.1
•	keras 3.8.0
•	matplotlib 3.10.0
  