# iMFP-BG
📋Discovery of novel multi-functional peptides by using protein language models and graph-based deep learning

## 📘 Abstract
&nbsp;&nbsp;&nbsp;&nbsp; Functional peptides are one kind of short protein fragments that have a wide range of beneficial functions for living organisms. The majority of previous research focused on mono-functional peptides, but a growing number of multi-functional peptides have been discovered. Although enormous experimental efforts endeavor to assay multi-functional peptides, only a small fraction of millions of known peptides have been explored. Effective and precise techniques for identifying multi-functional peptides can facilitate their discovery and mechanistic understanding. In this article, we presented a novel method, called iMFP-LG, for identifying multi-functional peptides based on protein language models (pLMs) and graph attention networks (GATs). Comparison results showed iMFP-LG significantly outperforms state-of-the-art methods on both multi-functional bioactive peptides and multi-functional therapeutic peptides datasets. The interpretability of iMFP-LG was also illustrated by visualizing attention patterns in pLMs and GATs. Regarding to the outstanding performance of iMFP-LG on the identification of multi-functional peptides, we employed iMFP-LG to screen novel candidate peptides with both ACP and AMP functions from millions of known peptides in the UniRef90. As a result, 8 candidate peptides were identified, and 1 candidate that exhibits significant antibacterial and anticancer effect was confirmed through molecular structure alignment and biological experiments. We anticipate iMFP-LG can assist in the discovery of multi-functional peptides and contribute to the advancement of peptide drug design. 

## 🧬 Model Structure
&nbsp;&nbsp;&nbsp;&nbsp; iMFP-LG consists of two modules: peptide representation module and a graph classification module. The peptide sequences are first fed into the pLM to extract high-quality representations, which are then transformed as node features by node feature encoders. The GAT is performed to fine-tune node features by learning the relationship of nodes. Finally, the updated node features are utilized to determine whether the peptides have corresponding function or not through node classifiers. 
<div align=center><img src=img/framework.png></div>

## 💻 Model Structure
The web server is available at: http://bioinformatics.hitsz.edu.cn/iMFP-LG/

## 🚀 Train
```
# Creating a virtual environment
conda activate imfp-lg

# the key elements of 'iMFP-LG' operating environment are listed below:
transformers==4.24.0
torch==1.12.0+cu113 (You can download it from the pytorch(https://pytorch.org/get-started/previous-versions) )
scikit-learn

# Clone this repository
git clone https://github.com/chen-bioinfo/iMFP-LG.git
cd iMFP-LG

# Download the pre-trained protein language models TAPE from the link (https://drive.google.com/drive/folders/1g9ygTC8azNd6S_UnKsTSdcc1wgY90BLk); Put it in the 'iMFP-LG' folder;

cd iMFP-LG/Task
python TaskForMultiPeptide_MFBP.py or python TaskForMultiPeptide_MFTP.py
```

## 🧐 Prediction
```
# Download the pre-trained protein language models TAPE from the link (https://drive.google.com/drive/folders/1g9ygTC8azNd6S_UnKsTSdcc1wgY90BLk); Put it in the 'iMFP-LG' folder;
# Download the model trained on all data(MFBP or MFTP) from link (https://drive.google.com/drive/folders/1g9ygTC8azNd6S_UnKsTSdcc1wgY90BLk);  Put it in the 'iMFP-LG/predict' folder;
tree predict
""" File Structure
predict/
|-- MFBP_alldata
|-- MFTP_alldata
|-- predict_MFBP.py
|-- predict_MFTP.py
"""

# start prediction
cd iMFP-LG/predict
python predict_MFBP.py --seq sequence(eg. FGLPMLSILPKALCILLKRKC)
python predict_MFTP.py --seq sequence(eg. FGLPMLSILPKALCILLKRKC)
```

## ✏️ Citation
If you use this code or our model for your publication, please cite the original paper:
