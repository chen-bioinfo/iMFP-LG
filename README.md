# iMFP-BG
📋Identifying multi-functional peptides using protein language model and graph attention network

## 📘 Abstract
In this study, we provide a multi-label framework using protein language model and graph attention network for identifying multi-functional peptides. Pre-trained protein language model is used to extract fine-grained features of different functional peptides from peptide sequences. We transformed the multi-label problem to graph node classification and the graph attention network is used to learn the relationship between different labels. We conducted a series of experiments on two datasets of multi-functional bioactive peptides(MFBP) and multi-functional therapeutics peptides(MFTP) to validate our framework. The result demonstrates our model greatly outperforms other methods and we believe it can be effectively expanded to other multi-label studies in bioinformatics.

## 🧬 Model Structure
<div align=center><img src="https://github.com/chen-bioinfo/iMFP-LG/tree/master/img/framework.png" width="700" /></div>

## 🚀 Train
```
# Creating a virtual environment
conda activate imfp-lg

# the key elements of 'iMFP-LG' operating environment are listed below:
transformers==4.24.0
torch==1.12.0+cu113 (You can download it from the [pytorch](https://pytorch.org/get-started/previous-versions/) )
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
