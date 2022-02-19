# COTED: Curriculum Contrastive Context Denoising for Few-shot Conversational Dense Retrieval
This is a temporary anonymous repository of the paper "Curriculum Contrastive Context Denoising for Few-shot Conversational Dense Retrieval""

![image](overview.pdf)

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data
We provide two raw and preprocessed CAsT datasets in the *datasets* folder. Besides, the human annotation data is in the *annotation_data* folder. **Please note that, although there are part of turn dependency annotations in the original dataset of CAsT 20, we find that it is not very accurate. Therefore, we refine the original annnotation by our team.**


## Main Files
- train.py: curriculum_sampling, two-step multi-task learning
- test.py: test with Faiss
- my_utils.py: useful functions
- models.py: CQE model architecture (i.e., ANCE)
- db_lib.py: data strctures, conversational data augmentation, curriculum_sampling
- running scripts:
  - train_cast19.sh
  - test_cast19.sh
  - train_cast20.sh
  - test_cast20.sh

## Training
First download the public pre-trained ANCE model to the *checkpoints* folder.
```bash
mkdir checkpoints
wget https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip
wget https://data.thunlp.org/convdr/ad-hoc-ance-orquac.cp
unzip Passage_ANCE_FirstP_Checkpoint.zip
mv "Passage ANCE(FirstP) Checkpoint" ad-hoc-ance-msmarco
```

To train our COTED, run the following scripts.
```bash
# params: training_epoch, aug_ratio, loss_weight

# CAsT-19
bash train_cast19.sh 6 2 0.01

# CAsT-20
bash train_cast20.sh 6 3 0.01
```

## Testing
For testing, you should first generate passages embeddings. One can refer to https://github.com/thunlp/ConvDR for generating these embeddings.
The passages embeddings are expected to stored at *./datasets/collections/cast_shared/passage_embeddings*.

Then, run the following scripts for testing. 
```bash
# param: test_epoch

# CAsT-19
bash test_cast19.sh 6 
or
bash test_cast19.sh final

# CAsT-20
bash test_cast20.sh 6
or
bash test_cast20.sh final
```

