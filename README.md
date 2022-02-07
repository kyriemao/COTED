# COTED: Curriculum Contrastive Context Denoising for Few-shot Conversational Dense Retrieval
This is a temporary anonymous repository of the paper "Curriculum Contrastive Context Denoising for Few-shot Conversational Dense Retrieval""

![image](https://github.com/kyriemao/COTED/edit/main/overview.pdf)

## Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data
We provide two raw and preprocessed CAsT datasets in the *datasets* folder. Besides, the human annotation data is in the *annotation_data* folder.


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

