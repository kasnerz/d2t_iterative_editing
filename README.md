# Data-to-Text Generation with Iterative Text Editing

The code for replicating the experiments described in: 

- Zdeněk Kasner & Ondřej Dušek (2020): [Data-to-Text Generation with Iterative Text Editing.](https://www.aclweb.org/anthology/2020.inlg-1.9/) In: *Proceedings of the 13th International Conference on Natural Language Generation (INLG 2020)* 

The method allows generating text from RDF triples by iteratively applying sentence fusion on the templates.

## Model Overview
![overview](model.png)

## Quickstart

1. Install the requirements
```
pip install -r requirements.txt
```
2. Download the datasets and models:
```
./download_datasets_and_models.sh
```

3. Run the *WebNLG* experiments including preprocessing the data, training the model and decoding the data: 
```
./run.sh
```
4. See the results in `experiments/webnlg_full/100` # TODO

## Usage Instructions

### Requirements
- Python 3 + pip
- packages
  - Tensorflow 1.15* (GPU version recommended)
  - PyTorch 🤗 Transformers
  - other packages specified in `requirements.txt`

All packages can be installed using
```
pip install -r requirements.txt
```
Select `tensorflow-1.15` instead of `tensorflow-1.15-gpu` if you wish not to use the GPU.


**The original implementation of LaserTagger based on BERT and Tensorflow 1.x was used in the experiments. PyTorch version of LaserTagger is currently in progress.*

### Dependencies
All datasets and models can be downloaded using the command: 
```
download_datasets_and_models.sh
```

The following is the description of the dependencies (datasets, models and external repositiories) which are downloaded by the script. The script does not download the dependencies which are already located in their respective path.

#### Datasets
- [WebNLG dataset](https://github.com/ThiagoCF05/webnlg) (v1.4)
- [Cleaned E2E Challenge dataset](https://github.com/tuetschek/e2e-cleaning)
- [DiscoFuse dataset](https://github.com/google-research-datasets/discofuse) (balanced Wikipedia part)

#### External Repositories
- [LaserTagger](https://github.com/kasnerz/lasertagger) - fork of the original LaserTagger implementation featuring a few changes for integration with the model

#### Models
- [BERT](https://github.com/google-research/bert) - original TensorFlow implementation from Google (utilized by [LaserTagger](https://github.com/google-research/lasertagger))
- Additionally, [LMScorer](https://github.com/simonepri/lm-scorer) requires [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html) downloaded automatically by 🤗 Transformers.

### Preprocessing
Preprocessing involves parsing the original data-to-text datasets and extracting examples for training the sentence fusion model.

Things you may want to consider:
- **The templates** for the predicates are included in the repository. In order to re-generate simple templates for WebNLG and double templates for E2E, use the flag `--force-generate-templates`. However, note that double templates for E2E have been manually denoised.
  - **TODO FLAG**
- **The mode** for selecting the lexicalizations (`--mode`) can be set to `full`, `best_tgt` or `best`. 
  - The default mode is `full`.
  - Modes `best_tgt` and `best`  use *LMScorer*  and can be sped up by using GPU (see `--lms_device`). 
  - Modes are described in the supplementary material.
- **A custom dataset** based on RDF triples **can be used** for the experiments. Doing so requires editing `datasets.py`, adding a custom class derived from `Dataset` and overriding relevant methods.

## Citation
```
@inproceedings{kasner-dusek-2020-data,
    title = "Data-to-Text Generation with Iterative Text Editing",
    author = "Kasner, Zden{\v{e}}k  and
      Du{\v{s}}ek, Ond{\v{r}}ej",
    booktitle = "Proceedings of the 13th International Conference on Natural Language Generation",
    month = dec,
    year = "2020",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.inlg-1.9",
    pages = "60--67"
}
```