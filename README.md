# CS410 - Text Classification Competition

## Overview
We participated in the Text Classification Competition for Sarcasm Detection in Tweets. Our team beat the baseline (0.723) and achieved an F1 score of `0.7542963307013469`.

The code can be used for training a preprocessing the given dataset (train.jsonl and test.jsonl) and train a BERT model. The usage of our solution can be found in the "Source Code Walkthrough section".

## Our Team: AmazingS3
-   Suraj	Bisht	      surajb2@illinois.edu (Team Leader)
-   Sithira	Serasinghe	sithira2@illinois.edu
-   Santosh	Kore	      kore3@illinois.edu

---
## Source Code Walkthrough

## 1. Prerequisites
 - Anaconda 1.9.12
 - Python 3.8.3
 - PyTorch 1.7.0
 - Transformers 3.0.0
  
## 2. Install dependencies

Make sure to run this program in an Ananconda environment (i.e. Conda console). This has been tested on *nix and Windows systems.

**1. Libs**
```bash
pip install tweet-preprocessor textblob wordsegment contractions tqdm
````

**2. Download TextBlob corpora**
```bash
python -m textblob.download_corpora
```

**3. Install PyTorch & Transformers**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch transformers
```

If it complains that the `transformers` lib's not installed, try this command:
```bash
conda install -c conda-forge transformers
```

## 3. Usage
First, `cd src` and run the following commands,

**tl;dr**
```bash
python clean.py && python train.py && python eval.py
```

This will preprocess, train and generate the `answer.txt` model which can be then submitted to the grader for evaluation.

**Description of each step:**
1. Clean the dataset
   `python clean.py`

2. Train the model
   `python train.py`

   Once the model is trained it will create an `input/model.bin` file which saves our model to a binary file. We can later load this file (in the evaluation step) to make predictions.

3. Make predictions & create the answer.txt file
   `python eval.py`
   The answer.txt file is created at the `output` folder.

The following section describes each of these steps in-depth.

## Data Cleaning / Preprocessing
We perform data cleaning steps for both `train.jsonl` and `test.jsonl` so that they are normalized for training and evaluation purposes. The algorithm for cleaning the data is as follows:

For each tweet:
1. Append all `context` to become one sentence and prefix it to the `response`.
2. Fix the tweet if it has special characters to support better expansion of contractions.
3. Remove all digits from the tweets.
4. Remove `<URL>` and `@USER` as they do not add any value.
5. Convert all tweets to lowercase.
6. Use NLTK's tweet processor to remove emojis, URLs, smileys, and '@' mentions
7. Do hashtag segmentation to expand any hashtags to words.
8. Expand contracted words.
9. Remove all special symbols.
10. Perform lemmatization on the words.

## Model Training
A model can be built and trained with the provided parameters by issuing a `python train.py` command. The following steps are run in sequence during the model training.
1. Read in the train.csv from the prior step.
2. Training dataset (5000 records) is split into training and validation as 80:20 ratio.
3. Feed in the parameters to the model.
4. Perform model training for the given number of epochs.
5. Calculate validation accuracy for each run and save the best model as a bin file


## Tuning the model
The following can be considered as parameters that could be optimized to achieve a better result.

**src/config.py**

```python
DEVICE = "cpu" # If you have CUDA GPU, change this to 'cuda'
MAX_LEN = 256  # Max length of the tokens in a given document
EPOCHS = 5 # Number of epochs to train the model for
BERT_PATH = "bert-base-uncased"  # Our base BERT model. Can plug in different models such as bert-large-uncased
TRAIN_BATCH_SIZE = 8 # Size of the training dataset batch
VALID_BATCH_SIZE = 4 # Size of the validation dataset batch
```

**src/train.py**
```python
L25: test_size=0.15 # Size of the validation dataset
L69: optimizer = AdamW(optimizer_parameters, lr=2e-5) # A different optimizer can be plugging or a learning rate can be defined here
L71: num_warmup_steps=2 # No. of warmup steps that need to run before the actual training step
```

**src/model.py**
```python
L13: nn.Dropout(0.1) # Configure the dropout value
```

## Evaluation of the model
A high-level view of the sequence of operations run during the evaluation step is as follows.

1. Load the test.csv file from the data transformation step.
2. Load the best performing model from the training step.
3. Perform predictions for each test tweet (1800 total records)
4. Generate answer.txt that will be submitted to the grader to the "output" folder.

---

## Contributions of the team members
-   Suraj	Bisht	      surajb2@illinois.edu (Team Leader)
    -   Improve the initial coding workflow (Google Colab, Local setup etc.).
    -   Investigating Sequential model, Logistic Regression, SVC etc.
    -   Investigating `bert-base-uncased` model.
    -   Investigating data preprocessing options.
    -   Hyperparameter tuning to improve the current model.
-   Sithira	Serasinghe	sithira2@illinois.edu
    -   Setting up the initial workflow.
    -   Investigating LSTM/BiDirectional LSTM, Random Forest etc.
    -   Investigating various data preprocessing options.
    -   Investigating `bert-base-uncased` model.
    -   Hyperparameter tuning to improve the current model.
-   Santosh	Kore	      kore3@illinois.edu
    -   Improve the initial coding workflow (Google Colab, Local setup etc.).
    -   Investigating Sequential models, SimpleRNN, CNN etc.
    -   Investigating `bert-large-uncased` model.
    -   Investigating data preprocessing options.
    -   Hyperparameter tuning to improve the current model.


## Future Enhancements

1. Cleaning data further with different methods.
2. Optimizing BERT model parameters and trying different BERT model (eg. RoBERTa)
3. Re-use some of the tried models and optimizing to beat F1 scores.
4. Extract Emoji's to add more meaning to the sentiments of the tweets.
5. Data augmentation steps to prevent overfitting.
6. Try an ensemble of models (eg. BERT + VLaD etc. )
7. Run our model on different test data and compare results against state-of-art.


## References/Credits

The usage of BERT model is inspired by https://github.com/abhishekkrthakur/bert-sentiment