"""
Contains the data cleaning/preprocessing functions.
This file can be run as a standalone script.
"""

import re
import pandas as pd
from config import TESTING_FILE, TESTING_FILE_ORIGINAL, TRAINING_FILE, TRAINING_FILE_ORIGINAL
import preprocessor as p
import contractions
from textblob import TextBlob 
from nltk.tokenize import TweetTokenizer 
from wordsegment import load, segment
load()

# set up punctuations we want to be replaced
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

def fix_spellings(line):
  tmp = TextBlob(line)

  # stemming vs lemmatization
  return ' '.join(w.lemmatize() for w in tmp.words)

"""
Clean a given DataFrame's response column by passing the tweets through several
cleaning functions.
"""
def clean_tweets(df):
  tempArr = []
  elements_to_remove = ['<URL>', '@USER']
  pattern = '|'.join(elements_to_remove)
  tk = TweetTokenizer()

  # set what we want to remove using tweet processor lib
  p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.MENTION)
  for line in df:

    # remove special character that caused issues when contracting words
    tmpL = re.compile("\"").sub("", line)
    tmpL = re.compile("\s’\s").sub("'", tmpL)

    # remove digits
    def remove_digits(w):
      return ''.join([i for i in w if not i.isdigit()])
    tmpL = ' '.join([remove_digits(w) for w in tmpL.split(' ')])

    tmpL = re.sub(pattern, '', tmpL)
    tmpL = re.sub('\.', '', tmpL)

    # send to tweet_processor
    tmpL = ' '.join([w for w in tmpL.split(' ')])
    tmpL = p.clean(tmpL)

    # hashtag segmentation
    tmpH = []
    for w in tk.tokenize(tmpL):
      if w.startswith('#'): 
        w = ' '.join(segment(w))
      tmpH.append(w)
    tmpL = ' '.join(tmpH)

    # expand word contractions
    tmpL = contractions.fix(tmpL)

    # remove punctuation
    tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
    tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)

    # lemmatize using TextBlob (NLTK didn't do a great job)
    tmpL = fix_spellings(tmpL)

    tempArr.append(tmpL)
  return tempArr

"""
Append the full context as a single sentence.
"""
def append_context(df, idx):
  cl = str(df["context"][idx])
  cl = "".join(cl.strip('][').split(','))
  return cl

"""
Main function that performs the preprocessing:
  1. Enumerate `response` column of both training and test DataFrames
  2. Append `context` to the `response` column
  3. Perform the data cleaning
  4. Output train.csv and test.csv files
"""
def run():
  train = pd.read_json(TRAINING_FILE_ORIGINAL, lines=True)
  test = pd.read_json(TESTING_FILE_ORIGINAL, lines=True)

  # clean and output training file
  for (idx,l) in enumerate(train["response"]):
    cl = append_context(train, idx)
    train["response"][idx] = cl+' '+str(l)
  train_tweets = clean_tweets(train["response"])
  train_tweets = pd.DataFrame(train_tweets)
  train['tweet'] = train_tweets
  write_data(train, TRAINING_FILE, ['label', 'tweet'])

  # clean and output test file
  for (idx,l) in enumerate(test["response"]):
    cl = append_context(test, idx)
    test["response"][idx] = cl+' '+str(l)

  test_tweets = clean_tweets(test["response"])
  test_tweets = pd.DataFrame(test_tweets)
  test["tweet"] = test_tweets
  write_data(test, TESTING_FILE, ['id', 'tweet'])

def write_data(df, file_path, selected_columns):
  df.to_csv(file_path, index=False, columns=selected_columns)

if __name__ == "__main__":
  run()