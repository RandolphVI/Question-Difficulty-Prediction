# Deep Learning for Question Difficulty Prediction
[![Python Version](https://img.shields.io/badge/language-python3.6-blue.svg)](https://www.python.org/downloads/) [![Build Status](https://travis-ci.org/RandolphVI/Question-Difficulty-Prediction.svg?branch=master)](https://travis-ci.org/RandolphVI/Question-Difficulty-Prediction) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/c45aac301b244316830b00b9b0985e3e)](https://www.codacy.com/app/chinawolfman/Question-Difficulty-Prediction?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RandolphVI/Question-Difficulty-Prediction&amp;utm_campaign=Badge_Grade) [![License](https://img.shields.io/github/license/RandolphVI/Question-Difficulty-Prediction.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Issues](https://img.shields.io/github/issues/RandolphVI/Question-Difficulty-Prediction.svg)](https://github.com/RandolphVI/Question-Difficulty-Prediction/issues)

This repository contains my implementations for question difficulty prediction task.

The main objective of the project is to predict the difficulty of each given question based on its context materials which include several components (such like document, question and option in English READING problems).

## Requirements

- Python 3.6
- PyTorch 1.6.0
- Tensorflow 1.15.0
- Tensorboard 1.15.0
- XGBoost 1.0.2
- Sklearn 0.19.1
- Numpy 1.16.2
- Gensim 3.5.0


## Introduction

In the widely used standard test, such as **TOEFL** or **SAT**, examinees are often allowed to retake tests and choose higher scores for college admission. This rule brings an important requirement that we should select test papers with consistent difficulties to guarantee the fairness. Therefore, measurements on tests have attracted much attention.

Among the measurements, one of the most crucial demands is predicting the difficulty of each specific test question, i.e., the percentage of examinees who answer the question wrong. Unfortunately, the ques-
tion difficulty is not directly observable before the test is conducted, and traditional methods often resort to expertise, such as manual labeling or artificial tests organization. Obviously, these human-based solutions are limited in that they are subjective and labor intensive, and the results could also be biased or misleading (we will illustrate this discovery experimentally). 

Therefore, it is an urgent issue to automatically predict question difficulty without manual intervention. Fortunately, with abundant tests recorded by automatic test paper marking systems, test logs of examinees and text materials of questions, as the auxiliary information, become more and more available, which benefits a data-driven solution to this Question Difficulty Prediction (QDP) task, especially for the typical English READING problems. For example, a English READING problem contains a document material and  the several corresponding questions, and each question contains  the corresponding options.

## Project

The project structure is below:

```text
.
├── TMLA(Traditional Machine Learning Algorithms)
│   ├── DTR / LR / SVM / XGBoost
│   └── utils
├── TF(TensorFlow)
│   ├── C-MIDP / R-MIDP / H-MIDP
│   ├── TARNN
│   └── utils
├── PyTorch
│   ├── C-MIDP / R-MIDP / H-MIDP
│   ├── TARNN
│   │   ├── test_tarnn.py
│   │   ├── text_tarnn.py
│   │   └── train_tarnn.py
│   └── utils
│       ├── param_parser.py
│       └── data_helpers.py
├── data
│   ├── word2vec_300.txt [Need Download]
│   ├── Train / Validation /Test_sample.json
│   ├── Train / Validation / Test_BOW_sample.json
│   └── Train / Validation / Test_pairwise_sample.json
├── LICENSE
├── README.md
└── requirements.txt
```

## Data

See data format in `/data` folder which including the data sample files. For example, `train_sample.json` is like:

```json
{"id": "6", "content": ["year", "ruined", "summer", "vacation-a", "two-week", "vacation", "wife", "family", "cabin", "lake", "northern", "ontario", "located", "boundary", "canada-by", "bringing", "modern", "convenience", "wa", "convenient", "good", "ipad", "admiring", "beauty", "nature", "checked", "e-mail", "paddling", "canoe", "twitter", "feed", "devouring", "great", "amusing", "stuck", "workday", "diet", "newspaper", "morning", "wa", "problem", "wa", "behaving", "office", "sticking", "unending", "news", "cycle", "body", "wa", "vacation", "head", "wasnt", "year", "made", "mind", "social", "medium", "experiment", "reverse", "withdrawal", "internet", "manage", "unplug", "knew", "wouldnt", "easy", "im", "good", "self-denial", "wa", "determined", "started", "physical", "restraint", "handing", "ipad", "wife", "helpfully", "announced", "wa", "read", "book", "club", "inclined", "relinquish", "tablet", "moment", "stroke", "luck", "cell", "phone", "signal", "canadian", "cabin", "wa", "spottier", "past", "making", "attempt", "cheating", "experience", "frustration", "wa", "trapped", "forced", "comply", "good", "intention", "largely", "cut", "e-mail", "twitter", "favorite", "newspaper", "website", "connect", "world", "radio-and", "radio", "listen", "choice", "planned", "read", "book", "experienced", "criminal", "plot", "street", "los", "angeles", "cutthroat", "battle", "cancer", "lab", "psyche", "london", "social", "butterfly", "magazine", "read", "im", "claiming", "cut", "internet", "completely", "day", "biked", "nearest", "town", "reward", "sat", "park", "bench", "front", "public", "library", "wi-fi", "back", "cabin", "suffered", "slow", "dial-up", "connection", "day", "check", "e-mail", "tale", "self-denial", "ha", "happy", "ending-for", "determination", "deep", "breathing", "strong", "support", "wife", "succeeded", "vacation", "struggle", "internet", "realizing", "finally", "wa", "ipad", "wa", "problem", "knew", "passed", "starbucks", "wife", "asked", "wanted", "stop", "wi-fi", "dont", "sound", "pleased", "return", "post-vacation", "situation", "test", "begin", "stay", "wagon", "im", "back", "work", "time", "compulsion", "whats", "overwhelming", "crucial", "livelihood", "intention", "giving", "membership", "cult", "immediacy", "hope", "resist", "temptation", "reflexively", "check", "e-mail", "minute", "lead", "long", "im", "checking", "twitter", "feed", "website", "vacation", "supposed", "reset", "brain", "productive", "hoping", "worked"], "question": ["doe", "underlined", "word", "restraint"], "pos_text": ["calm", "controlled", "behavior"], "neg_text": ["relaxing", "move", "strong", "determination", "unshakable", "faith"], "diff": 0.550373134328}
```

- **"id"**: just the id.
- **"content"**: the word segment of the content.
- **"question"**: The word segment of the question.
- **"pos_text"**: The word segment of the correct option.
- **"neg_text"**: The word segment of the wrong options.
- **"diff"**: The difficulty of the question.

### Text Segment

1. You can use `nltk` package if you are going to deal with the English text data.

2. You can use `jieba` package if you are going to deal with the Chinese text data.

### Data Format

This repository can be used in other similiar datasets in two ways:

1. Modify your datasets into the same format of [the sample](https://github.com/RandolphVI/Question-Difficulty-Prediction/tree/master/data).
2. Modify the data preprocessing code in `data_helpers.py`.


Anyway, it should depend on what your data and task are.

### Pre-trained Word Vectors

**You can download the [Word2vec model file](https://drive.google.com/open?id=1QQhm6vKdZmEHaVYvuFbA5Yj6RoVlOhzh) (dim=300). Make sure they are unzipped and under the `/data` folder.**

You can pre-training your word vectors (based on your corpus) in many ways:
- Use `gensim` package to pre-train data.
- Use `glove` tools to pre-train data.
- Even can use a **fasttext** network to pre-train data.

## Usage

See [Usage-TF](https://github.com/RandolphVI/Question-Difficulty-Prediction/blob/master/Usage-TF.md) & [Usage-PyTorch](https://github.com/RandolphVI/Question-Difficulty-Prediction/blob/master/Usage-PyTorch.md).

## Network Structure

Specifically, given the abundant historical test logs and text materials of question (including document, questions and options), we first design a LSTM-based architecture to extract sentence representations for the text materials. Then, we utilize an attention strategy to qualify the difficulty contribution of 1) each word in document to questions, and 2) each word in option to questions.

Considering the incomparability of question difficulties in different tests, we propose a test-dependent pairwise strategy for training TARNN and generating the difficulty prediction value.

![](https://farm8.staticflickr.com/7846/33643949658_9599454fdf_o.png)

The framework of TARNN:

1. The **Input Layer** comprises document representation (TD), question representation (TQ) and option representation (TO). 
2. The **Bi-LSTM Layer** learns the deep comparable semantic representations for text materials. 
3. The **Attention Layer** extracts words of the document (or the option) with high scores as dominant information for a specific question, which is helpful for visualizing the model and improving the performance.
4. Finally the **Prediction Layer** shows predicted difficulty scores of the given READING problem.

## Reference

**If you want to follow the paper or utilize the code, please note the following info in your work:** 

- **Model C-MIDP/R-MIDP/H-MIDP**

```bibtex
@article{佟威2019数据驱动的数学试题难度预测,
  author    = {佟威 and
               汪飞 and
               刘淇 and
               陈恩红},
  title     = {数据驱动的数学试题难度预测},
  journal   = {计算机研究与发展},
  pages     = {1007--1019},
  year      = {2019},
}
```

- **Model TARNN** (modified by TACNN)

```bibtex
@inproceedings{huang2017question,
  author    = {Zhenya Huang and
               Qi Liu and
               Enchong Chen and
               Hongke Zhao and
               Mingyong Gao and
               Si Wei and
               Yu Su and
               Guoping Hu},
  title     = {Question Difficulty Prediction for READING Problems in Standard Tests},
  booktitle = {Thirty-First AAAI Conference on Artificial Intelligence},
  year      = {2017},
}
```

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Ph.D.

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)