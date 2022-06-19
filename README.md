# Sentiment-Analysis
 Trained with over 60,000 dataset to categorize positive and negative reviews

![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Table of contents

- [Installation](#installation)
- [Usage example](#usage-example)
- [Adding new languages](#adding-new-languages)
- [Adding and overwriting words](#adding-and-overwriting-words)
- [API Reference](#api-reference)
- [How it works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Validation](#validation)
- [Testing](#testing)

# Step by step
1) To launch the app, clone the project and execute it
2) Key your information

# In a Nutshell   
In a nutshell here's how to use this template, so **for example** assume you want to implement ResNet-18 to train mnist, so you should do the following:
- In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model

```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 
- Statics
```python
CSV_URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
```


- Step 1 - Load Data
```python
df = pd.read_csv(CSV_URL)
df_copy = df.copy() #a copy to prevent wasting time loading the data later
```


- Step 2 - Data Inspection/Visualization
```python
df.head(10)
df.tail(10)
df.info() #cuz we deal with string not number
df.describe()

df['sentiment'].unique() #to get the unique target
df['review'][0] #a positive review for index 0
df['sentiment'][0] 

df.duplicated().sum() #There is 418 duplicates
df[df.duplicated()]
```

- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**

## This is to insert code in readme
` print('hello') `
` print('hi my name is khai') `

## To include url link

![markdown_badges]('[https://rahuldkjain.github.io/gh-profile-readme-generator/](https://github.com/Ileriayo/markdown-badges)')
[url_to_cheat_sheet](https://rahuldkjain.github.io/gh-profile-readme-generator/)

## Model architecture
![model_architecture](static/model_plot.png)
