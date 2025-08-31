#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
import json

# Load sarcasm dataset
sarcasm_path = "/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json"

with open(sarcasm_path, "r") as f:
    sarcasm_data = [json.loads(line) for line in f]

sarcasm_df = pd.DataFrame(sarcasm_data)

print(sarcasm_df.head())
print(sarcasm_df["is_sarcastic"].value_counts())



# In[10]:


import os

air_path = "/kaggle/input/air-quality-data-in-india"
print(os.listdir(air_path))


# In[11]:


import pandas as pd

air_file = "/kaggle/input/air-quality-data-in-india/city_day.csv"
air_df = pd.read_csv(air_file)

print(air_df.head())
print(air_df.columns)


# In[1]:


# Step 1: Import libraries
import pandas as pd

# Step 2: Load the News Headlines Sarcasm Dataset
# Make sure the dataset CSV is uploaded to Kaggle under /kaggle/input/news-headlines-dataset-for-sarcasm-detection/
file_path = "/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json"

# Dataset is in JSON format, one headline per line
df = pd.read_json(file_path, lines=True)

# Step 3: Peek at the dataset
print("Dataset shape:", df.shape)
print("Columns:", df.columns)
df.head()


# In[2]:


import random

# Step 2: Define AQI ranges and sarcastic remark templates
aqi_ranges = {
    "0-50": [
        "Wow, fresh air… almost tastes like lungs’ vacation.",
        "AQI is low, lungs are on holiday, enjoy the free oxygen!"
    ],
    "51-150": [
        "Mild poison is normal, right? Just a whiff of smog.",
        "Air is slightly toxic today, but who cares?"
    ],
    "151-200": [
        "Breathing this is basically instant smoker vibes.",
        "Air so polluted, even my mask is giving up."
    ],
    "201-300": [
        "Don’t need coffee, just inhale the morning smog.",
        "AQI is high, who needs clean air anyway?"
    ],
    "301+": [
        "Air so toxic, even zombies would think twice.",
        "Step outside, become one with the smog apocalypse."
    ]
}

# Step 3: Generate synthetic dataset
synthetic_data = []

for i in range(300):  # generate 300 samples
    aqi_level = random.randint(0, 350)
    
    if aqi_level <= 50:
        remark = random.choice(aqi_ranges["0-50"])
    elif aqi_level <= 150:
        remark = random.choice(aqi_ranges["51-150"])
    elif aqi_level <= 200:
        remark = random.choice(aqi_ranges["151-200"])
    elif aqi_level <= 300:
        remark = random.choice(aqi_ranges["201-300"])
    else:
        remark = random.choice(aqi_ranges["301+"])
    
    synthetic_data.append({
        "AQI": aqi_level,
        "remark": remark
    })

# Step 4: Convert to DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Step 5: Save CSV (optional)
synthetic_df.to_csv("/kaggle/working/synthetic_aqi_sarcasm.csv", index=False)

synthetic_df.head()


# In[3]:


import random
import pandas as pd

# Step 1: Define AQI ranges
aqi_ranges = {
    "0-50": "good",
    "51-150": "moderate",
    "151-200": "unhealthy",
    "201-300": "very unhealthy",
    "301+": "hazardous"
}

# Step 2: Define sentence fragments for dynamic remarks
starts = [
    "Breathe in…", "Step outside…", "Air today is", "Good luck with",
    "Lungs, prepare for", "Congrats on"
]

middles = {
    "good": ["fresh air", "almost perfect oxygen", "lungs’ vacation", "clean breeze"],
    "moderate": ["slightly toxic air", "mild poison", "casual smog", "lung seasoning"],
    "unhealthy": ["polluted haze", "instant smoker vibes", "smog overload", "toxic clouds"],
    "very unhealthy": ["lung torture session", "apocalypse air", "extreme smog", "dangerous fumes"],
    "hazardous": ["air so toxic even zombies complain", "death soup", "lethal smog", "lungs filing complaints"]
}

ends = [
    "enjoy it!", "if you dare.", "your lungs will thank you… maybe.", 
    "instant regret guaranteed.", "survival mode activated.", "just another day outside."
]

# Step 3: Generate synthetic dataset
synthetic_data = []

for i in range(300):  # generate 300 samples
    aqi_level = random.randint(0, 350)
    
    # Determine AQI category
    if aqi_level <= 50:
        category = "good"
    elif aqi_level <= 150:
        category = "moderate"
    elif aqi_level <= 200:
        category = "unhealthy"
    elif aqi_level <= 300:
        category = "very unhealthy"
    else:
        category = "hazardous"
    
    # Build dynamic remark
    remark = f"{random.choice(starts)} {random.choice(middles[category])}, {random.choice(ends)}"
    
    synthetic_data.append({
        "AQI": aqi_level,
        "remark": remark
    })

# Step 4: Convert to DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Step 5: Save CSV (optional)
synthetic_df.to_csv("/kaggle/working/synthetic_aqi_sarcasm_unique.csv", index=False)

synthetic_df.head(10)


# In[4]:


get_ipython().system('pip install transformers datasets')

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling


# In[7]:


# Load base sarcasm dataset (headlines)
sarcasm_df = pd.read_json("/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)
sarcasm_df = sarcasm_df[sarcasm_df['is_sarcastic']==1]  # only sarcastic examples
sarcasm_texts = sarcasm_df['headline'].tolist()

# Load synthetic AQI sarcasm dataset
aqi_df = pd.read_csv("/kaggle/working/synthetic_aqi_sarcasm_unique.csv")
aqi_texts = aqi_df['remark'].tolist()

# Combine both datasets
all_texts = sarcasm_texts + aqi_texts
dataset = Dataset.from_dict({"text": all_texts})


# In[8]:


# Show first 10 examples
print("Total samples:", len(dataset))
print("First 10 examples:")
for i, item in enumerate(dataset['text'][:10]):
    print(f"{i+1}: {item}")


# In[9]:


tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have a pad token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# In[12]:


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2 is causal LM
)


# In[13]:


model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.resize_token_embeddings(len(tokenizer))  # resize if we add tokens


# In[20]:


training_args = TrainingArguments(
    output_dir="/kaggle/working/gpt2-aqi-sarcasm",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    run_name="AQI_sarcasm_training",
    fp16=True,
    report_to="none"
)


# In[21]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
trainer.train()


# In[26]:


prompt = "AQI is 180:"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(
    inputs, 
    max_length=50, 
    do_sample=True, 
    top_k=50, 
    top_p=0.95, 
    temperature=0.8
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# In[23]:


training_args = TrainingArguments(
    output_dir="/kaggle/working/gpt2-aqi-sarcasm-quick",
    overwrite_output_dir=True,
    num_train_epochs=0.5,   # half an epoch
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    run_name="AQI_sarcasm_quick",
    fp16=True,
    report_to="none"
)


# In[24]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()


# In[25]:


import os

# Check if the model directory exists
model_dir = "/kaggle/working/gpt2-aqi-sarcasm"  # change if your output_dir is different
if os.path.exists(model_dir):
    print("Model directory exists! ✅")
    # List contents
    print("Contents of model directory:")
    print(os.listdir(model_dir))
else:
    print("Model directory not found. 😢")


# In[27]:


import shutil

# Path to your model directory
model_dir = "/kaggle/working/gpt2-aqi-sarcasm"

# Path for the zip file
zip_path = "/kaggle/working/gpt2-aqi-sarcasm.zip"

# Zip the folder
shutil.make_archive(base_name=zip_path.replace('.zip',''), format='zip', root_dir=model_dir)

print(f"Model zipped successfully! You can download it here: {zip_path}")


# In[28]:


from IPython.display import FileLink

# Path to your zip file
zip_path = "/kaggle/working/gpt2-aqi-sarcasm.zip"

# Create a clickable download link
FileLink(zip_path)

