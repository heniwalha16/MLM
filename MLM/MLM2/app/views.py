from django.shortcuts import render
import fastai
from fastai import *
from fastai.text import *
from fastai.callback import *
import transformers
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path 
from django.http import JsonResponse

from urllib import response
from django.shortcuts import render
from django.http import HttpResponse
import pickle
import json
from rest_framework import serializers, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
import os

import torch
import torch.optim as optim

import random 

# fastai
#from fastai import *
#from fastai.text import *
#from fastai.callback.all import *

# transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=pad_idx).type(input_ids.type()) 
        
        logits = self.transformer(input_ids,
                                  attention_mask = attention_mask)[0]   
        return logits
    
class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.model_max_length
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens
    
class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})


# Create your views here.
@csrf_exempt
@api_view(('POST',))
@action(detail=False, methods=['POST'])
def Scrapping(request):
    if request.method == 'POST':
        dict=json.loads(request.body)
        import requests
        from bs4 import BeautifulSoup
        import imdb

        # create an instance of the IMDb class
        ia = imdb.IMDb()
        movie_title=dict['title']
        # search for the movie by title
        results = ia.search_movie(movie_title)

        # get the first result from the search
        movie = results[0]

        # print the IMDb ID of the movie
        print(movie.getID())
        # Define the URL of the IMDb movie page you want to scrape
        url = f"https://www.imdb.com/title/tt{movie.getID()}/reviews"

        # Send a request to the website and get the HTML content
        response = requests.get(url)
        html_content = response.content

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Find all the review containers on the page
        review_containers = soup.find_all("div", class_="lister-item-content")

        # Set a variable to keep track of the number of reviews
        count = 0
        l=[]
        # Loop through each review container and extract the relevant information
        for container in review_containers:
            # Exit the loop if we have scraped 10 reviews
            if count == 10:
                break
            
            # Extract the review text
            review_text = container.find("div", class_="text").get_text().strip()
            
            # Extract the review rating
            review_rating = container.find("span", class_="rating-other-user-rating").find("span").get_text()
            
            # Print the review text and rating
            print("Review text: ", review_text)
            l.append(review_text)
            print("Review rating: ", review_rating)
            
            # Increment the count of reviews
            count += 1

        learner = load_learner("C:/Users/user/Downloads", file = 'transformer.pkl')

        MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)
}
        model_type = 'roberta'
        pretrained_model_name = 'roberta-base'
        model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
        transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
        pad_idx = transformer_tokenizer.pad_token_id
        
        list=[]
        for i in l:
            list.append(learner.predict(i))
        average=list.mean()
    return JsonResponse({'average':average})