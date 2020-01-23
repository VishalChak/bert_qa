import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model

download_model(model='bert-squad_1.1', dir='./models')


# Download pdf files from BNP Paribas public news
def download_pdf():
    import os
    import wget
    directory = './data/pdf/'
    models_url = [
      'https://invest.bnpparibas.com/documents/1q19-pr-12648',
      'https://invest.bnpparibas.com/documents/4q18-pr-18000',
      'https://invest.bnpparibas.com/documents/4q17-pr'
    ]

    print('\nDownloading PDF files...')

    if not os.path.exists(directory):
        os.makedirs(directory)
    for url in models_url:
        wget.download(url=url, out=directory)

#download_pdf()


df = pdf_converter(directory_path='./data/pdf/')
df.head()


cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib', max_df=1.0)

# Fit Retriever to documents
cdqa_pipeline.fit_retriever(df=df)

#query = 'Who is Komal Rajput?.'

while(True):
  print("Enter ur question")
  query = str(input())
  prediction = cdqa_pipeline.predict(query)
  
  print('query: {}'.format(query))
  print('answer: {}'.format(prediction[0]))
  print('title: {}'.format(prediction[1]))
  print('paragraph: {}'.format(prediction[2]))