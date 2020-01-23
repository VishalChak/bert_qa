import os
import torch
import joblib
from cdqa.reader import BertProcessor, BertQA
from cdqa.utils.download import download_squad

dataroot = "D:/datasets/sqad/"
download_squad(dir=dataroot)

train_processor = BertProcessor(do_lower_case=True, is_training=True)
train_examples, train_features = train_processor.fit_transform(X=os.path.join(dataroot, "SQuAD_1.1", "train-v1.1.json"))


reader = BertQA(train_batch_size=12,
                learning_rate=3e-5,
                num_train_epochs=2,
                do_lower_case=True,
                output_dir='models')

reader.fit(X=(train_examples, train_features))


reader.model.to('cpu')
reader.device = torch.device('cpu')


joblib.dump(reader, os.path.join(reader.output_dir, 'bert_qa.joblib'))