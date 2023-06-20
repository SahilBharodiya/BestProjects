import numpy as np
import torch
from keras.utils import pad_sequences
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel


PIL = input("Enter the path of the petition: ")

with open(PIL, 'r') as f:
    text = f.read()

print(text)


def input_id_maker(sen, tokenizer):
    input_ids = []
    lengths = []

    sen = tokenizer.tokenize(sen, add_prefix_space=True)
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    if (len(sen) > 510):
        sen = sen[len(sen)-510:]

    sen = [CLS] + sen + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

    input_ids = pad_sequences(
        input_ids, maxlen=512, value=0, dtype="long", truncating="pre", padding="post")
    return input_ids, lengths


model = "pytorch_model.bin"
config = "config.json"
vocab = "vocab.json"
merges = "merges.txt"

tokenizer = RobertaTokenizer(vocab, merges)
model = RobertaForSequenceClassification.from_pretrained(model, config=config)

input_ids, lengths = input_id_maker(text, tokenizer)
input_ids = torch.tensor(input_ids)
print("*================================================================================================================================*")

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()

    if np.argmax(logits) == 0:
        print('Petition Rejected')
    else:
        print('Petition Accepted')

print("*================================================================================================================================*")

if np.argmax(logits) == 0:
    print("No need to recommend similar Petitions")
    exit()

else:
    data = pd.read_csv('ILDC_multi.csv')
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    data_tfidf = tfidf.transform(data['text'])
    petetion = tfidf.transform([text])

    cosine_sim = linear_kernel(petetion, data_tfidf)
    cosine_sim_dict = {i: cosine_sim[0][i] for i in range(len(cosine_sim[0]))}
    cosine_sim_dict = sorted(cosine_sim_dict.items(),
                             key=lambda x: x[1], reverse=True)

    top10 = cosine_sim_dict[:10]

    print("Top 10 similar Petitions: ")
    for i in range(10):
        text = data.iloc[top10[i][0]]['text']
        print(f'Petition{i}: {text[:100]}... with similarity {top10[i][1]}')
        with open(f'recommanded_petitions/Petition{i}.txt', 'w') as f:
            f.write(text)
        print()
    print("Top 10 similar Petitions saved in 'recommanded_petitions' folder")
    print("*================================================================================================================================*")
