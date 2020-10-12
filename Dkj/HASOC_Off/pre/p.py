sentiment=' '
str1=[ ]
k=201
with open('E:/复现/BAKSA_IITK-master/HASOC_Off/pre/pred.csv', 'r') as out:
    for line in out:
        #line = line.strip().split(' ')
        if line == 0:
            sentiment = 'Off'
        else:
            sentiment = 'Not'
        str1 = str(k) + ',' + str(sentiment)
        k+=1
        print(str1)
        fd=open('E:/复现/BAKSA_IITK-master/HASOC_Off/pre/pred1.txt')


import pandas as pd
from sklearn.metrics import classification_report
test = pd.read_csv('C:/Users/dkj/Desktop/task1翻译/dev201-400标签.txt')#测试集标签
preds = pd.read_csv('E:/复现/BAKSA_IITK-master/HASOC_Off/pre/pred1.txt')#预测标签
results = {'preds': classification_report(test['Sentiment'], preds['Sentiment'], labels=['Off', 'Not'], output_dict=True, digits=6)}

formatted_results = [['model', 'precision', 'recall', 'accuracy', 'f1-score']]
for ki in results.keys():
    scores = results[ki]['macro avg']
    model = [ki, scores['precision'], scores['recall'], results[ki]['accuracy'], scores['f1-score']]
    formatted_results.append(model)

formatted_results = pd.DataFrame(formatted_results[1:], columns=formatted_results[0])
print(formatted_results)