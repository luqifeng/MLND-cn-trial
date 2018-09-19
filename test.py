## �벻Ҫ�޸��·�����
# ����������
import json
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# �������ݼ�
train_filename='train.json'
train_content = pd.read_json(codecs.open(train_filename, mode='r', encoding='utf-8'))

test_filename = 'test.json'
test_content = pd.read_json(codecs.open(test_filename, mode='r', encoding='utf-8'))
    
# ��ӡ���ص����ݼ�����
print("�������ݼ�һ������ {} ѵ������ �� {} ����������\n".format(len(train_content), len(test_content)))
if len(train_content)==39774 and len(test_content)==9944:
    print("���ݳɹ����룡")
else:
    print("�������������⣬�����ļ�·����")
    
## �벻Ҫ�޸��·�����
pd.set_option('display.max_colwidth',120)

### TODO����ӡtrain_content��ǰ5������������Ԥ������
train_content.head(5)

## �벻Ҫ�޸��·�����
## �鿴�ܹ���Ʒ����
categories=np.unique(train_content['cuisine'])
print("һ������ {} �ֲ�Ʒ���ֱ���:\n{}".format(len(categories),categories))

### TODO����������Ŀ������ֱ�ֵ
train_ingredients = train_content['ingredients']
train_targets = train_content['cuisine']

### TODO: ��ӡ���������Ƿ���ȷ��ֵ
print(train_ingredients.head(5))
print(train_targets.head(5))

## TODO: ͳ�����ϳ��ִ���������ֵ��sum_ingredients�ֵ���

result = {}
for x in train_ingredients:
    for y in x:
        if y in result:
            result[y] = result[y] + 1
        else:
            result[y] = 1
top_10 = dict(sorted(result.items(), key=lambda d:d[1], reverse = True)[0:10]).keys()
print(top_10)
sum_ingredients = result

## �벻Ҫ�޸��·�����
# Finally, plot the 10 most used ingredients
plt.style.use(u'ggplot')
fig = pd.DataFrame(sum_ingredients, index=[0]).transpose()[0].sort_values(ascending=False, inplace=False)[:10].plot(kind='barh')
fig.invert_yaxis()
fig = fig.get_figure()
fig.tight_layout()
fig.show()

## TODO: ͳ���������ϵ�����ϳ��ִ���������ֵ��italian_ingredients�ֵ���
italian_ingredients = []
data_set = train_content[train_content["cuisine"]=='italian']['ingredients']
#print(data_set)
result = {}
for x in data_set:
    for y in x:
        if y in result:
            result[y] = result[y] + 1
        else:
             result[y] = 1
top_10 = dict(sorted(result.items(), key=lambda d:d[1], reverse = True)[0:10]).keys()
print(top_10)
italian_ingredients = result

## �벻Ҫ�޸��·�����
# Finally, plot the 10 most used ingredients
fig = pd.DataFrame(italian_ingredients, index=[0]).transpose()[0].sort_values(ascending=False, inplace=False)[:10].plot(kind='barh')
fig.invert_yaxis()
fig = fig.get_figure()
fig.tight_layout()
fig.show()

## �벻Ҫ�޸��·�����
import re
from nltk.stem import WordNetLemmatizer
import numpy as np

def text_clean(ingredients):
    #ȥ�����ʵı����ţ�ֻ���� a..z A...Z�ĵ����ַ�
    ingredients= np.array(ingredients).tolist()
    print("��Ʒ���ϣ�\n{}".format(ingredients[9]))
    ingredients=[[re.sub('[^A-Za-z]', ' ', word) for word in component]for component in ingredients]
    print("ȥ��������֮��Ľ����\n{}".format(ingredients[9]))

    # ȥ�����ʵĵ�������ʱ̬��ֻ�������ʵĴʸ�
    lemma=WordNetLemmatizer()
    ingredients=[" ".join([ " ".join([lemma.lemmatize(w) for w in words.split(" ")]) for words in component])  for component in ingredients]
    print("ȥ��ʱ̬�͵�����֮��Ľ����\n{}".format(ingredients[9]))
    return ingredients

print("\n����ѵ����...")
train_ingredients = text_clean(train_content['ingredients'])
print("\n������Լ�...")
test_ingredients = text_clean(test_content['ingredients'])

## �벻Ҫ�޸��·�����
from sklearn.feature_extraction.text import TfidfVectorizer
# ������ת������������

# ���� ѵ����
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1),
                analyzer='word', max_df=.57, binary=False,
                token_pattern=r"\w+",sublinear_tf=False)
train_tfidf = vectorizer.fit_transform(train_ingredients).todense()

## ���� ���Լ�
test_tfidf = vectorizer.transform(test_ingredients)

## �벻Ҫ�޸��·�����
train_targets=np.array(train_content['cuisine']).tolist()
train_targets[:10]

# ����Ҫͨ��head()������Ԥ��ѵ����train_tfidf,train_targets����
print(train_tfidf[0:2].tolist())
print(train_targets[0:10])

### TODO�����ֳ���֤��

from sklearn.model_selection import train_test_split

X_train , X_valid , y_train, y_valid = train_test_split(train_tfidf,train_targets,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

## TODO: �����߼��ع�ģ��
parameters = {'C':range(1,11)}  

classifier = LogisticRegression()

grid = GridSearchCV(classifier,parameters)


## �벻Ҫ�޸��·�����
grid = grid.fit(X_train, y_train)

## �벻Ҫ�޸��·�����
from sklearn.metrics import accuracy_score ## ����ģ�͵�׼ȷ��

valid_predict = grid.predict(X_valid)
valid_score=accuracy_score(y_valid,valid_predict)

print("��֤���ϵĵ÷�Ϊ��{}".format(valid_score))

### TODO��Ԥ����Խ��
predictions = grid.predict(test_tfidf)

## �벻Ҫ�޸��·�����
print("Ԥ��Ĳ��Լ�����Ϊ��{}".format(len(predictions)))
test_content['cuisine']=predictions
test_content.head(10)

## ���ؽ����ʽ
submit_frame = pd.read_csv("sample_submission.csv")
## ������
result = pd.merge(submit_frame, test_content, on="id", how='left')
result = result.rename(index=str, columns={"cuisine_y": "cuisine"})
test_result_name = "tfidf_cuisine_test.csv"
result[['id','cuisine']].to_csv(test_result_name,index=False)