

import pickle
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

#  42 landmarks per hand * 2 hands = 84 values
fixed_length = 84  


data_fixed = [sample[:fixed_length] + [0] * (fixed_length - len(sample)) for sample in data]


data_fixed = np.asarray(data_fixed)
labels = np.asarray(labels)

print(data_fixed.shape)  


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


x_train, x_test, y_train, y_test = train_test_split(data_fixed, labels, test_size=0.2, shuffle=True, stratify=labels)


model = RandomForestClassifier()
model.fit(x_train, y_train)


y_predict = model.predict(x_test)


score = accuracy_score(y_predict, y_test)
print(f'{score * 100}% of samples were classified correctly!')


with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
