import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Determine the maximum length
max_length = max(len(d) for d in data)

# Pad sequences
def pad_sequence(seq, max_length):
    return seq + [0] * (max_length - len(seq))

data_padded = [pad_sequence(d, max_length) for d in data]

# Convert to numpy arrays
data_padded = np.asarray(data_padded)
labels = np.asarray(labels)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict
y_predict = model.predict(x_test)

# Accuracy score
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

f.close()
