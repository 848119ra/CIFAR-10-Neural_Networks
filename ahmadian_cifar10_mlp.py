import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Define a list to store data and labels from all data batch files
all_X = []
all_Y = []

all_X_test = []
all_Y_test = []

for i in range(1,6):
    datadict = unpickle(f'D:/university/Tampere/courses/3rd/Intro to Pattern Recognition & Machine Learning/exercise5/cifar-10-python/cifar-10-batches-py/data_batch_{i}')
    datadict_test = unpickle('D:/university/Tampere/courses/3rd/Intro to Pattern Recognition & Machine Learning/exercise5/cifar-10-python/cifar-10-batches-py/test_batch')

    X = datadict["data"]
    Y = datadict["labels"]

    X_test = datadict_test["data"]
    Y_test = datadict_test["labels"]

    all_X.append(X)
    all_Y.extend(Y)

    all_X_test.append(X_test)
    all_Y_test.extend(Y_test)

# Concatenate data from all batches
X = np.array(all_X)
Y = np.array(all_Y)

X_test = np.array(all_X_test)
Y_test = np.array(all_Y_test)


labeldict = unpickle('D:/university/Tampere/courses/3rd/Intro to Pattern Recognition & Machine Learning/exercise5/cifar-10-python/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(50000, 3072)
Y = np.array(Y)


#print(X.shape)
#for i in range(X.shape[0]):
    # Show some images randomly
    #if random() > 0.999:
     #   plt.figure(1);
      #  plt.clf()
       # plt.imshow(X[i])
        #plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        #plt.pause(1)


# Convert labels to one-hot encoding using a for loop for training data
num_classes = 10
Y_one_hot = np.zeros([Y.shape[0], 10])
for i, label in enumerate(Y):
    one_hot = [0] * num_classes
    one_hot[label] = 1
    Y_one_hot[i] = one_hot

# Convert labels to one-hot encoding using a for loop for test data
num_classes = 10
Y_one_hot_test = np.zeros([Y_test.shape[0], 10])
for i, label in enumerate(Y_test):
    one_hot_test = [0] * num_classes
    one_hot_test[label] = 1
    Y_one_hot_test[i] = one_hot_test

# Now Y_one_hot contains one-hot encoded labels.

# Example usage:
# To access the one-hot encoded label for the first image:
# Y_one_hot[0]

# To access the one-hot encoded label for the fifth image:
# Y_one_hot[4]

# To access the one-hot encoded label for label 2:
# Y_one_hot[2]
print("One-Hot Encoded Labels for the First Few Images for training data:")
for i in range(5):  # Print the first 5 images' one-hot encoded labels
    print(Y_one_hot[i])

print("One-Hot Encoded Labels for the First Few Images for test data:")
for i in range(5):  # Print the first 5 images' one-hot encoded labels
    print(Y_one_hot_test[i])



# Define and Compile the Neural Network:
# Create a Sequential model
model = Sequential()

# Add a dense layer with 500 neurons (you can adjust this)
model.add(Dense(500, input_dim=3072, activation='sigmoid'))

# Add a dense layer with 500 neurons (you can adjust this)
model.add(Dense(500, activation='sigmoid'))

# Add the output layer with 10 neurons (one for each CIFAR-10 class)
model.add(Dense(10, activation='softmax'))

# Define the optimizer (e.g., SGD) and compile the model
opt = keras.optimizers.SGD(lr=0.01)  # You can adjust the learning rate
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

# Train the Model:
# Train the model
# adjust the number of epochs
history = model.fit(X, Y_one_hot, epochs=100, verbose=1)

# Test training data
y_tr_pred = np.empty(Y_one_hot.shape)
y_tr_pred_2 = np.squeeze(model.predict(X))
for pred_ind in range(y_tr_pred_2.shape[0]):
    if y_tr_pred_2[pred_ind][0] > y_tr_pred_2[pred_ind][1]:
        y_tr_pred[pred_ind] = 1
    else:
        y_tr_pred[pred_ind] = 2

tot_correct = len(np.where(y_tr-y_tr_pred == 0)[0])
print(f'Classication accuracy (training data): {tot_correct/len(y_tr)*100}%')

# Test test data
y_te_pred = np.empty(y_te.shape)
y_te_pred_2 = np.squeeze(model.predict(x_te))
for pred_ind in range(y_te_pred_2.shape[0]):
    if y_te_pred_2[pred_ind][0] > y_te_pred_2[pred_ind][1]:
        y_te_pred[pred_ind] = 1
    else:
        y_te_pred[pred_ind] = 2

tot_correct = len(np.where(y_te-y_te_pred == 0)[0])
print(f'Classication accuracy (test data): {tot_correct/len(y_te)*100}%')



# Plot Training Loss Curve:
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()





