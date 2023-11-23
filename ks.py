import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from numpy import argmax
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from tqdm import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import cv2
def cnn_model(input_data):
    image_dir='tumordataset/tumordata/'
    no_tumor_images=os.listdir(image_dir+ '/no')
    yes_tumor_images=os.listdir(image_dir+ '/yes')
    print('The length of NO Tumor images is',len(no_tumor_images))
    print('The length of Tumor images is',len(yes_tumor_images))
    print("--------------------------------------\n")
    dataset=[]
    label=[]
    img_siz=(128,128)
    for i , image_name in tqdm(enumerate(no_tumor_images),desc="No Tumor"):
        if(image_name.split('.')[1]=='jpg'):
            image=cv2.imread(image_dir+'/no/'+image_name)
            image=Image.fromarray(image,'RGB')
            image=image.resize(img_siz)
            dataset.append(np.array(image))
            label.append(0)        
    for i ,image_name in tqdm(enumerate(yes_tumor_images),desc="Tumor"):
        if(image_name.split('.')[1]=='jpg'):
            image=cv2.imread(image_dir+'/yes/'+image_name)
            image=Image.fromarray(image,'RGB')
            image=image.resize(img_siz)
            dataset.append(np.array(image))
            label.append(1)
        dataset=np.array(dataset)
        label = np.array(label)
        print("--------------------------------------\n")
        print('Dataset Length: ',len(dataset))
        print('Label Length: ',len(label))
        print("--------------------------------------\n")
        print("--------------------------------------\n")
        print("Train-Test Split")
        x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
        x_train=tf.keras.utils.normalize(x_train,axis=1)
        x_test=tf.keras.utils.normalize(x_test,axis=1)
        model=tf.keras.models.Sequential([
               tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
               tf.keras.layers.MaxPooling2D((2,2)),
               tf.keras.layers.Flatten(),
               tf.keras.layers.Dense(256,activation='relu'),
               tf.keras.layers.Dropout(.5),
               tf.keras.layers.Dense(512,activation='relu'),
               tf.keras.layers.Dense(1,activation='sigmoid')
          ])
        model.summary()

        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        history=model.fit(x_train,y_train,epochs=5,batch_size =128,validation_split=0.1)
        plt.plot(history.epoch,history.history['accuracy'], label='accuracy')
        plt.plot(history.epoch,history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.savefig('C:/Users/Admin/Downloads/DL-ALGORITHMS-main/streamlit/tumor_detection/results/tumor_sample_accuracy_plot.png')
        plt.clf()
        plt.plot(history.epoch,history.history['loss'], label='loss')
        plt.plot(history.epoch,history.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig('C:/Users/Admin/Downloads/DL-ALGORITHMS-main/streamlit/tumor_detection/results/tumor_sample_loss_plot.png')
        loss,accuracy=model.evaluate(x_test,y_test)
        st.write(f'Accuracy: {round(accuracy*100,2)}')
        print("--------------------------------------\n")
        y_pred=model.predict(x_test)
        y_pred = (y_pred > 0.5).astype(int)
        st.write('classification Report\n',classification_report(y_test,y_pred))
        print("--------------------------------------\n")
        def make_prediction(img,model):
             img=cv2.imread(img)
             img=Image.fromarray(img)
             img=img.resize((128,128))
             img=np.array(img)
             input_img = np.expand_dims(img, axis=0)
             res = model.predict(input_img)
             if res:
                  print("Tumor Detected")
             else:
                  print("No Tumor")
def dnn_model(input_data):
    print("Loading Dataset generation Intiated..")
    path = 'https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/iris.csv'
    df = pd.read_csv(path, header=None)
    X=df.values[:,:-1]
    y=df.values[:, -1]
    X = X.astype('float')
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    n_features = X_train.shape[1]
    output_class = 3
    model = Sequential()
    model.add(Dense(64, activation='relu',input_shape=(n_features,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_class,activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("Training Started.")
    history=model.fit(X_train, y_train, epochs=200, batch_size=16)
    loss, acc = model.evaluate(X_test, y_test)
    print(f'Test Accuracy:{round(acc*100)}')
    row = [9.1,7.4,5.4,6.2]
    prediction = model.predict([row])
    print('Predicted: %s (class=%d)' % (prediction, argmax(prediction)))
def rnn_model(input_data):
    print("---------------------- Downloading Dataset -------------------------\n")
    dataset = pd.read_csv('https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/SMSSpamCollection',sep='\t',names=['label','message'])
    print(dataset.head())
    print("----------------------  -------------------------")
    print(dataset.groupby('label').describe())
    print("----------------------  -------------------------")
    dataset['label'] = dataset['label'].map( {'spam': 1, 'ham': 0} )
    X = dataset['message'].values
    y = dataset['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    tokeniser = tf.keras.preprocessing.text.Tokenizer()
    tokeniser.fit_on_texts(X_train)
    encoded_train = tokeniser.texts_to_sequences(X_train)
    encoded_test = tokeniser.texts_to_sequences(X_test)
    print(encoded_train[0:2])
    print("----------------------  Padding  -------------------------\n")
    max_length = 10
    padded_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
    padded_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
    print(padded_train[0:2])
    vocab_size = len(tokeniser.word_index)+1
    model=tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size,output_dim= 24, input_length=max_length),
        tf.keras.layers.SimpleRNN(24, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
      ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)
    model.fit(x=padded_train,y=y_train,epochs=50,validation_data=(padded_test, y_test),callbacks=[early_stop])
    def c_report(y_true, y_pred):
        print("Classification Report")
        print(classification_report(y_true, y_pred))
        acc_sc = accuracy_score(y_true, y_pred)
        print(f"Accuracy : {str(round(acc_sc,2)*100)}")
        return acc_sc
    def plot_confusion_matrix(y_true, y_pred):
        mtx = confusion_matrix(y_true, y_pred)
        sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5, cmap="Blues", cbar=False)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("C:/Users/Admin/Downloads/DL-ALGORITHMS-main/streamlit/results/test.jpg")
        preds = (model.predict(padded_test) > 0.5).astype("int32")
        c_report(y_test, preds)
        plot_confusion_matrix(y_test, preds)

def lstm_model(input_data):
    texts="Hello world"
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_length=100
    padded_sequences = sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
    tf.random.set_seed(7)
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=64)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
def perceptron_model(input_data):
    def __init__(self,learning_rate=0.01, epochs=100,activation_function='step'):
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_epochs = epochs
        self.activation_function = activation_function
    def activate(self, x):
        if self.activation_function == 'step':
            return 1 if x >= 0 else 0
        elif self.activation_function == 'sigmoid':
            return 1 if (1 / (1 + np.exp(-x)))>=0.5 else 0
        elif self.activation_function == 'relu':
            return 1 if max(0,x)>=0.5 else 0
    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.random.randint(n_features, size=(n_features))
        for epoch in tqdm(range(self.max_epochs)):
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]
                weighted_sum = np.dot(inputs, self.weights) + self.bias
                prediction = self.activate(weighted_sum)
        print("Training Completed")
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            inputs = X[i]
            weighted_sum = np.dot(inputs, self.weights) + self.bias
            prediction = self.activate(weighted_sum)
            predictions.append(prediction)
        return predictions

def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    if res:
        print("Tumor Detected")
    else:
        print("No Tumor")
    

def backpropagation_model(input_data):
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    learning_rate = 0.01
    max_epoch = 5000
    N = y_train.size
    input_size = X_train.shape[1]
    hidden_size = 2
    output_size = 3
    np.random.seed(10)
    W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
    W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def mean_squared_error(y_pred, y_true):
        y_true_one_hot = np.eye(output_size)[y_true]
        y_true_reshaped = y_true_one_hot.reshape(y_pred.shape)
        error = ((y_pred - y_true_reshaped)**2).sum() / (2*y_pred.size)
        return error
    def accuracy(y_pred, y_true):
        acc = y_pred.argmax(axis=1) ==  y_true.argmax(axis=1)
        return acc.mean()
    results = pd.DataFrame(columns=["mse", "accuracy"])
    for epoch in tqdm(range(max_epoch)):
        Z1 = np.dot(X_train, W1)
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2)
        A2 = sigmoid(Z2)
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(np.eye(output_size)[y_train], A2)
    new_row = pd.DataFrame({"mse": [mse], "accuracy": [acc]})
    results = pd.concat([results, new_row], ignore_index=True)
    E1 = A2 - np.eye(output_size)[y_train]
    dW1 = E1 * A2 * (1 - A2)
    E2 = np.dot(dW1, W2.T)
    dW2 = E2 * A1 * (1 - A1)
    W2_update = np.dot(A1.T, dW1) / N
    W1_update = np.dot(X_train.T, dW2) / N
    W2 = W2 - learning_rate * W2_update
    W1 = W1 - learning_rate * W1_update
    results.mse.plot(title="Mean Squared Error")
    plt.show()
    results.accuracy.plot(title="Accuracy")
    plt.show()
    Z1 = np.dot(X_test, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    test_acc = accuracy(np.eye(output_size)[y_test], A2)
    print("Test accuracy: {}".format(test_acc))
st.title("Machine learning in Streamlit")
task_type=st.sidebar.selectbox("Select task:",["Image Classification","Sentiment Analysis"])
uploaded_image=None
if task_type=="Image Classification":
    uploader=st.file_uploader("Select file",type=["jpg", "jpeg", "png"])
text_input=None
if task_type=="Sentiment Analysis":
    text_input=st.text_area("Enter text for sentiment analysis:")
selected_model=st.sidebar.radio("Select Model:", ["Perceptron", "Backpropagation", "CNN", "DNN", "RNN", "LSTM"])
if st.button("Run Predictions"):
    if task_type=="Image Classification" and uploaded_image is not None:
            try:
                img=cv2.imread(Image)
                img=Image.fromarray(img)
                img=img.resize((128,128))
                img=np.array(img)
                input_img = np.expand_dims(img, axis=0)
                if selected_model=="CNN":
                    predictions=cnn_model(input_img).make_prediction(input_img,cnn_model(input_img))
                    st.image(input_img, caption='Uploaded Image', use_column_width=True)
                    st.subheader(f"{selected_model} Image Classification Prediction:")
                    st.write(predictions)
            except Exception as e:
                 st.error(f"Error processing image:{e}")
elif task_type=="Sentiment Analysis" and text_input is not None:
        print(text_input)
        if selected_model == "Perceptron":
                try:
                    prediction = perceptron_model(text_input)
                    st.write(prediction)
                except IndexError as e:
                    print("IndexError:", e)
        elif selected_model=="Backpropagation":
                try:
                    prediction = backpropagation_model(text_input)
                    st.write(prediction)
                except IndexError as e:
                     print("IndexError:",e)
        elif selected_model=="DNN": 
            try:
                    prediction = dnn_model(text_input)
                    st.write(prediction)
            except IndexError as e:
                     print("IndexError:",e)
        elif selected_model=="RNN":
                try:
                    prediction = rnn_model(text_input)
                    st.write(prediction)
                except IndexError as e:
                     print("IndexError:",e)
        elif selected_model=="LSTM":
            try:
                prediction = lstm_model(text_input)
                st.write(prediction)
            except IndexError as e:
                     print("IndexError:",e)
        st.warning("Text input is empty.")
        st.subheader(f"{selected_model} Sentiment Analysis Prediction:")

    