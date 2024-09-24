import os
import re
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import pickle
import librosa
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import noisereduce as nr

data = []
# binary classification

# F = Female, M = Male
# hap+exc, ang+fru, xxx+oth
emotions = {
  'ang': 'angry',
  'hap': 'happy',
  'sad': 'sad',
  'neu': 'neutral',
  'fru': 'frustrated',
  'exc': 'excited',
  'fea': 'fearful',
  'sur': 'surprised',
  'dis': 'disgusted',
  'oth': 'other' # 'xxx'?
}
# # 71.43%
# calm_emotion = ['hap', 'neu', 'sur']
# uncalm_emotion = ['ang', 'sad', 'fru', 'fea', 'dis', 'oth', 'xxx', 'exc']

# 71.21%
calm_emotion = ['hap', 'neu']
uncalm_emotion = ['ang', 'sad', 'fru', 'fea', 'dis', 'oth', 'xxx', 'exc', 'sur']

def load_data():
  # TODO
  info = re.compile(r'\[.+\]\n', re.IGNORECASE)
  # for session in range(1, 6):
  for session in [1]:
    file_dir = f'/Users/apple/Desktop/UNSW 2022 T2/COMP9444/Project/code/speech emotion recognition/our-speech-emotion-recognition/code/IEMOCAP_full_release/Session{session}/dialog/EmoEvaluation/'
    files = [file for file in os.listdir(file_dir) if 'Ses' in file]
    # print(files)

    for file in files:
      with open(file_dir + file) as f:
        file_content = f.read()
      
      lines = re.findall(info, file_content)
      for line in lines[1:]:
        time, wav_filename, emo, val_act_dom = line.strip().split('\t')
        start, end = time[1:-1].split(' - ')
        val, act, dom = val_act_dom[1:-1].split(', ')
        if emo in calm_emotion:
          emotion = 'calm'
        else:
          emotion = 'uncalm'
        data.append([start, end, wav_filename, emotion, val, act, dom])

  df_data = pd.DataFrame(data, columns = ['start_time', 'end_time', 'wav_filename', 'emotion', 'val', 'act', 'dom'])
  df_data.to_csv('/Users/apple/Desktop/UNSW 2022 T2/COMP9444/9444Project/zhiqing/df_data.csv')

filename_list = []
emotion_list = []
filename_dir = '/Users/apple/Desktop/UNSW 2022 T2/COMP9444/Project/code/speech emotion recognition/our-speech-emotion-recognition/code/IEMOCAP_full_release/Session1/sentences/wav'
data_df = pd.read_csv('/Users/apple/Desktop/UNSW 2022 T2/COMP9444/9444Project/zhiqing/df_data.csv')
emotion_column = data_df.emotion
filename_column = data_df.wav_filename

def retrieve_data ():
  for file in filename_column:
    dirname = file[:-5]
    filename = f"{filename_dir}/{dirname}/{file}.wav"
    filename_list.append(filename)
  # print(filename_list)
  
  for emotion in emotion_column:
    emotion_list.append(emotion)

def extract_audio_features(wav, sample_rate, mfcc, chroma, mel):
  result = np.array([])
  mel = np.mean(librosa.feature.melspectrogram(y = wav, sr = sample_rate).T, axis = 0)
  result = np.hstack((result, mel))
  return result

def extract_feature():
  a,b = [], []
  for file in filename_list:
    y, sr = librosa.load(file, sr = 16000)
    y_trim, _ = librosa.effects.trim(y, top_db = 20) # trim leading/trailing silence
    reduced_noise = nr.reduce_noise(y = y_trim, sr = sr) # noice reduction
    # plot waveform
    # pd.Series(y_trim).plot(lw = 1, title = 'Raw Audio Trimmed and Reduced Noice')
    # plt.show()
    feature = extract_audio_features(reduced_noise, sr, True, True, True)
    a.append(feature)

  for emotion in emotion_list:
    b.append(emotion)

  # print(f"a.length = {len(a)}, b.length = {len(b)}")
  b = MultiLabelBinarizer().fit_transform(b)

  return train_test_split(a, b, test_size = 0.25, random_state = 9)

def main():
  #load_data()
  retrieve_data()
  train_data, test_data, train_labels, test_labels = extract_feature()
  # print(f"x.length = {len(train_data)}, y.length = {len(train_labels)}")

  # model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
  # accuracy = 67.69%, 69.57%, 69.23%, 66.59%, 70.55%, 69.67%, 68.13%
  # model=MLPClassifier(alpha=0.08, batch_size=64, epsilon=1e-04, hidden_layer_sizes=(50,), learning_rate='constant', max_iter=450)
  # accuracy = 70.33%
  # model=MLPClassifier(alpha=0.08, batch_size=64, epsilon=1e-04, hidden_layer_sizes=(100,), learning_rate='constant', max_iter=450)
  # accuracy = 70.77%
  # model=MLPClassifier(alpha=0.05, batch_size=64, epsilon=1e-04, hidden_layer_sizes=(100,), learning_rate='constant', max_iter=450)
  # accuracy = 70.55%
  # model=MLPClassifier(alpha=0.10, batch_size=64, epsilon=1e-04, hidden_layer_sizes=(100,), learning_rate='constant', max_iter=450)
  # accuracy = 69.23%
  # model=MLPClassifier(alpha=0.09, batch_size=64, epsilon=1e-04, hidden_layer_sizes=(50,), learning_rate='constant', max_iter=450)
  # accuracy = 70.33%, 70.99%, 71.43%
  # model=MLPClassifier(alpha=0.09, batch_size=64, epsilon=1e-04, hidden_layer_sizes=(100,), learning_rate='constant', max_iter=450, early_stopping=False, warm_start=True)
  # accuracy = 71.87%
  # model=MLPClassifier(alpha=0.09, batch_size=64, epsilon=1e-04, hidden_layer_sizes=(100,), learning_rate='adaptive', max_iter=450, early_stopping=False)
  # accuracy = 70.99%, 70.77%, 71.65%
  # model = MLPClassifier(alpha=0.08, batch_size=64, epsilon=1e-04, hidden_layer_sizes=(100,), 
  #                       learning_rate='adaptive', max_iter=450, early_stopping=True)
  # accuracy = 70.77%, 71.65%, 71.21%
  #model = MLPClassifier(alpha=0.01, solver='lbfgs', activation='relu',epsilon=1e-04,hidden_layer_sizes=(50,50), max_iter=5000,learning_rate='adaptive', early_stopping=True)
  # accuracy = 99.63%, 63.08%
  model = MLPClassifier(alpha=0.01, solver='lbfgs', activation='relu',epsilon=1e-04,hidden_layer_sizes=(50,50), max_iter=5000,learning_rate='adaptive', early_stopping=True)

  #train_predictions = model.predict(train_data)
  model.fit(train_data, train_labels)
  test_predictions = model.predict(test_data)
  train_score = model.score(train_data, train_labels) * 100
  test_score = model.score(test_data, test_labels) * 100
  print(f"Train accuracy = {train_score:.2f}%")
  print(f"Test accuracy = {test_score:.2f}%")
  print(f"loss = {model.loss_}")
  #plt.plot(model.loss_curve_, label = 'Loss Curve')
  #plt.plot(model.validation_scores_, label = 'Accuracy Score')
  #plt.title("Loss Curve and Accuracy Score", fontsize = 14)
  #plt.xlabel("iteration (epoch)")
  # plt.ylabel("training error")
  #plt.legend(loc='upper left')
  #plt.show()

  print(f"\n------\nClassification Report about accuracy on test data\n------\n")
  print(classification_report(test_labels, test_predictions))

if __name__ == '__main__':
    main()