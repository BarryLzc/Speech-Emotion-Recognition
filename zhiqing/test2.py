import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import noisereduce as nr
import seaborn as sns

data = []
# 71.21%
calm_emotion = ['hap', 'neu']
uncalm_emotion = ['ang', 'sad', 'fru', 'fea', 'dis', 'oth', 'xxx', 'exc', 'sur']

def load_data ():
    info = re.compile(r'\[.+\]\n', re.IGNORECASE)
    for session in [1]:
        file_dir = f'IEMOCAP/Session{session}/dialog/EmoEvaluation/'
        files = [file for file in os.listdir(file_dir) if 'Ses' in file]

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
    df_data.to_csv('df_data.csv')

filename_list = []
emotion_list = []
filename_dir = 'IEMOCAP/Session1/sentences/wav'
data_df = pd.read_csv('df_data.csv')
emotion_column = data_df.emotion
filename_column = data_df.wav_filename

def retrieve_data ():
    for file in filename_column:
        dirname = file[:-5]
        filename = f'{filename_dir}/{dirname}/{file}.wav'
        filename_list.append(filename)

    for emotion in emotion_column:
        emotion_list.append(emotion)

def extract_audio_features (wav, sample_rate):
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
        feature = extract_audio_features(reduced_noise, sr)
        a.append(feature)

    for emotion in emotion_list:
        b.append(emotion)

    b = MultiLabelBinarizer().fit_transform(b)
    return train_test_split(a, b, test_size = 0.25, random_state = 9)

def get_confusion_matrix(confusion_matrix, axes, class_names):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names,)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    axes.set_xlabel('Predicted label')
    axes.set_ylabel('True label')

def main():
    load_data()
    retrieve_data()
    train_data, test_data, train_labels, test_labels = extract_feature()

    # train accuracy = 77.05%, test accuracy = 71.43%, loss=0.98%
    model = MLPClassifier(solver='adam', alpha=0.13, batch_size=16, epsilon=1e-01, hidden_layer_sizes=(50, 30, 10), 
                        random_state=1, learning_rate='adaptive', learning_rate_init=0.001, max_iter=450, 
                        verbose=True, early_stopping=False)

    model.fit(train_data, train_labels)
    test_predictions = model.predict(test_data)

    # print accuracy scores
    train_score = model.score(train_data, train_labels) * 100
    test_score = model.score(test_data, test_labels) * 100
    print(f"Train accuracy = {train_score:.2f}%")
    print(f"Test accuracy = {test_score:.2f}%")
    print(f"Loss = {model.loss_:.2f}%")

    plt.plot(model.loss_curve_, label = 'Loss Curve')
    plt.title("Loss Curve", fontsize = 14)
    plt.xlabel("iteration (epoch)")
    plt.show()

    # print classification report
    print(f'\n----------------------------------------------------\n Classification Report about accuracy on test data \n')
    print(classification_report(test_labels, test_predictions))

    matrix = multilabel_confusion_matrix(test_labels, test_predictions)

    # plot confusion matrix
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for axes, c_matrix in zip(ax.flatten(), matrix):
        get_confusion_matrix(c_matrix, axes, ["Calm", "Uncalm"])
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()