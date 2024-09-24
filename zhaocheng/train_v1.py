import librosa
from sklearn.preprocessing import MultiLabelBinarizer
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

wave_open_path = '/Users/apple/Desktop/UNSW 2022 T2/COMP9444/Project/code/speech emotion recognition/our-speech-emotion-recognition/code/IEMOCAP_full_release/Session1/dialog/wav'
wave_path_list = os.listdir(wave_open_path)

emotion_file_open_path = '/Users/apple/Desktop/UNSW 2022 T2/COMP9444/Project/code/speech emotion recognition/our-speech-emotion-recognition/code/IEMOCAP_full_release/Session1/dialog/EmoEvaluation/evaluation'
emotion_file_path_list = os.listdir(emotion_file_open_path)

def extract_feature(wav,sample_rate,mfcc, chroma, mel):
    result=np.array([])
    '''
    if chroma:
        stft=np.abs(librosa.stft(wav))
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    '''
    if mel:   
        mel=np.mean(librosa.feature.melspectrogram(wav, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

emotions={
  '01':'Neutral',
  '02':'Excited',
  '03':'Frustration',
  '04':'Sadness',
  '05':'Angry',
  '06':'Fear',
  '07':'Disgust',
  '08':'Surprise',
  '09':'Happiness'
}
observed_emotions=[]

def load_data(test_size=0.2):
    x,y=[],[]
    for file in wave_path_list:
        file_path= wave_open_path + '/' + file
        wav, sr_ret = librosa.load(file_path, sr=16000)
        feature = extract_feature(wav, sr_ret, mfcc=True, chroma=True, mel=True)
        x.append(feature)

    for file in emotion_file_path_list:
        emotion_list = []
        file_path = emotion_file_open_path + '/' + file
        if (file == '.DS_Store'):
            continue
        f = open(file_path,'r')
        lines = f.readlines()
        for line in lines:
            list = line.split()
            for word in list:
                for key,emotion in emotions.items():
                    if (word[:-1] == emotion):
                        emotion_list.append(key)
        y.append(emotion_list)
        #print(y)
    y=MultiLabelBinarizer().fit_transform(y)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# 获取测试和训练集   
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

# 初始化一个MLPClassifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
# 拟合/训练模型
model.fit(x_train,y_train)

# 预测测试集的值
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

print(f"Loss = {model.loss_:.2f}%")
    
