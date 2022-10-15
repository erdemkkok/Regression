import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight','Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset=raw_dataset.copy()

dataset.isna().sum()# Veri setinde bulunan none degerleri True dondurur.sum() fonksiyonu ise bunu toparlar.

dataset = dataset.dropna()#Verisetinde bulunan none değerleri atar.

dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'}) #Origin sütununu labellama işlemi yapar.

dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')#Origin sütununu 3 e bölerek label'a göre 1  ve 0 a böler.
#print(dataset.tail())# Verilerin son satırlarını ekrana bastırır.
train_dataset = dataset.sample(frac=0.8, random_state=0)#Veriyi karıp %80'lik kısmını almaktadır.

test_dataset = dataset.drop(train_dataset.index)#Train dataset verisini index numarasına göre atmaktadır. Kalanlar test dataseti olmaktadır.

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')#Ekrana çizdirme fonksiyonudur.
#plt.show()
#Transpose ekrana daha sade bastırılması için alınmıştır.
#Burada modeldeki verileri yorumlamak önemlidir. max kısmına bakılarak verilerin alakasız olduğu yorumlandıktan sonra normalizasyon işlemi yapılmalıdır.
print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')#verideki MPG verisini siler.
test_labels = test_features.pop('MPG')#verideki MPG verisini siler.

print("------------------")
print(train_dataset.describe().transpose()[['mean', 'std']])# Mean ve std aldtında verileri basar
normalizer = tf.keras.layers.Normalization(axis=-1)#Normalizasyon için değerleri üretir.
normalizer.adapt(np.array(train_features))#Train data için normalizasyon değerleri döner
print("------------------")
print(normalizer.mean.numpy())
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)#ilk feature değerini gösterir
  print()
  print('Normalized:', normalizer(first).numpy())#Normalleşmiş feature değerini gösterir.
horsepower = np.array(train_features['Horsepower'])#horsepower sütununu np array'e çevirir.

horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)

horsepower_normalizer.adapt(horsepower)
test_results = {}
def build_and_compile_model(norm):
    model = keras.Sequential([
    norm,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
dnn_horsepower_model.summary()
history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)
x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(test_features['Horsepower'], test_labels,verbose=0)