import os
import networkx
from networkx.drawing.nx_pydot import write_dot
import itertools
from keras.models import Model

from keras.models import load_model
from keras import layers
import os

def pecheck(fullpath):
    return open(fullpath,"rb").read(2) == b"MZ"

def getstrings(fullpath): 
    strings = os.popen("strings '{0}'".format(fullpath)).read()
    strings = set(strings.split("\n"))
    return strings

def featurextrandnode(malware_paths):
    malware_attributes={}
    labels={}
    count=10
    for dirpath, dirnames, filenames in os.walk(malware_paths):
        for filename in filenames:
            path = os.path.join(dirpath,filename)
            if(pecheck(path)):
                attributes = getstrings(path)
                print("Extracted {0} attributes from {1} ...".format(len(attributes),path))
                malware_attributes[path]= attributes
                labels[path]= path # bu adı olacak 
    return malware_attributes, labels

def make_training_data_generator():
    path_to_training = "/content/malware_data_science/ch4/data"
    
    training_generator = featurextrandnode(
        malware_paths=path_to_training
    )
    #bu da gider aldığı pathlerden özellikleri çıkartır etiketler ve döner
    return training_generator

def my_model(input_length=1024):
    # Note that we can name any layer by passing it a "name" argument.
    input = layers.Input(shape=(input_length,), dtype='float32', name='input')

    # We stack a deep densely-connected network on tops
    x = layers.Dense(1024, activation='relu')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # And finally we add the last (logistic regression) layer:
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    #output değişmeli
    
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


features_length = 1024
num_obs_per_epoch = 5000
batch_size = 128

model = my_model(input_length=features_length)

training_generator = make_training_data_generator()

model.fit(
    training_generator,
    steps_per_epoch=num_obs_per_epoch / batch_size,
    epochs=10,
    verbose=1)
    
  #  deneme_path=""
  #  string_deneme =getstrings(deneme_path)
    # sonra buna label gerekiyorsa artık ve sonra predicte vereceğiz


#diğeriyle karşılaştır
#video ile modeli düzelt

#model ve predict