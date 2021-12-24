# ilk olarak malware detection deep learning olup
# verileri 0 1 olarak etiketleyecek

from keras.models import load_model
import numpy as np
import murmur
import re
import os
from keras.models import Model
from keras import layers

def read_file(sha, dir):
    with open(os.path.join(dir, sha), 'r') as fp:
        file = fp.read()
    return file

def extract_file(file, hash_dim=1024, split_regex=r"\s+"):

    tokens = re.split(pattern=split_regex, string=file)
    token_hash_buckets = [
        (murmur.string_hash(w) % (hash_dim - 1) + 1) for w in tokens
    ]
    token_bucket_counts = np.zeros(hash_dim)
    buckets, counts = np.unique(token_hash_buckets, return_counts=True)
    for bucket, count in zip(buckets, counts):
        token_bucket_counts[bucket] = count
    return np.array(token_bucket_counts)



def extract_features(sha, path_to_files_dir,
                     hash_dim=1024, split_regex=r"\s+"):

    file = read_file(sha=sha, dir=path_to_files_dir)

    tokens = re.split(pattern=split_regex, string=file)
    # 1024 1024 olarak alir okur
    token_hash_buckets = [
        (murmur.string_hash(w) % (hash_dim - 1) + 1) for w in tokens
    ]
    token_bucket_counts = np.zeros(hash_dim)
    buckets, counts = np.unique(token_hash_buckets, return_counts=True)
   #onu array olarak aliyor diyelim
    for bucket, count in zip(buckets, counts):
        token_bucket_counts[bucket] = count
    return np.array(token_bucket_counts)

def my_generator(benign_files, malicious_files,
                 path_to_benign_files, path_to_malicious_files,
                 batch_size, features_length=1024):
    n_samples_per_class = batch_size / 2
    assert len(benign_files) >= n_samples_per_class
    assert len(malicious_files) >= n_samples_per_class
    while True:
     
        ben_features = [
            extract_features(sha, path_to_files_dir=path_to_benign_files,
                             hash_dim=features_length)
            for sha in np.random.choice(benign_files, n_samples_per_class,
                                        replace=False)
        ]

        mal_features = [
            extract_features(sha, path_to_files_dir=path_to_malicious_files,
                             hash_dim=features_length)
            for sha in np.random.choice(malicious_files, n_samples_per_class,
                                        replace=False)
        ]
    
        all_features = ben_features + mal_features

        labels = [0 for i in range(n_samples_per_class)] + [1 for i in range(
            n_samples_per_class)]

        idx = np.random.choice(range(batch_size), batch_size)
        all_features = np.array([np.array(all_features[i]) for i in idx])
        labels = np.array([labels[i] for i in idx])
        yield all_features, labels



def make_training_data_generator(features_length, batch_size):
    path_to_training_benign_files = '~/Desktop/verilerim/egitim/normal/benignware/'
    path_to_training_malicious_files = '~/Desktop/verilerim/egitim/zararli/'

    train_benign_files = os.listdir(path_to_training_benign_files)
    train_malicious_files = os.listdir(path_to_training_malicious_files)

    training_generator = my_generator(
        benign_files=train_benign_files,
        malicious_files=train_malicious_files,
        path_to_benign_files=path_to_training_benign_files,
        path_to_malicious_files=path_to_training_malicious_files,
        batch_size=batch_size,
        features_length=features_length
    )
    return training_generator
    # egitim verilerini alir ozelliklerinin arrayi ve etiketlemelerini doner


def my_model(input_length=1024):
    #model olusturuyor diyelim
    input = layers.Input(shape=(input_length,), dtype='float32', name='input')

    x = layers.Dense(1024, activation='relu')(input)
    x = layers.normalization.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.normalization.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.normalization.BatchNormalization()(x)

    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



if __name__ == '__main__':
    features_length = 1024
    num_obs_per_epoch = 5000
    batch_size = 128

    model = my_model(input_length=features_length)

    training_generator = make_training_data_generator(
        batch_size=batch_size,
        features_length=features_length
    )

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_obs_per_epoch / batch_size,
        epochs=10,
        verbose=1)
    model.save('my_model.h5')
    
    same_model = load_model('my_model.h5') 

    ###modeli predict ettirmeli ilk sonra devam etmeli
    #predict icin kullanicidan aldigi veriyi ayni forma cevirip 
#ynew = model.predict_classes(Xnew)
# show the inputs and predicted outputs
#for i in range(len(Xnew)):
#	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
#extract_featuresa gonderip predict dersek
    file_path=''
    sha=''
   
    file_features = [
            extract_file(sha, path_to_files_dir=file_path,
                             hash_dim=features_length)
        ]
    label_file = model.predict_classes(file_features)

    for i in range(len(file_features)):
        print("%s, Predicted= %s" % (file_features[i],label_file[i]))
