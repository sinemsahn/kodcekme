from keras.models import load_model
import numpy as np
import murmur
import re
import os
from keras.models import Model
from keras import layers

def my_model(input_length=1024):
    # Note that we can name any layer by passing it a "name" argument.
    input = layers.Input(shape=(input_length,), dtype='float32', name='input')

    # We stack a deep densely-connected network on tops
    x = layers.Dense(1024, activation='relu')(input)
    x = layers.normalization.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.normalization.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.normalization.BatchNormalization()(x)

    # And finally we add the last (logistic regression) layer:
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def read_file(sha, dir):
    with open(os.path.join(dir, sha), 'r') as fp:
        file = fp.read()
    return file
# dosya içini okur

def extract_features(sha, path_to_files_dir,
                     hash_dim=1024, split_regex=r"\s+"):
    file = read_file(sha=sha, dir=path_to_files_dir)
    tokens = re.split(pattern=split_regex, string=file)
    token_hash_buckets = [
        (murmur.string_hash(w) % (hash_dim - 1) + 1) for w in tokens
    ]
    token_bucket_counts = np.zeros(hash_dim)
    buckets, counts = np.unique(token_hash_buckets, return_counts=True)
    for bucket, count in zip(buckets, counts):
        token_bucket_counts[bucket] = count
    return np.array(token_bucket_counts)

    #dosyayı alır ve içini okur array dönecek özellikleri

def my_generator( malicious_files,path_to_malicious_files, batch_size, features_length=1024):
    n_samples_per_class = batch_size 
    #zararlı dosya yolları 
    assert len(malicious_files) >= n_samples_per_class
    while True:
        ben_features = [
            extract_features(sha,path_to_files_dir=path_to_malicious_files,hash_dim=1024)
            for sha in np.random.choice(malicious_files, n_samples_per_class,
                                        replace=False)
        ]
        labels = [0 for i in range(n_samples_per_class)]

        idx = np.random.choice(range(batch_size), batch_size)
        ben_features = np.array([np.array(ben_features[i]) for i in idx])
        labels = np.array([labels[i] for i in idx])
        yield ben_features, labels

def make_training_data_generator(features_length, batch_size):
    path_to_training_files='../data'

    training_files= os.listdir(path_to_training_files)

    training_generator=my_generator(
        malicious_files=training_files,
        path_to_malicious_files=path_to_training_files,
        batch_size=batch_size,
        features_length=features_length
    )
    return training_generator

if __name__ == '__main__':
    features_length = 1024
    num_obs_per_epoch = 5000
    batch_size = 128
    model = my_model(input_length=features_length)

    # make the training data generator:
    training_generator = make_training_data_generator(
        batch_size=batch_size,
        features_length=features_length
    )
    # and now train the model:
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_obs_per_epoch / batch_size,
        epochs=10,
        verbose=1)
        # save the model
    model.save('my_model.h5')
    # load the model back into memory from the file:
    same_model = load_model('my_model.h5')  # from keras.models.load_model
    



