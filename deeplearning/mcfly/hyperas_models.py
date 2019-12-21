import numpy as np
from hyperas.distributions import choice, uniform

def data():
  data_info = json.load(open('tmp/data_info.csv'))
  nsamp = data_info['nsamp']; seqlen = data_info['seqlen']; nchannel = data_info['nchannel']
  nclass = data_info['nclass']
  X_train = np.memmap('tmp/X_aug.np', dtype='float32', mode='r', shape=(nsamp, seqlen, nchannel))
  y_train = np.memmap('tmp/y_aug.np', dtype='int32', mode='r', shape=(nsamp, nclass))
  
  # Use only a third of the data
  shuf_idx = np.arange(X_train.shape[0])
  np.random.shuffle(shuf_idx)
  shuf_idx = shuf_idx[:shuf_idx.shape[0]//3]
  X_train = X_train[shuf_idx]
  y_train = y_train[shuf_idx]

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
  return X_train, y_train, X_val, y_val

def DeepConvLSTM(X_train, y_train, X_val, y_val):
  nsamp, dim_length, dim_channels = X_train.shape
  output_dim = y_train.shape[1]  # number of classes
  weightinit = 'lecun_uniform'  # weight initialization
  regularization_rate = {{choice([10**-1, 10**-2, 10**-3])}}

  model = Sequential()  # initialize model
  model.add(BatchNormalization(input_shape=(dim_length, dim_channels)))
  # reshape a 2 dimensional array into 3D object
  #model.add(Reshape(target_shape=(dim_length, dim_channels, 1))

  # First conv layer
  model.add(Conv1D(int({{uniform(25,100)}}), kernel_size=3, padding='same', kernel_regularizer=l2(regularization_rate), kernel_initializer=weightinit))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  # Second and subsequent conv layers
  if {{choice(['one', 'two'])}} == 'two':
    model.add(Conv1D(int({{uniform(25,100)}}), kernel_size=3, padding='same', kernel_regularizer=l2(regularization_rate), kernel_initializer=weightinit))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if {{choice(['two', 'three'])}} == 'three':
      model.add(Conv1D(int({{uniform(25,100)}}), kernel_size=3, padding='same',\
                              kernel_regularizer=l2(regularization_rate), kernel_initializer=weightinit))
      model.add(BatchNormalization())
      model.add(Activation('relu'))
      if {{choice(['three', 'four'])}} == 'four':
        model.add(Conv1D(int({{uniform(25,100)}}), kernel_size=3, padding='same',\
                                kernel_regularizer=l2(regularization_rate), kernel_initializer=weightinit))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        if {{choice(['four', 'five'])}} == 'five':
          model.add(Conv1D(int({{uniform(25,100)}}), kernel_size=3, padding='same',\
                                  kernel_regularizer=l2(regularization_rate), kernel_initializer=weightinit))
          model.add(BatchNormalization())
          model.add(Activation('relu'))
          if {{choice(['five', 'six'])}} == 'six':
            model.add(Conv1D(int({{uniform(25,100)}}), kernel_size=3, padding='same',\
                                    kernel_regularizer=l2(regularization_rate), kernel_initializer=weightinit))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            if {{choice(['six', 'seven'])}} == 'seven':
              model.add(Conv1D(int({{uniform(25,100)}}), kernel_size=3, padding='same',\
                                      kernel_regularizer=l2(regularization_rate), kernel_initializer=weightinit))
              model.add(BatchNormalization())
              model.add(Activation('relu'))

  # First LSTM layer
  model.add(CuDNNLSTM(units=int({{uniform(100,500)}}), return_sequences=True))
  # Second and subsequent LSTM layers
  if {{choice(['one', 'two'])}} == 'two':
    model.add(CuDNNLSTM(units=int({{uniform(100,500)}}), return_sequences=True))
    if {{choice(['two', 'three'])}} == 'three':
      model.add(CuDNNLSTM(units=int({{uniform(100,500)}}), return_sequences=True))

  # Pool output of all timesteps and perform classification using the pooled output
  model.add(GlobalAveragePooling1D())
  model.add(Dropout(0.5)) # dropout before the dense layer
  model.add(Dense(units=output_dim, kernel_initializer=weightinit))
  model.add(BatchNormalization())
  model.add(Activation("softmax"))  # Final classification layer

  loss = 'categorical_crossentropy'
  metrics = [macro_f1]
  learning_rate = 0.01 # {{choice([10*-1,10*-2,10*-3])}}
  model.compile(loss=loss, optimizer=Adam(lr=learning_rate), metrics=metrics)

  model.fit(X_train, y_train, batch_size=50, epochs=1, verbose=2, validation_data=(X_val, y_val))
  score, f1_score = model.evaluate(X_val, y_val, verbose=0)
  print('Model validation macro_f1:', f1_score)
  return {'loss': -f1_score, 'status': STATUS_OK, 'model': model}

