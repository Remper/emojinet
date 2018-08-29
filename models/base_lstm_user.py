from keras import Model, Input
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, regularizers, Average, K, Lambda
import numpy as np
from keras.optimizers import Adam


def base_lstm_user(vocabulary_size: int, embedding_size: int, max_seq_length: int, embedding_matrix: np.array, y_dictionary: dict) -> Model:
    input = Input(shape=(max_seq_length,), name='main_input')
    history = Input(shape=(y_dictionary,), name='main_input')

    model = Embedding(input_dim=vocabulary_size,
                        output_dim=embedding_size,
                        weights=[embedding_matrix],
                        input_length=max_seq_length,
                        trainable=True,
                        embeddings_regularizer=regularizers.l2(0.000001))(input)
    model = Dropout(0.4)(model)
    model = Bidirectional(LSTM(256))(model)
    model = Dense(len(y_dictionary), activation='softmax')(model)

    history = Dense(len(y_dictionary), activation='softmax')(history)

    model = Average()([model, history])
    model = Model([input, history], model)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model