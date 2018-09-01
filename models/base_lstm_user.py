from keras import Model, Input
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, regularizers, K, Lambda, Concatenate, Permute, RepeatVector, Multiply, Flatten, Activation
import numpy as np
from keras.optimizers import Adam


def base_lstm_user(vocabulary_size: int, embedding_size: int, history_size: int, max_seq_length: int, embedding_matrix: np.array, y_dictionary: dict) -> Model:

    lstm_output_shape = 256

    input = Input(shape=(max_seq_length,), name='main_input')
    history = Input(shape=(history_size,), name='history_input')

    model = Embedding(input_dim=vocabulary_size,
                        output_dim=embedding_size,
                        weights=[embedding_matrix],
                        input_length=max_seq_length,
                        trainable=True,
                        embeddings_regularizer=regularizers.l2(0.000001))(input)
    model = Dropout(0.4)(model)
    model = Bidirectional(LSTM(lstm_output_shape, return_sequences=True))(model)

    attention = Dense(1, activation='tanh')(model)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(2*lstm_output_shape)(attention)
    attention = Permute([2, 1])(attention)

    attention = Multiply()([model, attention])
    model = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(2*lstm_output_shape,))(attention)

    h_model = history
    for i in range(2):
        h_model = Dense((2*lstm_output_shape)/(i+1), activation='tanh', kernel_regularizer=regularizers.l2(0.00001))(h_model)

    model = Concatenate()([model, h_model])
    model = Dense(len(y_dictionary), activation='softmax')(model)
    model = Model([input, history], model)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model