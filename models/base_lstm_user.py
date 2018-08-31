from keras import Model, Input
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, regularizers, Average, K, Lambda, Concatenate, Flatten, Activation, RepeatVector, Permute, Multiply, Reshape
import numpy as np
from keras.optimizers import Adam


def base_lstm_user(vocabulary_size: int, embedding_size: int, history_size: int, max_seq_length: int, embedding_matrix: np.array, y_dictionary: dict) -> Model:
    units = 256

    input = Input(shape=(max_seq_length,), name='main_input')
    history = Input(shape=(history_size,), name='history_input')

    model = Embedding(input_dim=vocabulary_size,
                        output_dim=embedding_size,
                        weights=[embedding_matrix],
                        input_length=max_seq_length,
                        trainable=True,
                        embeddings_regularizer=regularizers.l2(0.000001))(input)
    model = Dropout(0.4)(model)
    model = Bidirectional(LSTM(units, return_sequences = True))(model)

    # https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
    attention_input_dim = int(model.shape[2])
    attention = Permute((2, 1))(model)
    attention = Reshape((attention_input_dim, 20))(attention)
    attention = Dense(20, activation='softmax')(attention)
    #attention = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(attention)
    #attention = RepeatVector(attention_input_dim)(attention)
    attention = Permute((2, 1), name='attention_vec')(attention)
    attention = Multiply([model, attention], name='attention_mul')

    model = Flatten()(attention)

    h_model = history
    for i in range(2):
        h_model = Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(0.00001))(h_model)

    model = Concatenate()([model, h_model])
    model = Dense(len(y_dictionary), activation='softmax')(model)
    model = Model([input, history], model)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model