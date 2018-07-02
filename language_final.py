import tensorflow as tf
config = tf.ConfigProto(device_count={'CPU': 1})
sess = tf.Session(config=config)
with sess.as_default():
  print(tf.constant(42).eval())

##### which Gpu we are using

import os
def load_data(path):
    """
    Load the dataset
    """
    input_file=os.path.join(path)
    with open(input_file,"r") as f:
        data=f.read()
    return data.split('\n')
english_sentences=load_data('small_vocab_en.txt')
french_sentences=load_data('small_vocab_fr.txt')
for sample_i in range(2):
    print('small_vocab_en line {}: {}'.format(sample_i + 1,english_sentences[sample_i]))
    print('samll vocab_fr line {}: {}'.format(sample_i + 1,french_sentences[sample_i]))
import collections
english_words_counter=collections.Counter([word for sentence in english_sentences for word in sentence.split() ])
#print the count for each corresponding word
french_words_counter=collections.Counter([word for sentence in french_sentences for word in sentence.split() ])

#print('{} total english words'.format(english_words_counter))
ctr=0
#print("{} english words".format(len([word for sentence in english_sentences for word in sentence.split()])))
#print("{} unique english words".format(len(english_words_counter)))
#print("{} french words".format(len([word for sentence in french_sentences for word in sentence.split()])))
#print("{} unique french words".format(len(french_words_counter)))
#print("total sentences in french {}:",format(len([sentence for sentence in french_sentences])))
#print("total sentences in english {}: ",format(len([sentence for sentence in french_sentences])))
import numpy as np###########################3
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def _test_model(model, input_shape, output_sequence_length, french_vocab_size):
    if isinstance(model, Sequential):
        model = model.model

    assert model.input_shape == (None, *input_shape[1:]),\
        'Wrong input shape. Found input shape {} using parameter input_shape={}'.format(model.input_shape, input_shape)

    assert model.output_shape == (None, output_sequence_length, french_vocab_size),\
        'Wrong output shape. Found output shape {} using parameters output_sequence_length={} and french_vocab_size={}'\
            .format(model.output_shape, output_sequence_length, french_vocab_size)

    assert len(model.loss_functions) > 0,\
        'No loss function set.  Apply the `compile` function to the model.'

    assert sparse_categorical_crossentropy in model.loss_functions,\
        'Not using `sparse_categorical_crossentropy` function for loss.'


def test_tokenize(tokenize):
    sentences = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .']
    tokenized_sentences, tokenizer = tokenize(sentences)
    assert tokenized_sentences == tokenizer.texts_to_sequences(sentences),\
        'Tokenizer returned and doesn\'t generate the same sentences as the tokenized sentences returned. '


def test_pad(pad):
    tokens = [
        [i for i in range(4)],
        [i for i in range(6)],
        [i for i in range(3)]]
    padded_tokens = pad(tokens)
    padding_id = padded_tokens[0][-1]
    true_padded_tokens = np.array([
        [i for i in range(4)] + [padding_id]*2,
        [i for i in range(6)],
        [i for i in range(3)] + [padding_id]*3])
    assert isinstance(padded_tokens, np.ndarray),\
        'Pad returned the wrong type.  Found {} type, expected numpy array type.'
    assert np.all(padded_tokens == true_padded_tokens), 'Pad returned the wrong results.'

    padded_tokens_using_length = pad(tokens, 9)
    assert np.all(padded_tokens_using_length == np.concatenate((true_padded_tokens, np.full((3, 3), padding_id)), axis=1)),\
        'Using length argument return incorrect results'


def test_simple_model(simple_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_embed_model(embed_model):
    input_shape = (137861, 21)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_encdec_model(encdec_model):
    input_shape = (137861, 15, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_bd_model(bd_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_model_final(model_final):
    input_shape = (137861, 15)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)
import numpy as np###########################3
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def _test_model(model, input_shape, output_sequence_length, french_vocab_size):
    if isinstance(model, Sequential):
        model = model.model

    assert model.input_shape == (None, *input_shape[1:]),\
        'Wrong input shape. Found input shape {} using parameter input_shape={}'.format(model.input_shape, input_shape)

    assert model.output_shape == (None, output_sequence_length, french_vocab_size),\
        'Wrong output shape. Found output shape {} using parameters output_sequence_length={} and french_vocab_size={}'\
            .format(model.output_shape, output_sequence_length, french_vocab_size)

    assert len(model.loss_functions) > 0,\
        'No loss function set.  Apply the `compile` function to the model.'

    assert sparse_categorical_crossentropy in model.loss_functions,\
        'Not using `sparse_categorical_crossentropy` function for loss.'


def test_tokenize(tokenize):
    sentences = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .']
    tokenized_sentences, tokenizer = tokenize(sentences)
    assert tokenized_sentences == tokenizer.texts_to_sequences(sentences),\
        'Tokenizer returned and doesn\'t generate the same sentences as the tokenized sentences returned. '


def test_pad(pad):
    tokens = [
        [i for i in range(4)],
        [i for i in range(6)],
        [i for i in range(3)]]
    padded_tokens = pad(tokens)
    padding_id = padded_tokens[0][-1]
    true_padded_tokens = np.array([
        [i for i in range(4)] + [padding_id]*2,
        [i for i in range(6)],
        [i for i in range(3)] + [padding_id]*3])
    assert isinstance(padded_tokens, np.ndarray),\
        'Pad returned the wrong type.  Found {} type, expected numpy array type.'
    assert np.all(padded_tokens == true_padded_tokens), 'Pad returned the wrong results.'

    padded_tokens_using_length = pad(tokens, 9)
    assert np.all(padded_tokens_using_length == np.concatenate((true_padded_tokens, np.full((3, 3), padding_id)), axis=1)),\
        'Using length argument return incorrect results'


def test_simple_model(simple_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_embed_model(embed_model):
    input_shape = (137861, 21)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_encdec_model(encdec_model):
    input_shape = (137861, 15, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_bd_model(bd_model):
    input_shape = (137861, 21, 1)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)


def test_model_final(model_final):
    input_shape = (137861, 15)
    output_sequence_length = 21
    english_vocab_size = 199
    french_vocab_size = 344

    model = model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
    _test_model(model, input_shape, output_sequence_length, french_vocab_size)
from keras.preprocessing.text import Tokenizer
def tokenize(x):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x),tokenizer
test_tokenize(tokenize)
# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized,text_tokenizer=tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i,(sent,token_sent) in enumerate(zip(text_sentences,text_tokenized)):
    print('Sequence {} in x'.format(sample_i+1))
    print('Input: {}'.format(sent))
    print('Output {}'.format(token_sent))
import numpy as np
from keras.preprocessing.sequence import pad_sequences
def pad(x,length=None):
    return pad_sequences(x,maxlen=length,padding='post')
test_pad(pad)
test_pad=pad(text_tokenized)
for sample_i,(token_sent,pad_sent) in enumerate(zip(text_tokenized,test_pad)):
    ##print('Sequence {} in x'.format(sample_i+1))
    #print('Input {}'.format(np.array(token_sent)))
    #print('Output: {}'.format(pad_sent))
"""
alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']

for a, b in zip(alist, blist):
    print a, b
    results:
a1 b1
a2 b2
a3 b3
alist = ['a1', 'a2', 'a3']

for i, a in enumerate(alist):
    print i, a
Results:

0 a1
1 a2
2 a3
    
    """
def preprocess(x,y):#combination of the above two functions
    preprocess_x,x_tk=tokenize(x)
    preprocess_y,y_tk=tokenize(y)
    preprocess_x=pad(preprocess_x)
    preprocess_y=pad(preprocess_y)##preprocess_x after padding,x_tk=words in all the sentences
    print('shape before:',preprocess_y.shape)#preprocess_y after reshaping
    preprocess_y=preprocess_y.reshape(*preprocess_y.shape,1)
    print('shpe after:',preprocess_y.shape[-1])#shape 0:13861,shape 1:21....&&& shape -1:last,shape-2:second last........
    return(preprocess_x,preprocess_y,x_tk,y_tk)
preproc_english_sentences,preproc_french_sentences,english_tokenizer,french_tokenizer=preprocess(english_sentences,french_sentences)
print('DAta preprocessed')
"""
shape before:1378621 is the total no of sentences,
21 is the total words in 1 sentences ---we are doing it only for french"""
#######################to see the output properly we r converting indices into words
def logits_to_text(logits,tokenizer):
    index_to_words={id:word for word,id in tokenizer.word_index.items()}
    index_to_words[0]='<PAD>'
    return' '.join([index_to_words[prediction] for prediction in np.argmax(logits,1)])
##argmax at wich th efunction output are as large as possible
print('logits_to_ text loaded.................')
"""
pred = np.array([[31, 23,  4, 24, 27, 34],
                [18,  3, 25,  0,  6, 35],
                [28, 14, 33, 22, 20,  8],
                [13, 30, 21, 19,  7,  9],
                [16,  1, 26, 32,  2, 29],
                [17, 12,  5, 11, 10, 15]])

y = np.array([[31, 23,  4, 24, 27, 34],
                [18,  3, 25,  0,  6, 35],
                [28, 14, 33, 22, 20,  8],
                [13, 30, 21, 19,  7,  9],
                [16,  1, 26, 32,  2, 29],
                [17, 12,  5, 11, 10, 15]])
Evaluating tf.argmax(pred, 1) gives a tensor whose evaluation will give array([5, 5, 2, 1, 3, 0])
"""
#basic RNN
import os
from keras.models import load_model
import numpy as np
print(preproc_french_sentences.shape)
print(preproc_english_sentences.shape)
print("french",preproc_french_sentences.shape[2])
print(preproc_french_sentences[0])
print("english vocab size",len(english_tokenizer.word_index) )
print(preproc_french_sentences.shape[-2])#french we have reshaped before check the outputs for -3,-2,-1'
print(preproc_english_sentences.shape)
print(preproc_english_sentences)
# earlier  english sentences padding was 15,now we increased the padding to 21 equal to that of french,,done below 
tmp_x=pad(preproc_english_sentences,preproc_french_sentences.shape[1])
print(tmp_x)
tmp_x=tmp_x.reshape((-1,preproc_french_sentences.shape[-2],1))
print("input shape: ", tmp_x.shape)#english sentence padding

from keras.layers import RepeatVector
learning_rate=0.1
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.models import Model, Sequential
from keras.layers import Activation
from keras.optimizers import Adam
import os
from keras.models import load_model
import numpy as np
from keras.losses import sparse_categorical_crossentropy
def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size, learning_rate=0.01):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    vocab_size = max(english_vocab_size, french_vocab_size)
    # OPTIONAL: Implement
    model = Sequential()
    model.add(Embedding(vocab_size ,128 , input_length=input_shape[1]))
    #encoder
    model.add(Bidirectional(GRU(128, return_sequences=False)) )
    model.add(RepeatVector(output_sequence_length))
    #decoder
    model.add(Bidirectional(GRU(128, return_sequences=True)) )
    #output
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax') ))
    
    print('**summary**')
    model.summary()
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model
test_model_final(model_final)


print('Final Model Loaded')
X_input = pad(preproc_english_sentences)
model_final=model_final(
    X_input.shape,
    preproc_french_sentences.shape[1],
    len(english_tokenizer.word_index)+1,
    len(french_tokenizer.word_index)+1)

if os.path.exists(os.path.join("cache", "model_final.h5"))== False:
    print("train")
    history=model_final.fit(X_input, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.2)
else:
    print("load")
    model_final = load_model(os.path.join("cache", "model_final.h5"))
from keras.preprocessing.sequence import pad_sequences


def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    """
    # TODO: Train neural network using model_final
    # Save model, so that it can quickly load it in future (and perhaps resume training)
    if os.path.exists("final_model.h5") == False:
        model_final.save("final_model.h5")
    
    ## DON'T EDIT ANYTHING BELOW THIS LINE
    score = model_final.evaluate(X_input, preproc_french_sentences, verbose=0)
    print("Train accurancy: ", score[1])
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = 'he saw a yellow old truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model_final.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
   # print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.argmax(x)] for x in y[0]]))

    return model_final

final_model = final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)