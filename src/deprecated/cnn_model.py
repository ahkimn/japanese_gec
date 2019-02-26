import os
import time

import configx
import fairseq
import load

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from fairseq.models.fconv import FConvEncoder, FConvDecoder, FConvModel
from fairseq.data import Dictionary


def create_dictionary(source=True):
    """
    Create and update a Dictionary class instance for use in the Fairseq models
    """

    dictionary = Dictionary(configx.CONST_PAD_INDEX, 
                            configx.CONST_SENTENCE_DELIMITER_INDEX, 
                            configx.CONST_UNKNOWN_INDEX)
    if source:
        language = "error"
    
    else:
        language = "correct"

    search_dir = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY, configx.CONST_TEXT_OUTPUT_PREFIX, configx.CONST_CORPUS_SAVE_DIRECTORY)
    file_list = os.listdir(search_dir)

    sentences = list()

    for file_name in file_list:

        if language in file_name and "full" not in file_name:

            f = open(os.path.join(search_dir, file_name), "r")
            lines = f.readlines()            

            for line in lines:

                for token in line.split(" "):

                    dictionary.add_symbol(token)

            f.close()

    dictionary.finalize(threshold=configx.CONST_MIN_FREQUENCY, nwords=configx.CONST_MAX_DICT_SIZE)

    if not os.path.isdir(configx.CONST_CNN_SAVE_DIRECTORY):

        os.mkdir(configx.CONST_CNN_SAVE_DIRECTORY)

    if source:

        save_file = os.path.join(configx.CONST_CNN_SAVE_DIRECTORY, configx.CONST_SOURCE_DICTIONARY_NAME)
        dictionary.save(save_file)

    else:

        save_file = os.path.join(configx.CONST_CNN_SAVE_DIRECTORY, configx.CONST_TARGET_DICTIONARY_NAME)
        dictionary.save(save_file)

    return dictionary

def load_dictionary(source=True):

    if source:
    
        save_file = os.path.join(configx.CONST_CNN_SAVE_DIRECTORY, configx.CONST_SOURCE_DICTIONARY_NAME)
    
    else:

        save_file = os.path.join(configx.CONST_CNN_SAVE_DIRECTORY, configx.CONST_TARGET_DICTIONARY_NAME)

    dictionary = Dictionary.load(save_file)

    return dictionary

def iterate_batches(data_arrays, batch_size, shuffle=True):
    """
    Iterate over batches of a given size when the input is simply the raw image itself

    :param data_arrays: Tensors that need to be batched

    :param batch_size: The batch size to be iterated over
    :param shuffle: Variable determines whether or not to shuffle the order of the batches' indices each time

    :return: An iterator over the inputs/targets with each iteration being of size batch
    """

    n = len(data_arrays[0])
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start_index in range(0, n, batch_size):

        batch_indices = indices[start_index:start_index + batch_size]
        batch_indices = torch.from_numpy(batch_indices)

        yield list(arr[batch_indices] for arr in data_arrays) 

def save_data(dictionary, data="train", source=True):

    if source:
        suffix="error"

    else:
        suffix="correct"

    sentences, lengths = get_sentences(data, source)

    arr = np.zeros((len(sentences), configx.CONST_MAX_SENTENCE_LENGTH), dtype="int32")
    arr.fill(1)
    arr_lengths = np.zeros((len(lengths)), dtype="int32")

    arr_lengths[:] = lengths[:]

    for i in range(len(sentences)):

        arr[i][:len(sentences[i])] = list(dictionary.index(j) for j in sentences[i])

    save_name = os.path.join(configx.CONST_CNN_SAVE_DIRECTORY, data + suffix + ".npy")
    save_name_lengths = os.path.join(configx.CONST_CNN_SAVE_DIRECTORY, data + suffix + "lengths" + ".npy")

    np.save(save_name, arr)
    np.save(save_name_lengths, arr_lengths)

def load_data(data="train", source=True):

    if source:
        suffix="error"

    else:
        suffix="correct"

    save_name = os.path.join(configx.CONST_CNN_SAVE_DIRECTORY, data + suffix + ".npy")
    save_name_lengths = os.path.join(configx.CONST_CNN_SAVE_DIRECTORY, data + suffix + "lengths" + ".npy")

    return np.load(save_name), np.load(save_name_lengths)




def get_sentences(data, suffix):

    search_dir = os.path.join(configx.CONST_TEXT_OUTPUT_DIRECTORY, configx.CONST_TEXT_OUTPUT_PREFIX, configx.CONST_CORPUS_SAVE_DIRECTORY)
    file_list = os.listdir(search_dir)

    sentences = list()
    lengths = list()

    for file_name in file_list:

        if data in file_name and "full" not in file_name:

            f = open(os.path.join(search_dir, file_name), "r")
            lines = f.readlines()            

            for line in lines:
                # a = list(str(configx.CONST_SENTENCE_START_INDEX))
                a = list(i for i in line.strip().split(" "))
                sentences.append(a)
                lengths.append(len(a))

            f.close()

    return sentences, lengths

def train(net, n_epochs=1, batch_size=100):


    ts, tsl = load_data("train", True)
    tt, ttl = load_data("train", False)

    print(np.max(tsl))
    print(np.max(ttl))
    ts = torch.tensor(ts, dtype=torch.long)
    tt = torch.tensor(tt, dtype=torch.long)
    tsl = torch.tensor(tsl, dtype=torch.long)
    ttl = torch.tensor(ttl, dtype=torch.long)

    data_arrays = [ts, tt, tsl, ttl]

    # Define the loss function and the method of optimization (SGD w/ momentum)
    criterion = nn.NLLLoss()
    encoder_optimizer = optim.SGD(net.encoder.parameters(), lr=0.001, momentum=0.9)
    decoder_optimizer = optim.SGD(net.decoder.parameters(), lr=0.001, momentum=0.9)
    # print("Network Details...")
    # print("\t" + str(net))

    print("\nBeginning Training...")

    training_start = time.time()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_start = time.time()

        for batch in iterate_batches(data_arrays, batch_size, shuffle=True):

            batch_inputs = batch[0]
            batch_outputs= batch[1]
            batch_input_lengths = batch[2]

            batch_inputs, batch_input_lengths = Variable(batch_inputs), Variable(batch_input_lengths)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            enoder_out = net.encoder(batch_inputs, batch_input_lengths)
            decoder_out = net.decoder()
            raise
            batch_loss = criterion(batch_outputs, batch_labels)

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.data[0]
       
        print('\nEpoch: [%d]\n\tLoss: %.6f Elapsed Time: %.6fs' %
              (epoch + 1, running_loss / n_training, time.time() - epoch_start))
        running_loss = 0.0

    print("Finished Training in %.6fs" % (time.time() - training_start))





if __name__ == '__main__':

    # for data in ["train", "test", "validation"]:
    #     for source in [False, True]:
    #         save_ndarray(data, source)  

    # source_dict = create_dictionary()
    # target_dict = create_dictionary(False)

    source_dict = load_dictionary()
    target_dict = load_dictionary(False)

    embed = fairseq.utils.parse_embedding(configx.CONST_EMBEDDING_FAIRSEQ_SAVE)

    convs = [(128, 3)] * 9  # first 9 layers have 512 units
    convs += [(256, 3)] * 4  # next 4 layers have 1024 units
    convs += [(512, 1)] * 2  # final 2 layers use 1x1 convolutions

    # save_data(source_dict, "train", True)
    # save_data(target_dict, "train", False)

    encoder = FConvEncoder(source_dict, 
                           embed_dim=configx.CONST_EMBEDDING_SIZE, 
                           embed_dict=embed, 
                           max_positions=configx.CONST_MAX_SENTENCE_LENGTH,
                           convolutions=convs,
                           left_pad=False
                           )



    
    decoder = FConvDecoder(target_dict)

    net = FConvModel(encoder, decoder)
    # net.cuda()

    train(net)
