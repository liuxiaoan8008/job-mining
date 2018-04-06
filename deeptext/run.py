#-*- coding:utf-8 -*-

import logging
import sys
import os

from optparse import OptionParser
from deeptext.models.sequence_labeling.biLSTM_crf_sequence_labeling import BiCrfSequenceLabeling


def deep_ner(mode,model_dir,data_file,valid_data_file=None):
    params = {}

    params["max_document_len"] = 25
    params["embedding_size"] = 100
    params["dropout_prob"] = 0.5

    params["model_name"] = 'model_' + str(params["embedding_size"]) + '_' + str(params["max_document_len"])
    params["model_dir"] = model_dir

    model = BiCrfSequenceLabeling(params)

    if mode == 'train':
        steps = 500
        batch_size = 256

        if valid_data_file == None:
            valid_data_file = data_file
            print 'using training data as valid data.'
        model.fit(
            steps=steps,
            batch_size=batch_size,
            training_data_path=data_file,
            validation_data_path=valid_data_file
        )

    elif mode == 'test':
        model.pre_restore()
        model.restore(model.sess)
        model.evaluate(testing_data_path=data_file)

    elif mode == 'try':
        model.pre_restore()
        model.restore(model.sess)
        while True:
            logging.info("please input a sentence: ")
            sentence = unicode(raw_input(), 'utf8')
            sentence = u'^' + sentence + u'$'
            labels = model.predict([list(sentence)])
            print labels[0][0][2:-2]


if __name__ == '__main__':

    FORMAT = '[%(asctime)-15s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = OptionParser()

    parser.add_option("-m", "--mode",  help="interaction mode: train, test, try")
    parser.add_option("-o", "--model_dir",  help="dir for model weights, like '/usr/model'")

    parser.add_option("-d", "--data", help="train or test data file, like '/usr/data/train_data.txt'")
    parser.add_option("-v", "--valid_data", help="valid data file, like '/usr/data/valid_data.txt'")

    # train
    parser.add_option("-s", "--step", help="train step, default is 500")
    parser.add_option("-b", "--batch", help="batch size, default is 256")

    parser.add_option("-l", "--len", help="max_document_len, default is 25")
    parser.add_option("-e", "--embed", help="embedding_size, default is 100")
    parser.add_option("-p", "--prob", help="dropout_prob, default is 0.5")

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    (options, args) = parser.parse_args()
    assert options.mode is not None, "missing mode"
    assert options.model_dir is not None, "missing model file dir"

    if options.mode == "train":
        assert options.data is not None, "missing train data"
        assert options.valid_data is not None, "missing valid data"
    elif options.mode == "test":
        assert options.data is not None, "missing test data"
    elif options.mode == "try":
        pass
    else:
        assert False, "unknown mode"

    if not os.path.exists(options.model_dir):
            os.makedirs(options.model_dir)

    params = {}

    params["max_document_len"] = 25
    params["embedding_size"] = 100
    params["dropout_prob"] = 0.5

    if options.len is not None:
            params["max_document_len"] = options.len
    if options.embed is not None:
            params["embedding_size"] = options.embed
    if options.prob is not None:
            params["dropout_prob"] = options.prob


    params["model_name"] = 'model_'+str(params["embedding_size"])+'_'+str(params["max_document_len"])
    params["model_dir"] = options.model_dir



    model = BiCrfSequenceLabeling(params)
    if options.mode == 'train':
            steps = 500
            if options.step is not None:
                    steps = options.step

            batch_size = 256
            if options.batch is not None:
                    batch_size = options.batch

            model.fit(
                    steps=steps,
                    batch_size=batch_size,
                    training_data_path=options.data,
                    validation_data_path=options.valid_data
                    )

    elif options.mode == 'test':

            model.pre_restore()
            model.restore(model.sess)
            model.evaluate(testing_data_path=options.data)

    elif options.mode == 'try':
            model.pre_restore()
            model.restore(model.sess)
            while True:
                    logging.info("please input a sentence: ")
                    sentence = unicode(raw_input(), 'utf8')
                    sentence = u'^' + sentence + u'$'
                    labels = model.predict([list(sentence)])
                    print labels[0][0][2:-2]
