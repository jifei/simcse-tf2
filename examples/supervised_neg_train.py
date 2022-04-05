from simcse_tf2.simcse import simcse
from simcse_tf2.data import load_data, SimCseDataGenerator
from simcse_tf2.losses import simcse_hard_neg_loss
import tensorflow as tf

if __name__ == '__main__':
    # 1. bert config
    model_path = '/Users/jifei/models/bert/chinese_L-12_H-768_A-12'
    # model_path = '/Users/jifei/models/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12'
    checkpoint_path = '%s/bert_model.ckpt' % model_path
    config_path = '%s/bert_config.json' % model_path
    dict_path = '%s/vocab.txt' % model_path

    # 2. set hyper parameters
    max_len = 64
    pooling = 'cls'  # in ['first-last-avg', 'last-avg', 'cls', 'pooler']
    dropout_rate = 0.1
    batch_size = 64
    learning_rate = 5e-5
    epochs = 2
    output_units = 128
    activation = 'tanh'

    # 3. data generator
    # random negative sampling
    train_data = load_data('./data/sup_sample.csv', delimiter='\t', skip_header=False, random_negative_sampling=True)
    # with hard negative sampling
    # train_data = load_data('./data/sup_with_neg_sample.csv', delimiter='\t')

    train_generator = SimCseDataGenerator(train_data, dict_path, batch_size, max_len, text_tuple_size=3)
    # print(next(train_generator.forfit()))

    # 4. build model
    model = simcse(config_path, checkpoint_path, dropout_rate=dropout_rate, output_units=output_units,
                   output_activation=activation)

    # 5. model compile
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss=simcse_hard_neg_loss, optimizer=optimizer)

    # 6. model fit
    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs)
