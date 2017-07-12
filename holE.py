"""
Tensorflow implementation of Nickel et al, HolE, 2016

See https://arxiv.org/pdf/1510.04935.pdf
"""
import argparse
import errno
import itertools
import os
import random
import shutil
import sys

import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python import debug as tf_debug


FLAGS = None


def init_table(dictionary, key_dtype, value_dtype, name, pad_int=False):
    """Initializes the table variable and all of the inputs as constants."""
    padded_size = FLAGS.padded_size

    constants = []
    if pad_int:
        padded_size = FLAGS.padded_size
        for k, v in dictionary.iteritems():
            key = tf.constant(k, key_dtype)
            v = [random.choice(v) for _ in range(padded_size)]
            value = tf.constant(v, value_dtype)
            constants.append((key, value))
    else:
        key = tf.constant(dictionary.keys(), key_dtype)
        value = tf.constant(dictionary.values(), value_dtype)
        constants.append((key, value))

    if pad_int:
        default_value = tf.constant(padded_size * [-1], value_dtype)
    else:
        default_value = tf.constant('?', value_dtype)

    table = tf.contrib.lookup.MutableHashTable(key_dtype=key_dtype, value_dtype=value_dtype,
                                               default_value=default_value, shared_name=name, name=name)
    return table, constants


def populate_table(constants, table):
    for key, value in constants:
        table.insert(key, value).run()


def get_the_data():
    entity_file = 'diffbot_data/entity_metadata.tsv'
    relation_file = 'diffbot_data/relation_ids.txt'
    corrupt_triple_file = 'diffbot_data/triples.txt'

    type_to_ids = defaultdict(list)
    id_to_type = defaultdict(list)

    relation_count = sum(1 for line in open(relation_file))
    triple_count = sum(1 for line in open(corrupt_triple_file))

    entity_count = 0
    with open(entity_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            entity_count += 1
            index, diffbot_id, name, diffbot_type = line.split('\t')
            index = int(index)
            diffbot_type = diffbot_type.strip()
            type_to_ids[diffbot_type].append(index)
            id_to_type[index] = diffbot_type

    print 'Entities: ', entity_count - relation_count, 'Relations: ', relation_count, 'Triples: ', triple_count
    print 'Types: ', {k: len(v) for k, v in type_to_ids.iteritems()}
    for k, v in type_to_ids.iteritems():
        print k, np.random.choice(v, 10)

    with tf.name_scope('init_type_to_ids'):
        type_to_ids_table, type_to_ids_constants = init_table(type_to_ids, tf.string, tf.int64, 'type_to_ids',
                                                              pad_int=True)
    with tf.name_scope('init_id_to_type'):
        id_to_type_table, id_to_type_constants = init_table(id_to_type, tf.int64, tf.string, 'id_to_type')

    with tf.name_scope('input'):
        # Load triples from triple_file TSV
        reader = tf.TextLineReader()
        filename_queue = tf.train.string_input_producer([corrupt_triple_file] * FLAGS.num_epochs)
        key, value = reader.read(filename_queue)

        column_defaults = [tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32),
                           tf.constant([], dtype=tf.int32)]

        head_ids, tail_ids, relation_ids = tf.decode_csv(value, column_defaults, field_delim='\t')
        triples = tf.stack([head_ids, tail_ids, relation_ids])

    return type_to_ids_table, id_to_type_table, type_to_ids_constants, id_to_type_constants,\
        entity_count, relation_count, triple_count, triples


def corrupt_heads(type_to_ids, id_to_type, triples):
    # TODO: need to avoid type-safety for relation 'instance_of'
    with tf.name_scope('head'):
        head_column = tf.cast(tf.slice(triples, [0, 0], [-1, 1]), tf.int64)
        tail_column = tf.slice(triples, [0, 1], [-1, 1])
        relation_column = tf.slice(triples, [0, 2], [-1, 1])

        head_types = id_to_type.lookup(head_column)
        type_ids = tf.reshape(type_to_ids.lookup(head_types), [FLAGS.batch_size, FLAGS.padded_size])

        random_indices = tf.random_uniform([FLAGS.batch_size],
                                           maxval=FLAGS.padded_size,
                                           dtype=tf.int32)
        flattened_indices = tf.range(0, FLAGS.batch_size) * FLAGS.padded_size + random_indices
        corrupt_head_column = tf.reshape(tf.gather(tf.reshape(type_ids, [-1]), flattened_indices), [FLAGS.batch_size, 1])
        concat = tf.concat([tf.cast(corrupt_head_column, tf.int32), tail_column, relation_column], 1)
        return concat


def corrupt_tails(type_to_ids, id_to_type, triples):
    with tf.name_scope('tail'):
        head_column = tf.slice(triples, [0, 0], [-1, 1])
        tail_column = tf.cast(tf.slice(triples, [0, 1], [-1, 1]), tf.int64)
        relation_column = tf.slice(triples, [0, 2], [-1, 1])

        tail_types = id_to_type.lookup(tail_column)
        type_ids = tf.reshape(type_to_ids.lookup(tail_types), [FLAGS.batch_size, FLAGS.padded_size])

        random_indices = tf.random_uniform([FLAGS.batch_size],
                                           maxval=FLAGS.padded_size,
                                           dtype=tf.int32)
        flattened_indices = tf.range(0, FLAGS.batch_size) * FLAGS.padded_size + random_indices
        corrupt_tail_column = tf.reshape(tf.gather(tf.reshape(type_ids, [-1]), flattened_indices), [FLAGS.batch_size, 1])
        concat = tf.concat([head_column, tf.cast(corrupt_tail_column, tf.int32), relation_column], 1)
        return concat


def corrupt_entities(type_to_ids, id_to_type, triples):
    should_corrupt_heads = tf.less(tf.random_uniform([], 0, 1.0), 0.5)
    return tf.cond(should_corrupt_heads,
                   lambda: corrupt_heads(type_to_ids, id_to_type, triples),
                   lambda: corrupt_tails(type_to_ids, id_to_type, triples))


def corrupt_relations(relation_count, triples):
    with tf.name_scope('relation'):
        entity_columns = tf.slice(triples, [0, 0], [-1, 2])
        corrupt_relation_column = tf.random_uniform([FLAGS.batch_size, 1],
                                                    maxval=relation_count,
                                                    dtype=tf.int32)
        return tf.concat([entity_columns, corrupt_relation_column], 1)


def corrupt_batch(type_to_ids, id_to_type, relation_count, triples):
    should_corrupt_relation = tf.less(tf.random_uniform([], 0, 1.0), 0.5)
    return tf.cond(should_corrupt_relation,
                   lambda: corrupt_relations(relation_count, triples),
                   lambda: corrupt_entities(type_to_ids, id_to_type, triples))


def init_embedding(name, entity_count, embedding_dim):
    embedding = tf.get_variable(name, [entity_count, 2 * embedding_dim],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    return embedding


def get_embedding(layer_name, entity_ids, embeddings, embedding_dim):
    with tf.device('/cpu:0'):
        entity_embeddings = tf.nn.embedding_lookup(embeddings, entity_ids, max_norm=1)
        reshaped = tf.reshape(entity_embeddings, [-1, 2 * embedding_dim])
        return tf.complex(tf.slice(reshaped, [0, 0], [-1, embedding_dim]),
                          tf.slice(reshaped, [0, embedding_dim], [-1, embedding_dim]),
                          name=layer_name)


def complex_tanh(complex_tensor):
    summed = tf.reduce_sum(tf.real(complex_tensor) + tf.imag(complex_tensor), 1, keep_dims=True)
    return tf.tanh(summed)


def circular_correlation(h, t):
    # these ops are GPU only!
    ifft = tf.ifft(tf.multiply(tf.conj(tf.fft(h)), tf.fft(t)))
    return ifft


def evaluate_triples(triple_batch, embeddings, embedding_dim):
    # Load embeddings
    head_column = tf.slice(triple_batch, [0, 0], [-1, 1], name='h_id')
    head_embeddings = get_embedding('h', head_column, embeddings, embedding_dim)
    tail_column = tf.slice(triple_batch, [0, 1], [-1, 1], name='t_id')
    tail_embeddings = get_embedding('t', tail_column, embeddings, embedding_dim)
    relation_column = tf.slice(triple_batch, [0, 2], [-1, 1], name='r_id')
    relation_embeddings = get_embedding('r', relation_column, embeddings, embedding_dim)

    # Compute loss
    with tf.name_scope('eval'):
        if FLAGS.cpu:
            # TransE
            loss = complex_tanh(head_embeddings + relation_embeddings - tail_embeddings)
        else:
            # TODO: try Hermitian dot-product
            loss = complex_tanh(tf.multiply(relation_embeddings,
                                            circular_correlation(head_embeddings, tail_embeddings)))
        variable_summaries(loss)
        return loss


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def run_training(type_to_ids_table, id_to_type_table, type_to_ids_constants, id_to_type_constants,
                 entity_count, relation_count, triple_count, triples):
    # Initialize parameters
    margin = FLAGS.margin
    embedding_dim = FLAGS.embedding_dim
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    batch_count = triple_count / batch_size
    print 'Embedding dimension: ', embedding_dim, 'Batch size: ', batch_size, 'Batch count: ', batch_count

    # Warning: this will clobber existing summaries
    if not FLAGS.resume_checkpoint and os.path.isdir(FLAGS.output_dir):
        shutil.rmtree(FLAGS.output_dir)
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Initialize embeddings (TF doesn't support complex embeddings, split real part and imaginary part)
    embeddings = init_embedding('embeddings', entity_count, embedding_dim)

    with tf.name_scope('batch'):
        # Sample triples
        triple_batch = tf.train.shuffle_batch([triples], batch_size,
                                              num_threads=FLAGS.reader_threads,
                                              capacity=2*triple_count,
                                              #min_after_dequeue=batch_size,
                                              # TODO: this probably won't scale
                                              min_after_dequeue=triple_count,
                                              allow_smaller_final_batch=False)

        # Evaluate triples
        with tf.name_scope('train'):
            train_loss = evaluate_triples(triple_batch, embeddings, embedding_dim)
        with (tf.name_scope('corrupt')):
            corrupt_triples = corrupt_batch(type_to_ids_table, id_to_type_table, relation_count, triple_batch)
            corrupt_loss = evaluate_triples(corrupt_triples, embeddings, embedding_dim)

        # Score and minimize hinge-loss
        loss = tf.maximum(train_loss - corrupt_loss + margin, 0)

        # Log the total batch loss
        variable_summaries(loss)
        average_loss = tf.reduce_mean(loss)

        global_step = tf.Variable(0, trainable=False)
        lr_decay = tf.train.inverse_time_decay(learning_rate, global_step,
                                               decay_steps=FLAGS.learning_decay_steps*batch_count,
                                               decay_rate=FLAGS.learning_decay_rate)

        # TODO: experiment with other optimizers
        optimizer = tf.train.GradientDescentOptimizer(lr_decay).minimize(loss, global_step=global_step)

    summaries = tf.summary.merge_all()

    # Save embeddings
    saver = tf.train.Saver({'embeddings': embeddings})

    # TODO: supervisor freezes graph -- need to populate table inside session
    # supervisor = tf.train.Supervisor(logdir=FLAGS.output_dir)
    # with supervisor.managed_session() as sess:
    with tf.Session() as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

        with tf.name_scope('insert_id_to_type'):
            populate_table(id_to_type_constants, id_to_type_table)
        with tf.name_scope('insert_type_to_ids'):
            populate_table(type_to_ids_constants, type_to_ids_table)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Load the previous model if resume_checkpoint=True
        if FLAGS.resume_checkpoint:
            saver.restore(sess, FLAGS.output_dir + '/model.ckpt')
            # TODO: continue counting from last epoch

        summary_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            epoch = 0
            # while not supervisor.should_stop():
            while True:
                epoch += 1

                projector_config = projector.ProjectorConfig()
                embeddings_config = projector_config.embeddings.add()
                embeddings_config.tensor_name = embeddings.name
                embeddings.metadata_path = 'diffbot_data/entity_metadata.tsv'
                projector.visualize_embeddings(summary_writer, projector_config)

                batch_losses = []
                for batch in range(1, batch_count):
                    # Train and log batch to summary_writer
                    if batch % (batch_count / 4) == 0:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, batch_loss, summary = sess.run([optimizer, average_loss, summaries],
                                                          options=run_options,
                                                          run_metadata=run_metadata)
                        step = '{}-{}'.format(epoch, batch)
                        summary_writer.add_run_metadata(run_metadata, step)
                        summary_writer.add_summary(summary, epoch * batch_count + batch)
                        print '\tSaved summary for step {}...'.format(step)

                    else:
                        _, batch_loss = sess.run([optimizer, average_loss])
                    batch_losses.append(batch_loss)

                # Checkpoint
                # TODO: verify embeddings are being saved properly
                save_path = saver.save(sess, FLAGS.output_dir + '/model.ckpt', epoch)

                print('Epoch {} Loss: {}, (Model saved as {})'
                      .format(epoch, np.mean(batch_losses), save_path))

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            saver.save(sess, FLAGS.output_dir + '/model.ckpt')

        coord.join(threads)


def infer_triples():
    # TODO: this should be loaded from the saved model
    embedding_dim = FLAGS.embedding_dim
    batch_size = FLAGS.batch_size
    entity_file = 'diffbot_data/entity_metadata.tsv'

    type_to_ids = defaultdict(list)
    id_to_metadata = dict()

    entity_count = 0
    with open(entity_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            entity_count += 1
            index, diffbot_id, name, diffbot_type = line.split('\t')
            index = int(index)
            diffbot_type = diffbot_type.strip()
            type_to_ids[diffbot_type].append(index)
            id_to_metadata[index] = diffbot_id + ' ' + name

    print 'Types: ', {k: len(v) for k, v in type_to_ids.iteritems()}

    # Infer persons
    # TODO: feed the entire stream in batches
    infer_heads = np.random.choice(type_to_ids['P'], 100)

    # Infer skills
    infer_relations = [6]

    # Candidate targets
    infer_tails = type_to_ids['S']

    triples = tf.constant(list(itertools.product(infer_heads, infer_relations, infer_tails)))

    with tf.name_scope('inference'):
        embeddings = init_embedding('embeddings', entity_count, embedding_dim)

        triple_batch = tf.train.batch([triples], batch_size,
                                      capacity=4 * batch_size,
                                      enqueue_many=True,
                                      allow_smaller_final_batch=False)

        eval_loss = evaluate_triples(triple_batch, embeddings, embedding_dim)

        # Save embeddings
        saver = tf.train.Saver({'embeddings': embeddings})

        with tf.Session() as sess:
            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            saver.restore(sess, FLAGS.output_dir + '/model.ckpt')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while True:
                    triples, batch_loss = sess.run([triple_batch, eval_loss])
                    for pair in zip(batch_loss, triples):
                        loss = pair[0]
                        confidence = -loss
                        if confidence > FLAGS.inference_threshold:
                            print 'Confidence {} that http://localhost:9200/diffbot_entity/Person/{} has skill\n\t' \
                                  'http://localhost:9200/diffbot_entity/Skill/{}'.\
                                  format(confidence, id_to_metadata[pair[1][0]], id_to_metadata[pair[1][2]])
            except tf.errors.OutOfRangeError:
                print('Done evaluation -- triple limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)


def main(_):
    if FLAGS.infer:
        infer_triples()
    else:
        type_to_ids_table, id_to_type_table, type_to_ids_constants, id_to_type_constants, \
            entity_count, relation_count, triple_count, triples = get_the_data()
        run_training(type_to_ids_table, id_to_type_table, type_to_ids_constants, id_to_type_constants,
                     entity_count, relation_count, triple_count, triples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Disable GPU-only operations (namely FFT/iFFT).'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--learning_decay_steps',
        type=float,
        default=32,
        help='Learning rate decay steps (in epochs).'
    )
    parser.add_argument(
        '--learning_decay_rate',
        type=float,
        default=0.5,
        help='Learning decay rate.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Batch size.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1000,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=64,
        help='Embedding dimension.'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=1.,
        help='Hinge loss margin.'
    )
    parser.add_argument(
        '--padded_size',
        type=int,
        default=100000,
        help='The maximum number of entities to use for each type while sampling corrupt triples.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='holE',
        help='Tensorboard Summary directory.'
    )
    parser.add_argument(
        '--reader_threads',
        type=int,
        default=4,
        help='Number of training triple file readers.'
    )
    parser.add_argument(
        '--resume_checkpoint',
        action='store_true',
        help='Resume training on the checkpoint model. Otherwise start with randomly initialized embeddings.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run with interactive Tensorflow debugger.'
    )
    parser.add_argument(
        '--infer',
        action='store_true',
        help='Infer new triples from the latest checkpoint model.'
    )
    parser.add_argument(
        '--inference-threshold',
        type=float,
        default=0.8,
        help='Infer new triples from the latest checkpoint model.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

