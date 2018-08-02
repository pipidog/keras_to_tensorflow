# This file shows how to save a keras model to tensorflow pb file 
# and how to use tensorflow to reload the model and inferece by the model 

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
np.random.seed(0)

# parameter ==========================
wkdir = '/home/pipidog/keras_to_tensorflow'
pb_filename = 'model.pb'

# build a keras model ================
x = np.vstack((np.random.rand(1000,10),-np.random.rand(1000,10)))
y = np.vstack((np.ones((1000,1)),np.zeros((1000,1))))
print(x.shape)
print(y.shape)

model = Sequential()
model.add(Dense(units = 32, input_shape=(10,), activation ='relu'))
model.add(Dense(units = 16, activation ='relu'))
model.add(Dense(units = 1, activation ='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])
model.fit(x = x, y=y, epochs = 2, validation_split=0.2) 

# save model to pb ====================
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

# save keras model as tf pb files ===============
from keras import backend as K
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)


# # load & inference the model ==================

from tensorflow.python.platform import gfile
with tf.Session() as sess:
    # load model from pb file
    with gfile.FastGFile(wkdir+'/'+pb_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        g_in = tf.import_graph_def(graph_def)
    # write to tensorboard (check tensorboard for each op names)
    writer = tf.summary.FileWriter(wkdir+'/log/')
    writer.add_graph(sess.graph)
    writer.flush()
    writer.close()
    # print all operation names 
    print('\n===== ouptut operation names =====\n')
    for op in sess.graph.get_operations():
      print(op)
    # inference by the model (op name must comes with :0 to specify the index of its output)
    tensor_output = sess.graph.get_tensor_by_name('import/dense_3/Sigmoid:0')
    tensor_input = sess.graph.get_tensor_by_name('import/dense_1_input:0')
    predictions = sess.run(tensor_output, {tensor_input: x})
    print('\n===== output predicted results =====\n')
    print(predictions)