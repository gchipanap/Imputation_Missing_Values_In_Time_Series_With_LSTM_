import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from custom_loss import  Ltotal

# Define flags
class FLAGS:
    num_layers = 1
    hidden_size = 128
    embedding_size = 3
    missing_flag = -1.0

class LSTM_RSV(tf.keras.Model):    
    def __init__(self, is_training, config, FLAGS):
        super(LSTM_RSV, self).__init__()
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps 
        size = config.hidden_size  
        embedding_size = config.embedding_size 

        self.input_data = tf.keras.Input(shape=(self.num_steps, embedding_size), batch_size=self.batch_size)  # input
        self.targets = tf.keras.Input(shape=(self.num_steps, embedding_size), batch_size=self.batch_size)  # output
        lstm_cell = tf.keras.layers.LSTMCell(size, forget_bias=0.0)
        cell = tf.keras.layers.StackedRNNCells([lstm_cell] * config.num_layers)
        self.initial_state = cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)

        state = self.initial_state 
        self.F=[]

        for time_step in range(self.num_steps):
            if time_step > 0:
                tf.keras.backend.set_learning_phase(1)
            if time_step == 0:
                (cell_output, state) = cell(self.input_data[:, time_step, :], initial_state=state)
                self.F.append(cell_output)
            else:
                comparison = tf.equal(self.input_data[:, time_step, :], tf.constant(FLAGS.missing_flag))
                temp2 = tf.matmul(self.F[time_step - 1], variables_dict["W_imp"]) + variables_dict["bias"]
                input2 = tf.where(comparison, temp2, self.input_data[:, time_step, :])
                (cell_output, state) = cell(input2, initial_state=state)
                self.F.append(cell_output + tf.matmul(self.F[time_step - 1], variables_dict["W_r"]))

        F_out = tf.reshape(tf.concat(self.F, 1), [-1, size])
        self.prediction = tf.matmul(F_out, variables_dict["W_imp"]) + variables_dict["bias"]
        targets = tf.reshape(self.targets, [-1, embedding_size])
        comparison2 = tf.equal(targets, tf.constant(FLAGS.missing_flag))
        targets = tf.where(comparison2, self.prediction, targets)

        self.cost = tf.reduce_mean(tf.square(targets - self.prediction)) / self.batch_size
        self.final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(self._lr)

    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = tf.reduce_mean(tf.square(targets - predictions)) / self.batch_size
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def assign_lr(self, lr_value):
        self.optimizer.learning_rate = lr_value

class TrainConfig:
    def __init__(self):
        self.init_scale = 0.1
        self.num_layers = FLAGS.num_layers
        self.num_steps = 12
        self.hidden_size = FLAGS.hidden_size
        self.batch_size = 4
        self.embedding_size = FLAGS.embedding_size

class TestConfig:
    def __init__(self):
        self.init_scale = 0.1
        self.num_layers = FLAGS.num_layers
        self.num_steps = 12
        self.hidden_size = FLAGS.hidden_size
        self.batch_size = 4
        self.embedding_size = FLAGS.embedding_size

def run_epoch(session, model, data, eval_op):
    header_pre = []
    pre = []
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps, FLAGS.embedding_size)):
        _, prediction = session.run([eval_op, model.prediction], {model.input_data: x, model.targets: y})
        if step == 0:
            header_pre.append(prediction[:model.num_steps-1, :])
        for i in range(model.batch_size):
            pre.append(prediction[(i+1) * model.num_steps - 1, :])
    return np.concatenate((np.array(header_pre).reshape(-1, FLAGS.embedding_size), np.array(pre)), axis=0)

def get_config(flag):
    if flag == "Train":
        return TrainConfig()
    elif flag == "Test":
        return TestConfig()


print('Configurando la configuración de entrenamiento')
config = get_config('Train')
print("Configuración de entrenamiento establecida.")
test_config = get_config('Test')

print("Leyendo datos...")
raw_data, columns_name, norlizer = reader.read_raw_data("resultado.txt")
miss_data = reader.read_missing_data("resultado1.txt", norlizer, 3)

epochs = 150

print("Configurando las opciones de GPU...")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True
print("Opciones de GPU configuradas.")

with tf.compat.v1.Session(config=gpu_config) as session:
    print("Inicializando variables...")
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.compat.v1.variable_scope("LIMELSTM", reuse=None, initializer=initializer):
        model = LSTM_RSV(is_training=True, config=config, FLAGS=FLAGS)   	        # train model, is_trainable=True
    with tf.compat.v1.variable_scope("LIMELSTM", reuse=True, initializer=initializer):
        test_model = LSTM_RSV(is_training=False, config=test_config, FLAGS=FLAGS)   	        # test model, is_trainable=False

    print("Variables inicializadas.")
    tf.compat.v1.global_variables_initializer().run()
    model.assign_lr(session, 0.001)
    new_lr = model._lr
    for i in range(epochs):
        _ = run_epoch(session, model, miss_data, model.train_op)
        if i >= 10 and i % 10 == 0:
            if new_lr > 0.005:
                new_lr -= 0.003
            else:
                new_lr *= 0.5
            model.assign_lr(session, new_lr)
    prediction = run_epoch(session, test_model, miss_data, tf.no_op())

    imputed_data = np.concatenate((miss_data[0, :].reshape(1, -1), np.array(prediction)), axis=0)
    np.savetxt('imputed_data.txt', imputed_data, delimiter='\t')

