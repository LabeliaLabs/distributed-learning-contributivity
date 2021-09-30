import operator
import time

import numpy as np
import tensorflow as tf
from loguru import logger
from sklearn.metrics import confusion_matrix

from .basic_mpl import ALLOWED_PARAMETERS
from .utils import History
from .. import constants
from ..models import NoiseAdaptationChannel
from ..partner import Partner, PartnerMpl


####################################################
#
#       Optimized MPL approaches.
#
####################################################

# The few methods implemented in this module benefit from few improvements,
# which make them 5 to 10 times faster (even more for FastFedGrads).
#
# As these improvements rely on the tensorflow API, these methods can only be used with Keras models.
#
# The main improvements are:
#   - We build only one model, and never re-build the model. Instead, we keep the weights of each partners in tf tensor,
#   and load them into the model when needed.
#   - We benefit from tf.Dataset for the
#     batching, shuffling, and prefetching of data.
#   - We benefit from tf.function for the aggregation and training of an epoch. The fit_minibatch or fit_epoch method
#     must be redefined every tme we us the fit function, as it can initialize variable only at its first call.
#     For instance if you fit a first multi_partner_learning object, than a second, you will end up with an error.
#     The previously drawn
#     graph of the function fit_minibatch will be called, and asked to initialize  optimizer, and models weights.
#

class FastFedAvg:
    """ This class defines a multi_partner_learning method which is theoretically the same as FedAvg.
     It's just way faster, due to the use of tf.Dataset, and tf.function for the training.
     In this version each partner uses its own Adam optimizer."""

    name = 'FastFedAvg'

    def __init__(self, scenario, **kwargs):
        # Attributes related to the data and the model
        self.dataset = scenario.dataset
        self.partners_list = scenario.partners_list
        self.init_model_from = scenario.init_model_from
        self.use_saved_weights = scenario.use_saved_weights
        self.val_set = scenario.val_set
        self.test_set = scenario.test_set

        # Attributes to store results
        self.save_folder = scenario.save_folder

        # Attributes related to iterating at different levels
        self.epoch_count = scenario.epoch_count
        self.minibatch_count = scenario.minibatch_count
        self.gradient_updates_per_pass_count = scenario.gradient_updates_per_pass_count

        # Erase the default parameters (which mostly come from the scenario) if some parameters have been specified
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in ALLOWED_PARAMETERS)

        if self.partners_count == 1:
            raise ValueError('Only one partner is provided. Please use the dedicated SinglePartnerLearning class')

        # Convert partners to Mpl partners
        for partner in self.partners_list:
            assert isinstance(partner, Partner)
        partners_list = sorted(self.partners_list, key=operator.attrgetter("id"))
        logger.info(
            f"## Preparation of model's training on partners with ids: {['#' + str(p.id) for p in partners_list]}")
        self.partners_list = [PartnerMpl(partner, self) for partner in self.partners_list]

        # Attributes related to the _aggregation approach
        self.aggregator = self.init_aggregation_function(scenario.aggregation)
        assert self.aggregator.is_weights_constant, "For now, fast mpl methods cannot use changing aggregation " \
                                                    "weights. We are currently working on that. Please use the basic " \
                                                    "mpl methods, or contribute to the project to add this feature ! "
        self.aggregation_weights = tf.constant(self.aggregator.aggregation_weights, dtype=tf.float32)
        # Here we are doing a small hack. As we use tf function for training, we have to use tf functions for the
        # aggregation. So we just use the aggregator object to generate the aggregation weights, then use them in the
        # tf function for fitting the model. WE WONT USE THE AGGREGATOR ANYMORE. This must be improved

        # Convert the datasets into tf ones.
        self.dataset_name = self.dataset.name

        for p in self.partners_list:
            p.train_data = tf.data.Dataset.from_tensor_slices((p.x_train, p.y_train))
            p.train_data = p.train_data.shuffle(len(p.train_data), reshuffle_each_iteration=True)
            p.train_data = p.train_data.batch(p.batch_size, drop_remainder=True)
            p.train_data = p.train_data.batch(self.gradient_updates_per_pass_count)
            p.train_data = p.train_data.prefetch(1)
        self.train_dataset = [p.train_data for p in self.partners_list]

        self.val_data = tf.data.Dataset.from_tensor_slices((self.dataset.x_val, self.dataset.y_val))
        self.val_data = self.val_data.shuffle(len(self.dataset.y_val), reshuffle_each_iteration=True)
        self.val_data = self.val_data.batch(constants.DEFAULT_BATCH_SIZE).prefetch(1)

        self.test_data = tf.data.Dataset.from_tensor_slices((self.dataset.x_test, self.dataset.y_test))
        self.test_data = self.test_data.shuffle(len(self.dataset.y_test), reshuffle_each_iteration=True)
        self.test_data = self.test_data.batch(constants.DEFAULT_BATCH_SIZE).prefetch(1)

        # model related
        self.generate_new_model = self.dataset.generate_new_model
        self.model = self.init_model()
        if not isinstance(self.model, tf.keras.Model):
            raise Exception("This fast mpl method only handles Keras based model. Considere changing your model "
                            "definition, or using the basic equivalent of this mpl method")

        self.history = History(self)
        self.history.history['mpl_model'] = {}  # This history definition is not the same as current mplc def
        # , but it's close

        # for early stopping purpose. Those should be parameterizable
        self.is_early_stopping = True
        self.best = np.Inf
        self.monitor_value = 'val_loss'
        self.monitor_op = np.less
        self.min_delta = np.abs(constants.MIN_DELTA_FOR_EARLY_STOPPING)
        self.patience = constants.PATIENCE
        self.wait = 0
        self.epochs_index = 0

        # tf.function variables
        self.init_specific_tf_variable()

        self.learning_computation_time = 0
        self.timer = 0
        self.epoch_timer = 0

    def init_model(self):
        new_model = self.generate_new_model()

        if self.use_saved_weights:
            logger.info("Init model with previous coalition model")
            new_model.load_weights(self.init_model_from)
        else:
            logger.info("Init new model")

        return new_model

    def init_specific_tf_variable(self):
        # generate tf Variables in which we will store the model weights
        self.partners_weights = [[tf.Variable(initial_value=w.read_value()) for w in
                                  self.model.trainable_weights] for _ in self.partners_list]
        self.partners_optimizers = [self.model.optimizer.from_config(self.model.optimizer.get_config()) for _ in
                                    self.partners_list]

    def init_aggregation_function(self, aggregator):
        return aggregator(self)

    @property
    def partners_count(self):
        return len(self.partners_list)

    def log_epoch(self, epoch_number, history):
        logger.info(
            f'[{self.name}] > Epoch {str(epoch_number + 1).ljust(2)}/{self.epoch_count} -'
            f' {f"{np.round(time.time() - self.epoch_timer)} s.".ljust(6)} >'
            f' {" -- ".join(f"{key}: {str(np.round(value, 2)).ljust(5)}" for key, value in history.items())}')
        if not self.history.history['mpl_model']:
            self.history.history['mpl_model'] = {key: [value] for key, value in history.items()}
        else:
            for key, value in history.items():
                self.history.history['mpl_model'][key].append(value)

    def log_end_training(self):
        training_time = (time.time() - self.timer)
        self.history.score = self.model.evaluate(self.test_data, return_dict=True, verbose=False)
        logger.info(
            f'[{self.name}] > Training {self.epochs_index} epoch in {np.round(training_time, 3)} seconds.'
            f' Tests scores: '
            f'{" -- ".join(f"{key}: {np.around(value, 3)}" for key, value in self.history.score.items())}')
        self.learning_computation_time += training_time

    def get_epoch_history(self):
        # get the metrics computed. Here the metrics are computed on all the registered state (y, y_pred) stored,
        # which means on all the minibatches, for all partners.
        history = {m.name: float((m.result().numpy())) for m in self.model.metrics}
        # compute val accuracy and loss for global model
        val_hist = self.model.evaluate(self.val_data, return_dict=True, verbose=False)
        history.update({f'val_{key}': value for key, value in val_hist.items()})
        self.model.reset_metrics()
        return history

    def early_stop(self):
        """
        If the metric monitored decreases (or increases, depends on the metrics monitored)
        during patience epoch, we stop the learning.
        """
        if self.monitor_op(self.history.history['mpl_model'][self.monitor_value][-1] - self.min_delta, self.best):
            self.wait = 0
            self.best = self.history.history['mpl_model'][self.monitor_value][-1]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

    def fit(self):

        # TF function definition
        @tf.function
        def fit_minibatch(model, partners_minibatches, partners_optimizers, partners_weights, aggregation_weights):
            for p_id, minibatch in enumerate(partners_minibatches):  # minibatch == (x,y)
                # minibatch[0] in a tensor of shape=(number of batch, batch size, img).
                # We cannot iterate on tensors, so we convert this tensor to a list of
                # *number of batch* tensors with shape=(batch size, img)
                x_minibatch = tf.unstack(minibatch[0], axis=0)
                y_minibatch = tf.unstack(minibatch[1], axis=0)  # same here, with labels

                for model_w, partner_w in zip(model.trainable_weights,
                                              partners_weights[p_id]):  # set the model weights to partner's ones
                    model_w.assign(partner_w.read_value())

                for x, y in zip(x_minibatch, y_minibatch):  # iterate over batches
                    with tf.GradientTape() as tape:
                        y_pred = model(x)
                        loss = model.compiled_loss(y, y_pred)
                    model.compiled_metrics.update_state(y, y_pred)  # log the loss and accuracy
                    partners_optimizers[p_id].minimize(loss, model.trainable_weights,
                                                       tape=tape)  # perform local optimization

                for model_w, partner_w in zip(model.trainable_weights,
                                              partners_weights[p_id]):  # update the partner's weights
                    partner_w.assign(model_w.read_value())
            # at the end of the minibatch, aggregate all the local weights
            for i, weights_per_layer in enumerate(zip(*partners_weights)):
                model.trainable_weights[i].assign(tf.tensordot(weights_per_layer, aggregation_weights, [0, 0]))

            # inform the partners with the new weights
            for p_id in range(len(partners_minibatches)):
                for new_w, partner_w in zip(model.trainable_weights, partners_weights[p_id]):
                    partner_w.assign(new_w.read_value())

        # Execution
        self.timer = time.time()
        for e in range(self.epoch_count):
            self.epoch_timer = time.time()
            for partners_minibatches in zip(*self.train_dataset):  # <- partners_minibatches == [(x, y)] * nb_partners
                fit_minibatch(self.model,
                              partners_minibatches,
                              self.partners_optimizers,
                              self.partners_weights,
                              self.aggregation_weights)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch _history to self _history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()


class FastFedAvgSmodel(FastFedAvg):
    """
    This class defines a multi_partner_learning method which is theoretically the same as FedAvgSmodel.
     It's just way faster, due to the use of tf.Dataset, and tf.function for the training.

    """
    name = 'FastFedAvgSmodel'

    def __init__(self, scenario, pretrain_epochs=0, epsilon=0.5, **kwargs):
        super(FastFedAvgSmodel, self).__init__(scenario, **kwargs)

        self.pretrain_epochs = pretrain_epochs
        self.epsilon = epsilon
        self.smodel_list = [self.generate_smodel() for _ in self.partners_list]

    def generate_smodel(self):
        smodel = tf.keras.models.Sequential()
        smodel.add(tf.keras.Input(shape=(self.model.output_shape[-1],)))
        smodel.add(NoiseAdaptationChannel(name='s-model'))
        opt = self.model.optimizer.from_config(self.model.optimizer.get_config())
        smodel.compile(optimizer=opt, loss='categorical_crossentropy', metrics=self.dataset.model_metrics_names[1:])
        return smodel

    def fit(self):
        # TF function definition
        @tf.function
        def fit_pretrain_minibatch(model,
                                   partners_minibatches,
                                   partners_optimizers,
                                   partners_weights,
                                   aggregation_weights):
            for p_id, minibatch in enumerate(partners_minibatches):  # minibatch == (x,y)
                # minibatch[0] in a tensor of shape=(number of batch, batch size, img).
                # We cannot iterate on tensors, so we convert this tensor to a list of
                # *number of batch* tensors with shape=(batch size, img)
                x_minibatch = tf.unstack(minibatch[0], axis=0)
                y_minibatch = tf.unstack(minibatch[1], axis=0)  # same here, with labels

                for model_w, partner_w in zip(model.trainable_weights,
                                              partners_weights[p_id]):  # set the model weights to partner's ones
                    model_w.assign(partner_w.read_value())

                for x, y in zip(x_minibatch, y_minibatch):  # iterate over batches
                    with tf.GradientTape() as tape:
                        y_pred = model(x)
                        loss = model.compiled_loss(y, y_pred)
                    model.compiled_metrics.update_state(y, y_pred)  # log the loss and accuracy
                    partners_optimizers[p_id].minimize(loss, model.trainable_weights,
                                                       tape=tape)  # perform local optimization

                for model_w, partner_w in zip(model.trainable_weights,
                                              partners_weights[p_id]):  # update the partner's weights
                    partner_w.assign(model_w.read_value())
            # at the end of the minibatch, aggregate all the local weights
            for i, weights_per_layer in enumerate(zip(*partners_weights)):
                model.trainable_weights[i].assign(tf.tensordot(weights_per_layer, aggregation_weights, [0, 0]))

            # inform the partners with the new weights
            for p_id in range(len(partners_minibatches)):
                for new_w, partner_w in zip(model.trainable_weights, partners_weights[p_id]):
                    partner_w.assign(new_w.read_value())

        @tf.function
        def fit_minibatch(model, partners_minibatches, partners_optimizers, partners_weights, smodel_list,
                          aggregation_weights):
            for p_id, minibatch in enumerate(partners_minibatches):  # minibatch == (x,y)
                # minibatch[0] in a tensor of shape=(number of batch, batch size, img).
                # We cannot iterate on tensors, so we convert this tensor to a list of
                # *number of batch* tensors with shape=(batch size, img)
                x_minibatch = tf.unstack(minibatch[0], axis=0)
                y_minibatch = tf.unstack(minibatch[1], axis=0)  # same here, with labels

                for model_w, partner_w in zip(model.trainable_weights,
                                              partners_weights[p_id]):  # set the model weights to partner's ones
                    model_w.assign(partner_w.read_value())

                for x, y in zip(x_minibatch, y_minibatch):  # iterate over batches
                    s_model_p = smodel_list[p_id]
                    with tf.GradientTape(persistent=True) as tape:
                        yt_pred = model(x)
                        yo_pred = s_model_p(yt_pred)
                        loss = model.compiled_loss(y, yo_pred)
                    s_model_p.optimizer.minimize(loss, s_model_p.trainable_weights, tape=tape)
                    partners_optimizers[p_id].minimize(loss, model.trainable_weights,
                                                       tape=tape)
                    model.compiled_metrics.update_state(y, yo_pred)

                for model_w, partner_w in zip(model.trainable_weights,
                                              partners_weights[p_id]):  # update the partner's weights
                    partner_w.assign(model_w.read_value())
            # at the end of the minibatch, aggregate all the local weights
            for i, weights_per_layer in enumerate(zip(*partners_weights)):
                model.trainable_weights[i].assign(tf.tensordot(weights_per_layer, aggregation_weights, [0, 0]))

            # inform the partners with the new weights
            for p_id in range(len(partners_minibatches)):
                for new_w, partner_w in zip(model.trainable_weights, partners_weights[p_id]):
                    partner_w.assign(new_w.read_value())

        # Execution
        self.timer = time.time()
        if self.pretrain_epochs > 0:
            for p_e in range(self.pretrain_epochs):
                logger.info(f'[{self.name}] > Training {self.pretrain_epochs} pretrain epochs first.')
                self.epoch_timer = time.time()
                for partners_minibatches in zip(
                        *self.train_dataset):  # <- partners_minibatches == [(x, y)] * nb_partners
                    fit_pretrain_minibatch(self.model,
                                           partners_minibatches,
                                           self.partners_optimizers,
                                           self.partners_weights,
                                           self.aggregation_weights)
                epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
                # add the epoch _history to self _history, and log epoch number, and metrics values.
                self.log_epoch(p_e, epoch_history)
                self.epochs_index += 1

            for p in self.partners_list:
                confusion = confusion_matrix(np.argmax(p.y_train, axis=1),
                                             np.argmax(self.model.predict(p.x_train), axis=1),
                                             normalize='pred')
                p.noise_layer_weights = [np.log(confusion.T + 1e-8)]
        else:
            for p in self.partners_list:
                confusion = np.identity(10) * (1 - self.epsilon) + (self.epsilon / 10)
                p.noise_layer_weights = [np.log(confusion + 1e-8)]

        for p, smodel in zip(self.partners_list, self.smodel_list):
            smodel.set_weights(p.noise_layer_weights)

        for e in range(self.epoch_count - self.pretrain_epochs):
            self.epoch_timer = time.time()
            for partners_minibatches in zip(*self.train_dataset):  # <- partners_minibatches == [(x, y)] * nb_partners
                fit_minibatch(self.model,
                              partners_minibatches,
                              self.partners_optimizers,
                              self.partners_weights,
                              self.smodel_list,
                              self.aggregation_weights)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch _history to self _history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()


class FastFedGrad(FastFedAvg):
    """
    This class provides a multi-partner-learning method with is theoretically equivalent to FedGrads.
    Each partner computes a forward and backward propagation with its copy of the global model. Then the resulting
     gradients are aggregated, and a global optimizer uses the aggregated gradient to update the global model weights.
    Compared to FedAvg, this method requires more communications between partner (one each local batch against one
    every minibatch (what is to say every GRADIENT_UPDATE_PER_PASS local batches)
    But because this method allows the use of a state of the art optimizer (as Adam, Adamamx, etc), less epochs
    are required and so this method performs in fact very well.

    """
    name = 'FastFedGrad'

    def __init__(self, scenario, **kwargs):
        super(FastFedGrad, self).__init__(scenario, **kwargs)
        for p in self.partners_list:
            p.train_data = tf.data.Dataset.from_tensor_slices((p.x_train, p.y_train))
            p.train_data = p.train_data.shuffle(len(p.train_data), reshuffle_each_iteration=True)
            p.train_data = p.train_data.batch(p.batch_size, drop_remainder=True)
            p.train_data = p.train_data.prefetch(1)
        self.train_dataset = [p.train_data for p in self.partners_list]

    def init_specific_tf_variable(self):
        # generate tf Variables in which we will store the model weights
        self.partners_grads = [[tf.Variable(initial_value=w.read_value()) for w in self.model.trainable_weights]
                               for _ in self.partners_list]
        self.global_grad = [tf.Variable(initial_value=w.read_value()) for w in self.model.trainable_weights]
        self.partners_optimizers = [self.model.optimizer.from_config(self.model.optimizer.get_config()) for _ in
                                    self.partners_list]

    def fit(self):
        # TF function definition
        @tf.function
        def fit_epoch(model, train_dataset, partners_grads, global_grad, aggregation_weights):
            for minibatch in zip(*train_dataset):
                for p_id, (x_partner_batch, y_partner_batch) in enumerate(minibatch):
                    with tf.GradientTape() as tape:
                        y_pred = model(x_partner_batch)
                        loss = model.compiled_loss(y_partner_batch, y_pred)
                        par_grad = tape.gradient(loss, model.trainable_weights)
                        for holder, grad in zip(partners_grads[p_id], par_grad):
                            holder.assign(grad)
                    model.compiled_metrics.update_state(y_partner_batch, y_pred)

                for i, grads_per_layer in enumerate(zip(*partners_grads)):
                    global_grad[i].assign(tf.tensordot(grads_per_layer, aggregation_weights, [0, 0]))
                model.optimizer.apply_gradients(zip(global_grad, model.trainable_weights))

        # Execution
        self.timer = time.time()
        for e in range(self.epoch_count):
            self.epoch_timer = time.time()
            fit_epoch(self.model, self.train_dataset, self.partners_grads, self.global_grad, self.aggregation_weights)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch _history to self _history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()


class FastGradSmodel(FastFedGrad):
    """
        This class defines a multi_partner_learning method which combine the Smodel method, with the FedGrad algorithm
        for the training of the global model.
        For each partner the gradient of the whole model (global model + personal Smodel layer) is computed locally.
        The weights of the Smodel layer are optimized according to the partial gradient of this layer.
        The remaining part of the gradient is averaged over the partners, and used by a global optimizer to train the
        global model, whose updated weights are then send back to each partner.
    """

    name = 'FastGradSmodel'

    def __init__(self, scenario, pretrain_epochs=0, epsilon=0.5, **kwargs):
        super(FastGradSmodel, self).__init__(scenario, **kwargs)

        self.pretrain_epochs = pretrain_epochs
        self.epsilon = epsilon
        self.smodel_list = [self.generate_smodel() for _ in self.partners_list]

    def generate_smodel(self):
        smodel = tf.keras.models.Sequential()
        smodel.add(tf.keras.Input(shape=(self.model.output_shape[-1],)))
        smodel.add(NoiseAdaptationChannel(name='s-model'))
        opt = self.model.optimizer.from_config(self.model.optimizer.get_config())
        smodel.compile(optimizer=opt, loss='categorical_crossentropy', metrics=self.dataset.model_metrics_names[1:])
        return smodel

    def fit(self):
        @tf.function
        def fit_pretrain_epoch(model, train_dataset, partners_grads, global_grad, aggregation_weights):
            for minibatch in zip(*train_dataset):
                for p_id, (x_partner_batch, y_partner_batch) in enumerate(minibatch):
                    with tf.GradientTape() as tape:
                        y_pred = model(x_partner_batch)
                        loss = model.compiled_loss(y_partner_batch, y_pred)
                        par_grad = tape.gradient(loss, model.trainable_weights)
                        for holder, grad in zip(partners_grads[p_id], par_grad):
                            holder.assign(grad)
                    model.compiled_metrics.update_state(y_partner_batch, y_pred)
                for i, grads_per_layer in enumerate(zip(*partners_grads)):
                    global_grad[i].assign(tf.tensordot(grads_per_layer, aggregation_weights, [0, 0]))
                model.optimizer.apply_gradients(zip(global_grad, model.trainable_weights))

        @tf.function
        def fit_epoch(model, train_dataset, partners_grads, smodel_list, global_grad, aggregation_weights):
            for minibatch in zip(*train_dataset):
                for p_id, (x_partner_batch, y_partner_batch) in enumerate(minibatch):
                    with tf.GradientTape(persistent=True) as tape:
                        s_model_p = smodel_list[p_id]
                        yt_pred = model(x_partner_batch)
                        yo_pred = s_model_p(yt_pred)
                        loss = model.compiled_loss(y_partner_batch, yo_pred)

                    par_grad = tape.gradient(loss, model.trainable_weights)
                    for holder, grad in zip(partners_grads[p_id], par_grad):
                        holder.assign(grad)
                    s_model_p.optimizer.minimize(loss, s_model_p.trainable_weights, tape=tape)
                    model.compiled_metrics.update_state(y_partner_batch, yo_pred)
                for i, grads_per_layer in enumerate(zip(*partners_grads)):
                    global_grad[i].assign(tf.tensordot(grads_per_layer, aggregation_weights, [0, 0]))
                model.optimizer.apply_gradients(zip(global_grad, model.trainable_weights))

        # Execution
        self.timer = time.time()
        if self.pretrain_epochs > 0:
            for p_e in range(self.pretrain_epochs):
                logger.info(f'[{self.name}] > Training {self.pretrain_epochs} pretrain epochs first.')
                self.epoch_timer = time.time()
                fit_pretrain_epoch(self.model,
                                   self.train_dataset,
                                   self.partners_grads,
                                   self.global_grad,
                                   self.aggregation_weights)
                epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
                # add the epoch _history to self _history, and log epoch number, and metrics values.
                self.log_epoch(p_e, epoch_history)
                self.epochs_index += 1

            for p in self.partners_list:
                confusion = confusion_matrix(np.argmax(p.y_train, axis=1),
                                             np.argmax(self.model.predict(p.x_train), axis=1),
                                             normalize='pred')
                p.noise_layer_weights = [np.log(confusion.T + 1e-8)]
        else:
            for p in self.partners_list:
                confusion = np.identity(10) * (1 - self.epsilon) + (self.epsilon / 10)
                p.noise_layer_weights = [np.log(confusion + 1e-8)]

        for p, smodel in zip(self.partners_list, self.smodel_list):
            smodel.set_weights(p.noise_layer_weights)

        for e in range(self.epoch_count - self.pretrain_epochs):
            self.epoch_timer = time.time()
            fit_epoch(self.model,
                      self.train_dataset,
                      self.partners_grads,
                      self.smodel_list,
                      self.global_grad,
                      self.aggregation_weights)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch _history to self _history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()


class FastFedGDO(FastFedAvg):
    """
     This method is inspired from Federated gradient, but with modification on the local computation of the gradient.
    In this version we use a local optimizer (partner-specific) to do several minimization steps of the local-loss
    during a minibatch. We use the sum of these weighs-updates as the gradient which is sent to the global optimizer.
    The global optimizer aggregates these gradients-like which have been sent by the partners,
    and performs a optimization step with this aggregated gradient. FedGDO stands for Federated Gradient with double
    optimizers.
    """
    name = 'FastFedGDO'

    def __init__(self, scenario, reset_local_optims=False, global_optimiser=None, **kwargs):
        self.global_optimiser = global_optimiser
        self.reset_local_optims = reset_local_optims
        super(FastFedGDO, self).__init__(scenario, **kwargs)

    def init_specific_tf_variable(self):
        # generate tf Variables in which we will store the model weights
        self.model_stateholder = [tf.Variable(initial_value=w.read_value()) for w in self.model.trainable_weights]
        self.partners_grads = [[tf.Variable(initial_value=w.read_value()) for w in self.model.trainable_weights]
                               for _ in self.partners_list]
        self.global_grad = [tf.Variable(initial_value=w.read_value()) for w in self.model.trainable_weights]
        self.partners_optimizers = [self.model.optimizer.from_config(self.model.optimizer.get_config()) for _ in
                                    self.partners_list]
        if self.global_optimiser:
            # If the global optimizer is specified in the kwargs. If not the global optimizer used is the
            # default one, same as the local optimizer, and we don't need to re-compile the model
            self.model.compile(optimizer=self.global_optimiser)

    def fit(self):
        # TF function definition
        @tf.function
        def fit_minibatch(model, model_stateholder, partners_minibatches, partners_optimizers, partners_grads,
                          global_grad, aggregation_weights):
            for model_w, old_w in zip(model.trainable_weights, model_stateholder):  # store model weights
                old_w.assign(model_w.read_value())

            for p_id, minibatch in enumerate(partners_minibatches):  # minibatch == (x,y)
                # minibatch[0] in a tensor of shape=(number of batch, batch size, img).
                # We cannot iterate on tensors, so we convert this tensor to a list of
                # *number of batch* tensors with shape=(batch size, img)
                x_minibatch = tf.unstack(minibatch[0], axis=0)
                y_minibatch = tf.unstack(minibatch[1], axis=0)  # same here, with labels

                for x, y in zip(x_minibatch, y_minibatch):  # iterate over batches
                    with tf.GradientTape() as tape:
                        y_pred = model(x)
                        loss = model.compiled_loss(y, y_pred)
                    model.compiled_metrics.update_state(y, y_pred)  # log the loss and accuracy
                    partners_optimizers[p_id].minimize(loss, model.trainable_weights,
                                                       tape=tape)  # perform local optimizations
                # get the gradient as theta_before_minibatch - theta_after_minibatch
                for grad_per_layer, w_old, w_new in zip(partners_grads[p_id], model_stateholder,
                                                        model.trainable_weights):
                    grad_per_layer.assign((w_old - w_new))

                for model_w, old_w in zip(model.trainable_weights,
                                          model_stateholder):  # reset the model's weights for the next partner
                    model_w.assign(old_w.read_value())

            # at the end of the minibatch, aggregate all the local grads
            for i, grads_per_layer in enumerate(zip(*partners_grads)):
                global_grad[i].assign(tf.tensordot(grads_per_layer, aggregation_weights, [0, 0]))

            # perform one optimization update using the aggregated gradient
            model.optimizer.apply_gradients(
                zip(global_grad, model.trainable_weights))

        # Execution
        self.timer = time.time()
        for e in range(self.epoch_count):
            self.epoch_timer = time.time()
            for partners_minibatches in zip(*self.train_dataset):  # <- partners_minibatches == [(x, y)] * nb_partners
                if self.reset_local_optims:
                    self.partners_optimizers = [self.model.optimizer.from_config(self.model.optimizer.get_config()) for
                                                _ in
                                                self.partners_list]  # reset the local optimizers
                fit_minibatch(self.model,
                              self.model_stateholder,
                              partners_minibatches,
                              self.partners_optimizers,
                              self.partners_grads,
                              self.global_grad,
                              self.aggregation_weights)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch _history to self _history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()
