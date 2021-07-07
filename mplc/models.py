import numpy as np
import collections
from joblib import dump, load
from loguru import logger
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.metrics import accuracy_score, log_loss
from tensorflow.keras.backend import dot
from tensorflow.keras.layers import Dense
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms


class LogisticRegression(skLR):
    """
    Wrap sklearn Logistic regression in order to mimic the Keras API.
    """

    def __init__(self):
        super(LogisticRegression, self).__init__(max_iter=10000, warm_start=1, random_state=0)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x_train, y_train, batch_size, validation_data, epochs=1, verbose=False, callbacks=None):

        if callbacks:
            logger.debug('Callbacks parameters are ignored for LogisticRegression')

        history = super(LogisticRegression, self).fit(x_train, y_train)
        [loss, acc] = self.evaluate(x_train, y_train)
        [val_loss, val_acc] = self.evaluate(*validation_data)
        # Mimic Keras' history
        history.history = {
            'loss': [loss],
            'accuracy': [acc],
            'val_loss': [val_loss],
            'val_accuracy': [val_acc]
        }

        return history

    @property
    def trainable_weights(self):
        if self.coef_ is None:
            return None
        else:
            return np.concatenate((self.coef_, self.intercept_.reshape(1, 1)), axis=1)

    def evaluate(self, x_eval, y_eval, **kwargs):
        if self.coef_ is None:
            model_evaluation = [0] * 2
        else:
            loss = log_loss(y_eval, self.predict(x_eval))  # mimic keras model evaluation
            accuracy = self.score(x_eval, y_eval)
            model_evaluation = [loss, accuracy]

        return model_evaluation

    def save_weights(self, path):
        if self.coef_ is None:
            raise ValueError(
                'Coef and intercept are set to None, it seems the model has not been fit properly.')
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .npy')
            path.replace('.h5', '.npy')
        np.save(path, self.get_weights())

    def load_weights(self, path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .npy')
            path.replace('.h5', '.npy')
        weights = load(path)
        self.set_weights(weights)

    def get_weights(self):
        if self.coef_ is None:
            return None
        else:
            return np.concatenate((self.coef_, self.intercept_.reshape(1, 1)), axis=1)

    def set_weights(self, weights):
        if weights is None:
            self.coef_ = None
            self.intercept_ = None
        else:
            self.coef_ = np.array(weights[0][:-1]).reshape(1, -1)
            self.intercept_ = np.array(weights[0][-1]).reshape(1)

    def save_model(self, path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .joblib')
            path.replace('.h5', '.joblib')
        dump(self, path)

    @staticmethod
    def load_model(path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .joblib')
            path.replace('.h5', '.joblib')
        return load(path)

class cifar100_dataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform=[]):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):

        x = self.x[index]
        y = torch.tensor(int(self.y[index][0]))

        if self.transform:
            x = self.transform(x)

        return x, y

class ModelPytorch(nn.Module):
    def __init__(self):
        super(ModelPytorch, self).__init__()
        model = torchvision.models.vgg16()
        self.features = nn.Sequential(model.features)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1000)
        )
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


    def fit(self, x_train, y_train, batch_size, validation_data, epochs=1, verbose=False, callbacks=None):
        criterion = nn.CrossEntropyLoss()
        transform = transforms.Compose([transforms.ToTensor()])

        train_data = cifar100_dataset(x_train, y_train, transform)
        train_loader = data.DataLoader(train_data, batch_size=int(batch_size), shuffle=True)

        history = super(ModelPytorch, self).train()

        for batch_idx, (image, label) in enumerate(train_loader):
            images, labels = torch.autograd.Variable(image), torch.autograd.Variable(label)

            outputs = self.forward(images)
            loss = criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        [loss, acc] = self.evaluate(x_train, y_train)
        [val_loss, val_acc] = self.evaluate(*validation_data)
        # Mimic Keras' history
        history.history = {
            'loss': [loss],
            'accuracy': [acc],
            'val_loss': [val_loss],
            'val_accuracy': [val_acc]
        }

        return history

    def evaluate(self, x_eval, y_eval, **kwargs):
        criterion = nn.CrossEntropyLoss()
        transform = transforms.Compose([transforms.ToTensor()])

        test_data = cifar100_dataset(x_eval, y_eval, transform)
        test_loader = data.DataLoader(test_data, shuffle=True)

        self.eval()

        with torch.no_grad():

            y_true_np = []
            y_pred_np = []
            count=0
            for i, (images, labels) in enumerate(test_loader):
                count+= 1
                N = images.size(0)

                images = torch.autograd.Variable(images)
                labels = torch.autograd.Variable(labels)

                outputs = self(images)
                predictions = outputs.max(1, keepdim=True)[1]

                val_loss =+ criterion(outputs, labels).item()
                val_acc =+ (predictions.eq(labels.view_as(predictions)).sum().item() / N)

            model_evaluation = [val_loss/count, val_acc/count]

        return model_evaluation


    def save_weights(self, path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .pth')
            path.replace('.h5', '.pth')
        torch.save(self.state_dict(), path)


    def load_weights(self, path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .pth')
            path.replace('.h5', '.pth')
        weights = torch.load(path)
        self.set_weights(weights)


    def get_weights(self):
        self.state_dict()
        weights = []
        for layer in self.state_dict().keys(): 
            weights.append(self.state_dict()[layer].numpy())
        return weights


    def set_weights(self, weights):
        for i, layer in enumerate(self.state_dict().keys()):
                self.state_dict()[layer]= torch.Tensor(weights[i])


    def save_model(self, path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .pth')
            path.replace('.h5', '.pth')
        torch.save(self, path)


    @staticmethod
    def load_model(path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .pth')
            path.replace('.h5', '.pth')
        model = torch.load(path)
        return model.eval()


class EnsemblePredictionsModel():
    """
    Ensemble (average) predictions of several input models
    """

    def __init__(self, partners_model_list):
        self.partners_model_list = partners_model_list

    def fit(self, x_train, y_train, batch_size, validation_data, epochs=1, verbose=False, callbacks=None):
        pass

    def evaluate(self, x_eval, y_eval, **kwargs):
        predictions_list = []
        for model in self.partners_model_list:

            predictions = model.predict(x_eval)
            predictions_list.append(predictions)

        y_pred = np.mean(predictions_list, axis=0)

        loss = log_loss(y_eval, y_pred)

        y_true = np.argmax(y_eval, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        metric = accuracy_score(y_true, y_pred)

        return [loss, metric]

    def save_weights(self, path):
        pass


class NoiseAdaptationChannel(Dense):
    """
    Implement simple noise adaptation layer.
    References
        Goldberger & Ben-Reuven, Training deep neural-networks using a noise
        adaptation layer, ICLR 2017
        https://openreview.net/forum?id=H12GRgcxg
    # Arguments
        output_dim: int > 0
              default is input_dim which is known at build time
        See Dense layer for more arguments. There is no bias and the arguments
        `bias`, `b_regularizer`, `b_constraint` are not used.
    """

    def __init__(self, units=0, **kwargs):
        kwargs['use_bias'] = False
        if 'activation' not in kwargs:
            kwargs['activation'] = 'softmax'
        super(NoiseAdaptationChannel, self).__init__(units, **kwargs)

    def build(self, input_shape):
        if self.units == 0:
            self.units = input_shape[-1]
        super(NoiseAdaptationChannel, self).build(input_shape)

    def call(self, x):
        """
        :param x: the output of a baseline classifier model passed as an input
        It has a shape of (batch_size, input_dim) and
        baseline_output.sum(axis=-1) == 1
        :param mask: ignored
        :return: the baseline output corrected by a channel matrix
        """
        # convert W to the channel probability (stochastic) matrix
        # channel_matrix.sum(axis=-1) == 1
        # channel_matrix.shape == (input_dim, input_dim)
        channel_matrix = self.activation(self.kernel)

        # multiply the channel matrix with the baseline output:
        # channel_matrix[0,0] is the probability that baseline output 0 will get
        #  to channeled_output 0
        # channel_matrix[0,1] is the probability that baseline output 0 will get
        #  to channeled_output 1 ...
        # ...
        # channel_matrix[1,0] is the probability that baseline output 1 will get
        #  to channeled_output 0 ...
        #
        # we want output[b,0] = x[b,0] * channel_matrix[0,0] + \
        #                              x[b,1] * channel_matrix[1,0] + ...
        # so we do a dot product of axis -1 in x with axis 0 in channel_matrix
        return dot(x, channel_matrix)
