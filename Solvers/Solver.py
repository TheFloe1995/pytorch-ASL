import os
from datetime import datetime as dt
import torch
from torch.autograd import Variable
from warnings import warn
import numpy as np


class Solver(object):
    """
    Solver/Trainer class for the model. The model is passed on initialization and stored in self.model.
    All static training parameters are stored in a dictionary.
    The parameters and the model should not be externally changed while using a Solver.
    A Solver is meant to train only one model based on a non-changing set of static training parameters.
    """

    def __init__(self, params, model):
        """
        Initilialize a new Solver object.
        :param params: Dictionary containing all static training hyper paramters
        :param model: The model that is to be trained
        """
        if not type(params) is dict:
            raise Exception("Error: The passed 'params' object is not a dictionary.")

        self.params = params
        self.model = model

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            warn("Cuda is not available!")

        target_layers = self._extract_target_layers(params['layer_names'])
        self._disable_gradiens(target_layers)

        optimizer = params['optimizer']
        self.optimizer = optimizer(target_layers.parameters(), **(params['optimizer_args']))

        scheduler = params["scheduler"]
        self.scheduler = scheduler(self.optimizer, **(params['scheduler_args']))

        self.loss_function = params["loss_function"]()

    def train(self, training_data_loader, validation_data_loader, small_validation_data_loader, epoch_count, log_frequency = 1, log_loss=False, verbose=False):
        """
        Trains the model associated with the Solver object using the given training and validation data over the specified number of epochs.
        The training accuracy, the validation accuracy and optionally the training loss is logged during training and returned at the end.

        :param training_data_loader: DataLoader object that provides the data for training.
        :type  training_data_loader: torch.utils.data.DataLoader

        :param validation_data_loader: DataLoader object that provides the data for validation.
        :type  validation_data_loader: torch.utils.data.DataLoader

        :param epoch_count: Number of epochs to train. In each epoch, the method iterates over all batches that are randomly provided by the training_data_loader. This equals the number of the original training samples (images).
        :type  epoch_count: int

        :param log_frequency: The method logs the training and validation accuracy log_frequency (<int>) times per epoch. In any case, the validation and test accuracy is calculated and logged at the end of each epoch.
        :type  log_frequency: int > 0

        :param log_loss: Enable or disable logging of the training loss after each forward pass.
        :type  log_loss: bool

        :param verbose: Enable or disable a more verbose output during training.
        :type: verbose: bool

        :return: log, best_weights: Training log and the model weights that performed best on the validation set during training
        :type: log: Dictionary
        :type: weights: Dictionary
        """
        start_time = dt.now()
        print("START")
        print("Time: ", start_time.strftime("%H:%M:%S"))

        best_val_acc = 0.0
        best_weights = self.model.state_dict()

        log = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'small_val_acc': []}

        iter_per_epoch = len(training_data_loader)
        log_iterations = self._get_log_iterations(iter_per_epoch, log_frequency)
        average_loss = 0.0

        for epoch in range(epoch_count):
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, epoch_count - 1))

            running_loss = 0.0
            running_train_acc = 0.0

            if self.params["requires_metric"]:
                self.scheduler.step(average_loss)
            else:
                self.scheduler.step()

            for i, (inputs, targets) in enumerate(training_data_loader):
                inputs, targets = self._prepare_data_for_usage(inputs, targets)

                self.optimizer.zero_grad()

                outputs, predictions = self.forward_and_predict(inputs, targets)

                loss = self._backward_and_optimize(outputs, targets)
                running_loss += loss
                if log_loss:
                    log['train_loss'].append(loss)

                train_acc = self.calc_accuracy(predictions, targets)
                running_train_acc += train_acc

                if i in log_iterations:
                    log['train_acc'].append(train_acc)

                    small_val_acc = self.test(small_validation_data_loader)

                    log['small_val_acc'].append(small_val_acc)

                    if verbose:
                        print("Iteration %d/%d: loss: %.3f,\t train acc: %.3f,\t small val acc: %.3f" % (i, iter_per_epoch - 1, loss, train_acc, small_val_acc))
                        self._print_elapsed_time(start_time)

            average_loss = running_loss / iter_per_epoch
            average_train_acc = running_train_acc / iter_per_epoch

            val_acc = self.test(validation_data_loader)
            log['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.model.state_dict()

            print("Finished epoch. Average loss: %.3f,\t Average train acc: %.3f, Final val acc: %.3f" % (average_loss, average_train_acc, val_acc))
            if verbose:
                self._print_elapsed_time(start_time)
            print()

        print("FINISH")
        print("Best validation accuracy:", best_val_acc)

        return log, best_weights

    def forward_and_predict(self, inputs, targets):
        outputs = self.model(inputs)
        _, predictions = torch.max(outputs, 1)
        return outputs, predictions

    def forward_and_predict_top_k(self, inputs, targets, top_k):
        outputs = self.model(inputs)
        _, predictions = torch.topk(outputs, top_k)
        return outputs, predictions

    def calc_accuracy(self, predictions, targets):
        """
        Calculates the classification accuracy given a set of predictions and a set of targets.
        """
        return np.mean((predictions == targets).data.cpu().numpy())

    def calc_accuracy_top_k(self, predictions, targets):
        """ Calculates the top5 accuracy """
        acc = 0.0
        for index in range(len(targets)):
            target = targets.data[index]
            prediction = predictions[index].data.cpu().numpy()
            if(target in prediction):
                acc += 1
        return acc/len(targets)

    def set_model_params(self, state_dict):
        self.model.load_state_dict(state_dict)

    def test(self, loader, create_confusion_matrix=False, num_classes = 24):
        self.model.eval()
        acc = 0.0
        if create_confusion_matrix:
            cm = np.zeros((num_classes, num_classes))
        for inputs, targets in loader:
            inputs, targets = self._prepare_data_for_usage(inputs, targets)
            outputs, predictions = self.forward_and_predict(inputs, targets)
            if create_confusion_matrix:
                for t,p in zip(targets.cpu().data.numpy(), predictions.cpu().data.numpy()):
                    cm[t, p] += 1

            acc += self.calc_accuracy(predictions, targets)
        self.model.train()
        acc = acc / len(loader)
        if create_confusion_matrix:
            return acc, cm
        else:
            return acc

    def test_top_k(self, loader, top_k=5):
        self.model.eval()
        acc = 0.0
        for inputs, targets in loader:
            inputs, targets = self._prepare_data_for_usage(inputs, targets)
            outputs, predictions = self.forward_and_predict_top_k(inputs, targets, top_k)
            acc += self.calc_accuracy_top_k(predictions, targets)
        acc = acc / len(loader)
        self.model.train()
        return acc

    def _extract_target_layers(self, layer_names):
        if layer_names == "ALL":
            return self.model
        else:
            return torch.nn.ModuleList([getattr(self.model, layer_name) for layer_name in layer_names])

    def _disable_gradiens(self, required_layers):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in required_layers.parameters():
            param.requires_grad = True

    def _get_log_iterations(self, iter_per_epoch, log_frequency):
        logging_epochs = [(iter_per_epoch - 1) - log * (iter_per_epoch // log_frequency) for log in range(0, log_frequency)]
        return logging_epochs

    def _prepare_data_for_usage(self, inputs, targets):
        inputs, targets = Variable(inputs), Variable(targets)
        if self.model.is_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        return inputs, targets

    def _backward_and_optimize(self, outputs, targets):
        loss = self.loss_function(outputs, targets)
        loss.backward()
        self.optimizer.step()
        loss = loss.data.cpu().numpy()[0]
        return loss

    def _print_elapsed_time(self, start_time):
        time_elapsed = dt.now() - start_time
        hours, remainder = divmod(time_elapsed.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("Time since start: {:.0f}h {:.0f}m {:.0f}s".format(hours, minutes, seconds))


