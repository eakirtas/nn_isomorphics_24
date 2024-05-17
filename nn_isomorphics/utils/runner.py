from typing import Tuple

import numpy as np
import torch as T

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class MulticlassRunner():
    def __init__(self,
                 criterion=T.nn.NLLLoss,
                 device=DEVICE,
                 gradient_clipping=None,
                 scheduler=None):
        self.device = device
        self.criterion = criterion
        self.gradient_clipping = gradient_clipping
        self.scheduler = scheduler

    def run_epoch(self,
                  model: T.nn.Module,
                  optimizer: T.optim.Optimizer,
                  train_dataloader: T.utils.data.DataLoader,
                  type=T.cuda.FloatTensor):
        running_loss, running_accuracy, total = 0.0, 0.0, 0.0

        for inputs, target in train_dataloader:

            model.zero_grad()
            optimizer.zero_grad()

            inputs = inputs.to(self.device).type(type)
            target = self.prep_target(target.to(self.device))

            with T.set_grad_enabled(True):
                output = model(inputs)
                pred = self.predict(output)
                loss = self.compute_cost(output, target)

                loss.backward()

                if self.gradient_clipping is not None:
                    self.gradient_clipping(model.parameters())

                optimizer.step()

            running_loss += loss.detach()
            total += target.size(0)
            running_accuracy += (pred.detach()
                                 == target.detach()).sum().item() * 1.0

            del output, pred, loss

        if self.scheduler is not None:
            self.scheduler.step()

        return running_loss / total, running_accuracy / total

    def fit(self,
            model: T.nn.Module,
            optimizer: T.optim.Optimizer,
            train_dataloader: T.utils.data.DataLoader,
            num_epochs: int,
            verbose: int = 2,
            type=T.cuda.FloatTensor):

        model.to(self.device)
        model.train()

        acc_loss, acc_accuracy = 0, 0
        for _ in range(num_epochs):

            running_loss, running_accuracy = self.run_epoch(
                model, optimizer, train_dataloader, type)

            if (verbose > 1):
                print('\nTraining: Loss: {:.4f} Acc: {:.4f}'.format(
                    running_loss, running_accuracy))

            acc_loss += running_loss
            acc_accuracy += running_accuracy

        return acc_loss / num_epochs, acc_accuracy / num_epochs

    def eval(self, model, eval_dataloader, verbose=2, type=T.cuda.FloatTensor):

        model.to(self.device)
        model.eval()

        running_loss, running_corrects, total = 0.0, 0.0, 0.0

        for inputs, target in eval_dataloader:

            inputs = inputs.to(self.device).type(type)
            target = self.prep_target(target.to(self.device))

            with T.no_grad():
                output = model(inputs)
                pred = self.predict(output)
                loss = self.compute_cost(output, target)

            running_loss += loss.detach()
            total += target.size(0)
            running_corrects += (pred.detach()
                                 == target.detach()).sum().item() * 1.0

            del output, pred, loss

        final_loss, final_accuracy = running_loss / total, running_corrects / total

        if (verbose > 1):
            print('Evaluation: Loss: {:.4f} Acc: {:.4f}'.format(
                final_loss, final_accuracy))

        return final_loss, final_accuracy

    def predict(self, output: T.Tensor):
        return T.argmax(output, dim=1)

    def prep_target(self, target: T.Tensor):
        if len(target.size()) > 1:
            target = target.squeeze(dim=1)
        return target

    def compute_cost(self, output: T.Tensor, target: T.Tensor):
        return self.criterion(output, target)


class ImbalanceMulticlassRunner(MulticlassRunner):
    def __init__(self,
                 criterion=T.nn.NLLLoss,
                 device=DEVICE,
                 gradient_clipping=None,
                 scheduler=None):

        super().__init__(criterion=criterion,
                         device=device,
                         gradient_clipping=gradient_clipping,
                         scheduler=scheduler)

    def eval(
        self,
        model: T.nn.Module,
        eval_dataloader: T.utils.data.DataLoader,
        verbose: int = 2,
        type=T.cuda.FloatTensor
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:

        model.to(self.device)
        model.eval()

        running_loss, running_corrects, total = 0.0, 0.0, 0

        acc_pred = np.zeros(len(eval_dataloader.dataset), dtype=np.int)
        acc_target = np.zeros(len(eval_dataloader.dataset), dtype=np.int)

        for inputs, target in eval_dataloader:

            inputs = inputs.to(self.device).type(type)
            target = self.prep_target(target.to(self.device))

            with T.no_grad():
                output = model(inputs)
                pred = self.predict(output)
                loss = self.compute_cost(output, target)

            running_loss += loss.detach()
            total += target.size(0)
            running_corrects += (pred.detach()
                                 == target.detach()).sum().item() * 1.0

            acc_pred[total - target.size(0):total] = pred.detach().long().cpu(
            ).numpy().squeeze()
            acc_target[total - target.size(0):total] = target.detach().cpu(
            ).numpy().squeeze()

            del output, pred, loss

        final_loss, final_accuracy = running_loss / float(
            total), running_corrects / float(total)

        if (verbose > 1):
            print('Evaluation: Loss: {:.4f} Acc: {:.4f}'.format(
                final_loss, final_accuracy))

        return final_loss, final_accuracy, acc_target, acc_pred
