import wandb

from .runner import MulticlassRunner


class WandbDecor(MulticlassRunner):

    def __init__(self, runner: MulticlassRunner):
        self.runner = runner

    def zero_epoch_log(self, model, train_dataloader, val_dataloader=None):
        model.eval()
        results = {}

        results['train_loss'], results['train_accuracy'] = self.runner.eval(
            model, train_dataloader, verbose=0)

        if val_dataloader is not None:
            results['val_loss'], results['val_accuracy'] = self.runner.eval(
                model, val_dataloader, verbose=0)

        wandb.log(results)

        model.train()

    def fit(self, model, optimizer, train_dataloader, num_epochs, verbose=2):
        model.to(self.runner.device)
        model.train()

        acc_loss, acc_accuracy = 0, 0

        self.zero_epoch_log(model, train_dataloader)

        for epoch in range(num_epochs):

            running_loss, running_accuracy = self.runner.run_epoch(
                model, optimizer, train_dataloader)

            if (verbose > 1):
                print('\nTraining: Loss: {:.4f} Acc: {:.4f}'.format(
                    running_loss, running_accuracy))

            wandb.log({
                'train_loss': running_loss,
                'train_accuracy': running_accuracy,
            })

            acc_loss += running_loss
            acc_accuracy += running_accuracy

        return acc_loss / num_epochs, acc_accuracy / num_epochs

    def eval(self, model, eval_dataloader, verbose=2):
        eval_loss, eval_acc = self.runner.eval(model, eval_dataloader, verbose)
        wandb.log({'eval_loss': eval_loss, 'eval_accuracy': eval_acc})

        return eval_loss, eval_acc

    def fit_eval(self,
                 model,
                 optimizer,
                 train_dataloader,
                 val_dataloader,
                 num_epochs,
                 verbose=2):

        model.to(self.runner.device)

        self.zero_epoch_log(model, train_dataloader)
        for epoch in range(num_epochs):

            model.train()
            train_loss, train_accuracy = self.runner.run_epoch(
                model, optimizer, train_dataloader)

            model.eval()
            val_loss, val_accuracy = self.runner.eval(model,
                                                      val_dataloader,
                                                      verbose=0)
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            if (verbose > 1):
                print('\nTraining: Loss: {:.4f} Acc: {:.4f}'.format(
                    train_loss, train_accuracy))

    def predict(self, output):
        self.runner.predict(output)

    def compute_cost(self, output, target):
        self.runner.compute_cost(output, target)

    def prep_target(self, target):
        return self.runner.prep_target(target)
