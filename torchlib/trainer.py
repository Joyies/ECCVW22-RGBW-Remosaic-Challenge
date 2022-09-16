import time
import logging

from tqdm import tqdm
import numpy as np

import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader

import torchlib.utils as utils

class Trainer(object):
    """Trainer"""

    class Parameters(object):
        def __init__(self, optimizer=optim.Adam,
                     optimizer_params={},
                     batch_size=1, lr=1e-4, wd=0, viz_smoothing=0.999,
                     viz_step=100,
                     checkpoint_interval=60):
            self.batch_size = batch_size
            self.lr = lr
            self.wd = wd
            self.optimizer = optimizer
            self.optimizer_params = optimizer_params
            self.viz_smoothing = viz_smoothing
            self.viz_step = viz_step
            self.checkpoint_interval = checkpoint_interval

    def cuda(self, b,gpuIdx):
        self._cuda = b
        self.device =None

        if b:
            self.device="cuda:"+str(gpuIdx)

            self.model.to(self.device)

            for l in self.criteria:
                self.criteria[l].to(self.device)

            for l in self.metrics:
                self.metrics[l].to(self.device)

            self.log.debug("swich to cuda")












    # constructor of Trainer
    #[in]
    #   trainset: an object of BayerDataset
    #   model
    #   criteria
    #   output
    #   model_params
    #   params
    #   metrics
    #   cuda
    #   fp16_scaling
    #   profile
    #   callbacks
    #   valset
    #   verbose
    #
    def __init__(self, trainset, model, criteria, output=None,
                 model_params=None,
                 params=None,
                 metrics={},
                 cuda=False,
                 gpuIdx=0,
                 fp16_scaling=-1,
                 profile=False,
                 callbacks=None,
                 valset=None,
                 verbose=False):

        self.verbose = verbose
        self.log = logging.getLogger("trainer")
        self.log.setLevel(logging.INFO)
        self.profile = profile
        if self.verbose:
            self.log.setLevel(logging.DEBUG)

        self.trainset = trainset
        self.valset = valset



        if params is None:
            self.params = Trainer.Parameters()
        else:
            self.params = params

        self.criteria = criteria
        self.metrics = metrics
        self.log_keys = list(criteria.keys()) + ["loss"]

        if metrics is not None:
            self.log_keys += list(self.metrics.keys())

        self.ema = utils.ExponentialMovingAverage(
            self.log_keys, alpha=self.params.viz_smoothing)
        self.averager = utils.Averager(self.log_keys)

        self.callbacks = callbacks


        self.model = model
        self.cuda(cuda,gpuIdx)


        params_to_optimize = self.model.parameters()

        self.optimizer = self.params.optimizer(
            [p for p in params_to_optimize if p.requires_grad],
            lr=self.params.lr,
            weight_decay=self.params.wd, **self.params.optimizer_params)

        if output is not None:
            self.checkpointer = utils.Checkpointer(
                output, self.model, self.optimizer,
                meta_params={"model": model_params},
                interval=self.params.checkpoint_interval)
        else:
            self.checkpointer = None

        self.train_loader = DataLoader(
            self.trainset, batch_size=self.params.batch_size,
            shuffle=True, num_workers=4, worker_init_fn=np.random.seed)

        if self.valset is not None:
            self.val_loader = DataLoader(
                self.valset, batch_size=min(self.params.batch_size, len(self.valset)),
                shuffle=True, num_workers=0,
                drop_last=True)  # so we have a fixed batch size for averaging in val
        else:
            self.val_loader = None

        self.log.debug("Model: {}\n".format(self.model))
        self.log.debug("Parameters to train:")
        for n, p in model.named_parameters():
            self.log.debug('  - {}'.format(n))

        if self.checkpointer is None:
            self.log.warn("No checkpointer provided, progress will not be saved.")

        self._set_model()


    # execute callbacks when an epoch begins
    #
    def _on_epoch_begin(self):
        self.log.debug("Epoch begins")



    # execute callbacks when an epoch ends
    #
    def _on_epoch_end(self, logs):
        if logs is None:
            return

        self.log.debug("Epoch ends")



    # exeute when a batch ends
    #
    def _on_batch_end(self, batch, batch_id, num_batches, logs):
        self.log.debug("Batch ends")





    # train the model for one epoch
    #[in]
    #   num_epochs: total num of epochs
    #
    def _train_one_epoch(self, num_epochs):

        self.model.train(True)

        with tqdm(total=len(self.train_loader), unit=' batches') as pbar:
            pbar.set_description("Epoch {}/{}".format(self.epoch+1, num_epochs if num_epochs > 0 else "--"))  # shwow the current epoch / total epoch


            for batch_id, batch in enumerate(self.train_loader):

                # batch_v = utils.make_variable(batch, cuda=self._cuda)

                batch_v = utils.make_variable(batch, device=self.device)


                self.model.zero_grad()

                start = time.time()

###################################################                     predict output by the model                         #########################################\


                # batch_v is a dict structure composed of {'mosaic','mask','target'}
                # 'mosaic' is the input of the model                            [batch_size,3,h,w]
                # 'mask' is used to create the mosaic from a RGB image          [batch_size,3,h,w]
                # 'target' is the ground truth (an rgb image)                   [batch_size,3,h,w]
                #
                # the h and w of batch_v may be different from the orignial image becaues of the cropping in dataset.py
                #
                output = self.model(batch_v)





                ###################################################                     predict output by the model                         #########################################\



                elapsed = (time.time() - start)*1000.0
                self.log.debug("Forward {:.1f} ms".format(elapsed))

                # Compute all losses
                c_out = []
                for k in self.criteria.keys():
                    crit = self.criteria[k](batch_v, output)
                    c_out.append(crit)
                    self.ema.update(k, crit.detach().cpu().item())


                loss = sum(c_out)
                self.ema.update("loss", loss.detach().cpu().data.item())



                # Compute all metrics
                for k in self.metrics.keys():
                    m = self.metrics[k](batch_v, output)
                    self.ema.update(k, m.detach().cpu().item())




                loss.backward()



                self.optimizer.step()



                logs = {k: self.ema[k] for k in self.log_keys}
                pbar.set_postfix(logs)

                if pbar.n % self.params.viz_step == 0:
                    self._on_batch_end(batch, batch_id, len(self.train_loader), logs)

                pbar.update(1)

                if self.checkpointer is not None:
                    self.checkpointer.periodic_checkpoint(self.epoch)













    def _set_model(self):
        if self.checkpointer:
            chkpt_name, epoch = self.checkpointer.load_latest()
            if chkpt_name is None:
                self.log.info("Starting training from scratch")
            else:
                self.log.info("Resuming from latest checkpoint {}.".format(chkpt_name))
        else:
            epoch = 0
        self.epoch = epoch

    def override_parameters(self, checkpoint):
        if checkpoint and self.checkpointer:
            self.log.info("Overriding parameters:")
            names = self.checkpointer.override_params(args.checkpoint)
            for n in names:
                self.log.info("  - {}".format(n))





    # train the NN model
    #[in]
    #   num_epochs: num of epochs to train the model
    #
    def train(self, num_epochs=-1):

        best_val_loss = None


        try:
            while True:
                # Training
                self._on_epoch_begin()


################################                  start training                        ################################

                self._train_one_epoch(num_epochs)

################################                  end training                        ################################



                # Validation
                val_loss, val_logs = self._run_validation(num_epochs)

                if best_val_loss is None:
                    best_val_loss = val_loss                                                                    # save the best validation loss results
                if self.checkpointer and val_loss and val_loss <= best_val_loss:
                    self.checkpointer.save_best(self.epoch)                                                     # save the best checkpoint


                self.epoch += 1

                self._on_epoch_end(val_logs)

                if num_epochs > 0 and self.epoch >= num_epochs:
                    self.log.info("Ending training at epoch {} of {}".format(self.epoch, num_epochs))

        except KeyboardInterrupt:
            self.log.info("training interrupted")









    # run validation
    #[in]
    #   num_epochs:
    #
    def _run_validation(self, num_epochs):

        count = self.params.batch_size
        logs = None

        if self.val_loader is None:
            return None, logs

        with th.no_grad():
            self.model.train(False)
            self.averager.reset()

            logs = None

            with tqdm(total=len(self.val_loader), unit=' batches') as pbar:

                pbar.set_description("Epoch {}/{} (val)".format(self.epoch+1, num_epochs if num_epochs > 0 else "--"))

                for batch_id, batch in enumerate(self.val_loader):
                    batch_v = utils.make_variable(batch, device=self.device)
                    output = self.model(batch_v)

                    # Compute all losses
                    c_out = []
                    for k in self.criteria.keys():
                        crit = self.criteria[k](batch_v, output)
                        c_out.append(crit)
                        self.averager.update(k, crit.cpu().data.item())
                    loss = sum(c_out)
                    self.averager.update("loss", loss.cpu().data.item(), count)

                    # Compute all metrics
                    for k in self.metrics.keys():
                        m = self.metrics[k](batch_v, output)
                        self.averager.update(k, m.cpu().data.item())

                    pbar.update(1)

                    logs = {k: self.averager[k] for k in self.log_keys}
                    pbar.set_postfix(logs)

                return self.averager["loss"], logs
