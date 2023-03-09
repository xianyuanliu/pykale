import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from kale.pipeline.domain_adapter import (  # get_metrics_from_parameter_dict,
    get_aggregated_metrics,
    get_aggregated_metrics_from_dict,
)
from kale.predict import losses


class BaseTrainer(pl.LightningModule):
    def __init__(self, optimizer, max_epochs, init_lr=0.001, adapt_lr=False):
        super(BaseTrainer, self).__init__()
        self._init_lr = init_lr
        self._optimizer_params = optimizer
        self._adapt_lr = adapt_lr
        self._max_epochs = max_epochs

    def configure_optimizers(self):
        """
        Config adam as default optimizer.
        """
        if self._optimizer_params is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self._init_lr)
            return [optimizer]
        if self._optimizer_params["type"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self._init_lr, **self._optimizer_params["optim_params"],)
            return [optimizer]
        if self._optimizer_params["type"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self._init_lr, **self._optimizer_params["optim_params"],)

            if self._adapt_lr:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._max_epochs, last_epoch=-1)
                return [optimizer], [scheduler]
            return [optimizer]
        raise NotImplementedError(f"Unknown optimizer type {self._optimizer_params['type']}")

    def forward(self, x):
        """
        Same as :meth:`torch.nn.Module.forward()`
        """
        raise NotImplementedError("Forward pass needs to be defined.")

    def training_step(self, train_batch, batch_idx):
        """
        Compute and return the training loss on one step
        """
        x, y = train_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.logger.log_metrics({"train_loss": loss}, self.global_step)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Compute and return the validation loss on one step
        """
        x, y = val_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Compute and return the test loss on one step
        """
        x, y = test_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        return loss


class ActionRecogTrainer(BaseTrainer):
    def __init__(self, feature_extractor, task_classifier, image_modality, batch_size, **kwargs):
        super(ActionRecogTrainer, self).__init__(**kwargs)
        self.feat = feature_extractor
        self.classifier = task_classifier
        self.image_modality = image_modality
        self._batch_size = batch_size
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]

    def forward(self, x):
        if self.rgb_feat is not None:
            x = self.rgb_feat(x)
        else:
            x = self.flow_feat(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return x, output

    def compute_loss(self, batch, split_name="valid"):
        # if len(batch) == 3:  # Video, audio, labels
        #     x, _, y = batch
        # elif len(batch) > 3:
        #     x = batch["video"]
        #     y = batch["label"]
        #     # print(split_name, batch["clip_index"], batch["video_name"])  # for debugging
        # else:  # Video, labels
        #     x, y = batch
        x, y, _ = batch
        y = y[0]  # only one label
        feat, y_hat = self.forward(x)

        # # Saving i3d output features for tsne
        # df = pd.DataFrame(feat.detach().cpu().numpy())
        # df["class_id"] = y.detach().cpu().numpy()
        # save_path = "D:/Projects/GitHub/tools/tsne/feats/ar/adl-src-all.csv"
        # df.to_csv(save_path, index=False, header=False, mode="a")

        loss, log_metrics = self.get_loss_log_metrics(split_name, y_hat, y)
        return loss, log_metrics

    def training_step(self, batch, batch_idx):
        loss, log_metrics = self.compute_loss(batch, "train")
        log_metrics = get_aggregated_metrics_from_dict(log_metrics)
        log_metrics["train_loss"] = loss

        for key in log_metrics:
            self.log(key, log_metrics[key])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_metrics = self.compute_loss(batch, split_name="valid")
        log_metrics["valid_loss"] = loss
        return log_metrics

    def validation_epoch_end(self, outputs):
        metrics_to_log = self.create_metrics_log("valid")
        log_dict = get_aggregated_metrics(metrics_to_log, outputs)

        for key in log_dict:
            self.log(key, log_dict[key], prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, log_metrics = self.compute_loss(batch, split_name="test")
        log_metrics["test_loss"] = loss
        return log_metrics

    def test_epoch_end(self, outputs):
        metrics_at_test = self.create_metrics_log("test")

        log_dict = get_aggregated_metrics(metrics_at_test, outputs)
        for key in log_dict:
            self.log(key, log_dict[key], prog_bar=True)

    def create_metrics_log(self, split_name):
        metrics_to_log = (
            "{}_loss".format(split_name),
            # "{}_acc".format(split_name),
            "{}_top1_acc".format(split_name),
            "{}_top5_acc".format(split_name),
        )
        return metrics_to_log

    def get_loss_log_metrics(self, split_name, y_hat, y):
        """Get the loss, top-k accuracy and metrics for a given split."""

        task_loss, ok = losses.cross_entropy_logits(y_hat, y)
        prec1, prec5 = losses.topk_accuracy(y_hat, y, topk=(1, 5))

        log_metrics = {
            # f"{split_name}_acc": ok,
            f"{split_name}_top1_acc": prec1,
            f"{split_name}_top5_acc": prec5,
        }
        return task_loss, log_metrics


class ActionFeatRecogTrainer(ActionRecogTrainer):
    def __init__(self, dataset, feature_extractor, task_classifier, image_modality, batch_size, **kwargs):
        super(ActionFeatRecogTrainer, self).__init__(
            feature_extractor, task_classifier, image_modality, batch_size, **kwargs
        )
        self.dataset = dataset
        self.dataset.prepare_data_loaders()
        self.feat = feature_extractor
        self.classifier = task_classifier
        self.image_modality = image_modality
        self._batch_size = batch_size
        self.rgb_feat = self.feat["rgb"]
        self.flow_feat = self.feat["flow"]
        self.audio_feat = self.feat["audio"]

    def forward(self, x):
        if self.rgb_feat is not None:
            x_rgb = self.rgb_feat(x["rgb"])
        if self.flow_feat is not None:
            x_flow = self.flow_feat(x["flow"])
        if self.audio_feat is not None:
            x_audio = self.audio_feat(x["audio"])

        x = torch.cat((x_rgb, x_flow, x_audio), dim=-1)

        x = x.view(x.size(0), -1)

        output = self.classifier(x)
        return x, output

    def compute_loss(self, batch, split_name="valid"):
        (x_rgb, y, _), (x_flow, _, _), (x_audio, _, _) = batch

        feat, y_hat = self.forward({"rgb": x_rgb, "flow": x_flow, "audio": x_audio})

        # # Saving output features for tsne
        # df = pd.DataFrame(feat.detach().cpu().numpy())
        # df["class_id"] = y.detach().cpu().numpy()
        # save_path = "D:/Projects/GitHub/tools/tsne/feats/ar/adl-src-all.csv"
        # df.to_csv(save_path, index=False, header=False, mode="a")

        loss, log_metrics = self.get_loss_log_metrics(split_name, y_hat, y)
        return loss, log_metrics

    def get_loss_log_metrics(self, split_name, y_hat, y):
        """Get the loss, top-k accuracy and metrics for a given split."""

        loss_cls_verb, ok_verb = losses.cross_entropy_logits(y_hat[0], y[0])
        loss_cls_noun, ok_noun = losses.cross_entropy_logits(y_hat[1], y[1])

        prec1_verb, prec5_verb = losses.topk_accuracy(y_hat[0], y[0], topk=(1, 5))
        prec1_noun, prec5_noun = losses.topk_accuracy(y_hat[1], y[1], topk=(1, 5))
        prec1_action, prec5_action = losses.multitask_topk_accuracy((y_hat[0], y_hat[1]), (y[0], y[1]), topk=(1, 5))
        task_loss = loss_cls_verb + loss_cls_noun

        log_metrics = {
            f"{split_name}_loss_cls_verb": loss_cls_verb,
            f"{split_name}_loss_cls_noun": loss_cls_noun,
            f"{split_name}_verb_top1_acc": prec1_verb,
            f"{split_name}_verb_top5_acc": prec5_verb,
            f"{split_name}_noun_top1_acc": prec1_noun,
            f"{split_name}_noun_top5_acc": prec5_noun,
            f"{split_name}_action_top1_acc": prec1_action,
            f"{split_name}_action_top5_acc": prec5_action,
        }

        return task_loss, log_metrics

    def create_metrics_log(self, split_name):
        metrics_to_log = (
            "{}_loss".format(split_name),
            "{}_loss_cls_verb".format(split_name),
            "{}_loss_cls_noun".format(split_name),
            "{}_verb_top1_acc".format(split_name),
            "{}_verb_top5_acc".format(split_name),
            "{}_noun_top1_acc".format(split_name),
            "{}_noun_top5_acc".format(split_name),
            "{}_action_top1_acc".format(split_name),
            "{}_action_top5_acc".format(split_name),
        )
        return metrics_to_log

    def train_dataloader(self):
        dataloader = self.dataset.get_multi_loaders(split="train", batch_size=self._batch_size)
        # self._nb_training_batches = len(dataloader)
        return dataloader

    def val_dataloader(self):
        return self.dataset.get_multi_loaders(split="valid", batch_size=self._batch_size)

    def test_dataloader(self):
        return self.dataset.get_multi_loaders(split="test", batch_size=self._batch_size)
