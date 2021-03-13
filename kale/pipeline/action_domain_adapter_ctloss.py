import torch

import kale.predict.losses as losses
from kale.pipeline.domain_adapter_ctloss import (
    BaseMMDLike,
    CDANtrainer,
    DANNtrainer,
    Method,
    ReverseLayerF,
    set_requires_grad,
    WDGRLtrainer, get_aggregated_metrics_from_dict, get_metrics_from_parameter_dict,
)


def create_mmd_based_4video(
        method: Method, dataset, image_modality, feature_extractor, task_classifier, **train_params
):
    """MMD-based deep learning methods for domain adaptation: DAN and JAN
    """
    if not method.is_mmd_method():
        raise ValueError(f"Unsupported MMD method: {method}")
    if method is Method.DAN:
        return DANtrainer4Video(
            dataset=dataset,
            image_modality=image_modality,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            method=method,
            **train_params
        )
    if method is Method.JAN:
        return JANtrainer4Video(
            dataset=dataset,
            image_modality=image_modality,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            method=method,
            kernel_mul=[2.0, 2.0],
            kernel_num=[5, 1],
            **train_params,
        )


def create_dann_like_4video(
        method: Method, dataset, image_modality, feature_extractor, task_classifier, critic, **train_params
):
    """DANN-based deep learning methods for domain adaptation: DANN, CDAN, CDAN+E
    """
    # if dataset.is_semi_supervised():
    #     return create_fewshot_trainer(
    #         method, dataset, feature_extractor, task_classifier, critic, **train_params
    #     )

    if method.is_dann_method():
        alpha = 0 if method is Method.Source else 1
        return DANNtrainer4Video(
            alpha=alpha,
            image_modality=image_modality,
            dataset=dataset,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            **train_params,
        )
    elif method.is_cdan_method():
        return CDANtrainer4Video(
            dataset=dataset,
            image_modality=image_modality,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            use_entropy=method is Method.CDAN_E,
            **train_params,
        )
    elif method is Method.WDGRL:
        return WDGRLtrainer4Video(
            dataset=dataset,
            image_modality=image_modality,
            feature_extractor=feature_extractor,
            task_classifier=task_classifier,
            critic=critic,
            method=method,
            **train_params,
        )
    # elif method is Method.WDGRLMod:
    #     return WDGRLtrainerMod4Video(
    #         dataset=dataset,
    #         image_modality=image_modality,
    #         feature_extractor=feature_extractor,
    #         task_classifier=task_classifier,
    #         critic=critic,
    #         method=method,
    #         **train_params,
    #     )
    else:
        raise ValueError(f"Unsupported method: {method}")


class BaseMMDLike4Video(BaseMMDLike):
    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            kernel_mul=2.0,
            kernel_num=5,
            **base_params,
    ):
        """Common API for MME-based deep learning DA methods: DAN, JAN
        """

        super().__init__(dataset, feature_extractor, task_classifier, kernel_mul, kernel_num, **base_params)
        self.image_modality = image_modality
        self.rgb_feat = self.feat['rgb']
        self.flow_feat = self.feat['flow']

    def forward(self, x):
        if self.feat is not None:
            if self.image_modality in ['rgb', 'flow']:
                if self.rgb_feat is not None:
                    x = self.rgb_feat(x)
                else:
                    x = self.flow_feat(x)
                x = x.view(x.size(0), -1)
                class_output = self.classifier(x)
                return x, class_output

            elif self.image_modality == 'joint':
                x_rgb = self.rgb_feat(x['rgb'])
                x_flow = self.flow_feat(x['flow'])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
                x_flow = x_flow.view(x_flow.size(0), -1)
                x = torch.cat((x_rgb, x_flow), dim=1)
                class_output = self.classifier(x)
                return [x_rgb, x_flow], class_output

    def compute_loss(self, batch, split_name="V"):
        if self.image_modality == 'joint' and len(batch) == 4:
            (x_s_rgb, y_s), (x_s_flow, y_s_flow), (x_tu_rgb, y_tu), (x_tu_flow, y_tu_flow) = batch
            [phi_s_rgb, phi_s_flow], y_hat = self.forward({'rgb': x_s_rgb, 'flow': x_s_flow})
            [phi_t_rgb, phi_t_flow], y_t_hat = self.forward({'rgb': x_tu_rgb, 'flow': x_tu_flow})
            mmd_rgb = self._compute_mmd(phi_s_rgb, phi_t_rgb, y_hat, y_t_hat)
            mmd_flow = self._compute_mmd(phi_s_flow, phi_t_flow, y_hat, y_t_hat)
            mmd = mmd_rgb + mmd_flow
        elif self.image_modality in ['rgb', 'flow'] and len(batch) == 2:
            (x_s, y_s), (x_tu, y_tu) = batch
            phi_s, y_hat = self.forward(x_s)
            phi_t, y_t_hat = self.forward(x_tu)
            mmd = self._compute_mmd(phi_s, phi_t, y_hat, y_t_hat)
        else:
            raise NotImplementedError("Batch len is {}. Check the Dataloader.".format(len(batch)))

        # Uncomment when checking whether rgb & flow labels are equal.
        # print('rgb_s:{}, flow_s:{}, rgb_f:{}, flow_f:{}'.format(y_s, y_s_flow, y_tu, y_tu_flow))
        # print('equal: {}/{}'.format(torch.all(torch.eq(y_s, y_s_flow)), torch.all(torch.eq(y_tu, y_tu_flow))))

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        task_loss = loss_cls
        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_mmd": mmd,
        }
        return task_loss, mmd, log_metrics


class DANtrainer4Video(BaseMMDLike4Video):
    """This is an implementation of DAN for video data.
    """

    def __init__(self, dataset, image_modality, feature_extractor, task_classifier, **base_params):
        super().__init__(dataset, image_modality, feature_extractor, task_classifier, **base_params)

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        batch_size = int(phi_s.size()[0])
        kernels = losses.gaussian_kernel(
            phi_s, phi_t, kernel_mul=self._kernel_mul, kernel_num=self._kernel_num,
        )
        return losses.compute_mmd_loss(kernels, batch_size)


class JANtrainer4Video(BaseMMDLike4Video):
    """This is an implementation of JAN for video data.
    """

    def __init__(self,
                 dataset,
                 image_modality,
                 feature_extractor,
                 task_classifier,
                 kernel_mul=(2.0, 2.0),
                 kernel_num=(5, 1),
                 **base_params,
                 ):
        super().__init__(
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            **base_params,
        )

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):
        softmax_layer = torch.nn.Softmax(dim=-1)
        source_list = [phi_s, softmax_layer(y_hat)]
        target_list = [phi_t, softmax_layer(y_t_hat)]
        batch_size = int(phi_s.size()[0])

        joint_kernels = None
        for source, target, k_mul, k_num, sigma in zip(
                source_list, target_list, self._kernel_mul, self._kernel_num, [None, 1.68]
        ):
            kernels = losses.gaussian_kernel(
                source, target, kernel_mul=k_mul, kernel_num=k_num, fix_sigma=sigma
            )
            if joint_kernels is not None:
                joint_kernels = joint_kernels * kernels
            else:
                joint_kernels = kernels

        return losses.compute_mmd_loss(joint_kernels, batch_size)


class DANNtrainer4Video(DANNtrainer):
    """This is an implementation of DANN for video data.
    """

    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            critic,
            method,
            **base_params,
    ):
        super(DANNtrainer4Video, self).__init__(
            dataset, feature_extractor, task_classifier, critic, method, **base_params
        )
        self.image_modality = image_modality
        self.rgb_feat = self.feat['rgb']
        self.flow_feat = self.feat['flow']

        self.domain_classifier = torch.nn.ModuleList(self.domain_classifier)

    def forward(self, x):
        # if self.feat is not None:
        #     if self.image_modality in ['rgb', 'flow']:
        #         if self.rgb_feat is not None:
        #             x = self.rgb_feat(x)
        #         else:
        #             x = self.flow_feat(x)
        #         x = x.view(x.size(0), -1)
        #         class_output = self.classifier(x)
        #
        #         reverse_feature = ReverseLayerF.apply(x, self.alpha)
        #
        #         adversarial_output = self.domain_classifier(reverse_feature)
        #         # print(x)
        #         # print(class_output)
        #         # print(adversarial_output)
        #         return x, class_output, adversarial_output

        # if self.feat is not None:
        #     if self.image_modality in ['rgb', 'flow']:
        #         if self.rgb_feat is not None:
        #             x, x_4b, x_4c, x_4d, x_4e, x_4f = self.rgb_feat(x)
        #         else:
        #             x = self.flow_feat(x)
        #         x = x.view(x.size(0), -1)
        #         x_4b = x_4b.view(x_4b.size(0), -1)
        #         x_4c = x_4c.view(x_4c.size(0), -1)
        #         x_4d = x_4d.view(x_4d.size(0), -1)
        #         x_4e = x_4e.view(x_4e.size(0), -1)
        #         x_4f = x_4f.view(x_4f.size(0), -1)
        #         class_output = self.classifier(x)
        #
        #         reverse_feature = ReverseLayerF.apply(x, self.alpha)
        #         rf_4b = ReverseLayerF.apply(x_4b, self.alpha)
        #         rf_4c = ReverseLayerF.apply(x_4c, self.alpha)
        #         rf_4d = ReverseLayerF.apply(x_4d, self.alpha)
        #         rf_4e = ReverseLayerF.apply(x_4e, self.alpha)
        #         rf_4f = ReverseLayerF.apply(x_4f, self.alpha)
        #
        #         adversarial_output = self.domain_classifier[0](reverse_feature)
        #         ao_4b = self.domain_classifier[1](rf_4b)
        #         ao_4c = self.domain_classifier[2](rf_4c)
        #         ao_4d = self.domain_classifier[3](rf_4d)
        #         ao_4e = self.domain_classifier[4](rf_4e)
        #         ao_4f = self.domain_classifier[5](rf_4f)
        #
        #         return x, class_output, adversarial_output, ao_4b, ao_4c, ao_4d, ao_4e, ao_4f

        if self.feat is not None:
            if self.image_modality in ['rgb', 'flow']:
                if self.rgb_feat is not None:
                    x, c4b, c4c, c4d, c4e, c4f, t4b, t4c, t4d, t4e, t4f = self.rgb_feat(x)
                else:
                    x = self.flow_feat(x)
                x = x.view(x.size(0), -1)
                c4b = c4b.view(c4b.size(0), -1)
                c4c = c4c.view(c4c.size(0), -1)
                c4d = c4d.view(c4d.size(0), -1)
                c4e = c4e.view(c4e.size(0), -1)
                c4f = c4f.view(c4f.size(0), -1)

                t4b = t4b.view(t4b.size(0), -1)
                t4c = t4c.view(t4c.size(0), -1)
                t4d = t4d.view(t4d.size(0), -1)
                t4e = t4e.view(t4e.size(0), -1)
                t4f = t4f.view(t4f.size(0), -1)

                class_output = self.classifier(x)

                reverse_feature = ReverseLayerF.apply(x, self.alpha)
                c4b = ReverseLayerF.apply(c4b, self.alpha)
                c4c = ReverseLayerF.apply(c4c, self.alpha)
                c4d = ReverseLayerF.apply(c4d, self.alpha)
                c4e = ReverseLayerF.apply(c4e, self.alpha)
                c4f = ReverseLayerF.apply(c4f, self.alpha)

                t4b = ReverseLayerF.apply(t4b, self.alpha)
                t4c = ReverseLayerF.apply(t4c, self.alpha)
                t4d = ReverseLayerF.apply(t4d, self.alpha)
                t4e = ReverseLayerF.apply(t4e, self.alpha)
                t4f = ReverseLayerF.apply(t4f, self.alpha)

                adversarial_output = self.domain_classifier[0](reverse_feature)
                c4b = self.domain_classifier[1](c4b)
                c4c = self.domain_classifier[2](c4c)
                c4d = self.domain_classifier[3](c4d)
                c4e = self.domain_classifier[4](c4e)
                c4f = self.domain_classifier[5](c4f)

                t4b = self.domain_classifier[6](t4b)
                t4c = self.domain_classifier[7](t4c)
                t4d = self.domain_classifier[8](t4d)
                t4e = self.domain_classifier[9](t4e)
                t4f = self.domain_classifier[10](t4f)

                # c_loss = (c4b + c4c + c4d + c4e + c4f) / 5
                # t_loss = (t4b + t4c + t4d + t4e + t4f) / 5
                # fe_loss = (c_loss + t_loss) / 2

                return x, class_output, adversarial_output, c4b, c4c, c4d, c4e, c4f, t4b, t4c, t4d, t4e, t4f

            elif self.image_modality == 'joint':
                x_rgb = self.rgb_feat(x['rgb'])
                x_flow = self.flow_feat(x['flow'])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
                x_flow = x_flow.view(x_flow.size(0), -1)
                x = torch.cat((x_rgb, x_flow), dim=1)

                class_output = self.classifier(x)

                reverse_feature_rgb = ReverseLayerF.apply(x_rgb, self.alpha)
                reverse_feature_flow = ReverseLayerF.apply(x_flow, self.alpha)

                adversarial_output_rgb = self.domain_classifier(reverse_feature_rgb)
                adversarial_output_flow = self.domain_classifier(reverse_feature_flow)
                return [x_rgb, x_flow], class_output, [adversarial_output_rgb, adversarial_output_flow]

    def compute_loss(self, batch, split_name="V"):
        if self.image_modality == 'joint' and len(batch) == 4:
            (x_s_rgb, y_s), (x_s_flow, y_s_flow), (x_tu_rgb, y_tu), (x_tu_flow, y_tu_flow) = batch
            _, y_hat, [d_hat_rgb, d_hat_flow] = self.forward({'rgb': x_s_rgb, 'flow': x_s_flow})
            _, y_t_hat, [d_t_hat_rgb, d_t_hat_flow] = self.forward({'rgb': x_tu_rgb, 'flow': x_tu_flow})
            batch_size = len(y_s)
            loss_dmn_src_rgb, dok_src_rgb = losses.cross_entropy_logits(d_hat_rgb, torch.zeros(batch_size))
            loss_dmn_src_flow, dok_src_flow = losses.cross_entropy_logits(d_hat_flow, torch.zeros(batch_size))
            loss_dmn_tgt_rgb, dok_tgt_rgb = losses.cross_entropy_logits(d_t_hat_rgb, torch.ones(batch_size))
            loss_dmn_tgt_flow, dok_tgt_flow = losses.cross_entropy_logits(d_t_hat_flow, torch.ones(batch_size))
            loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow
            loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow

            loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
            _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
            adv_loss = loss_dmn_src + loss_dmn_tgt  # adv_loss = src + tgt
            task_loss = loss_cls

            log_metrics = {
                f"{split_name}_source_acc": ok_src,
                f"{split_name}_target_acc": ok_tgt,
                f"{split_name}_domain_acc": torch.cat((dok_src_rgb, dok_src_flow, dok_tgt_rgb, dok_tgt_flow)),
                f"{split_name}_source_domain_acc": torch.cat((dok_src_rgb, dok_src_flow)),
                f"{split_name}_target_domain_acc": torch.cat((dok_tgt_rgb, dok_tgt_flow)),
            }
        # elif self.image_modality in ['rgb', 'flow'] and len(batch) == 2:
        #     (x_s, y_s), (x_tu, y_tu) = batch
        #     _, y_hat, d_hat = self.forward(x_s)
        #     _, y_t_hat, d_t_hat = self.forward(x_tu)
        #     batch_size = len(y_s)
        #     loss_dmn_src, dok_src = losses.cross_entropy_logits(d_hat, torch.zeros(batch_size))
        #     loss_dmn_tgt, dok_tgt = losses.cross_entropy_logits(d_t_hat, torch.ones(batch_size))
        #
        #     loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        #     _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)
        #     adv_loss = loss_dmn_src + loss_dmn_tgt  # adv_loss = src + tgt
        #     task_loss = loss_cls
        #
        #     log_metrics = {
        #         f"{split_name}_source_acc": ok_src,
        #         f"{split_name}_target_acc": ok_tgt,
        #         f"{split_name}_domain_acc": torch.cat((dok_src, dok_tgt)),
        #         f"{split_name}_source_domain_acc": dok_src,
        #         f"{split_name}_target_domain_acc": dok_tgt,
        #     }

        elif self.image_modality in ['rgb', 'flow'] and len(batch) == 2:

            (x_s, y_s), (x_tu, y_tu) = batch
            _, y_hat, d_hat, d_c4b_hat, d_c4c_hat, d_c4d_hat, d_c4e_hat, d_c4f_hat, d_t4b_hat, d_t4c_hat, d_t4d_hat, d_t4e_hat, d_t4f_hat = self.forward(x_s)
            _, y_t_hat, d_t_hat, d_t_c4b_hat, d_t_c4c_hat, d_t_c4d_hat, d_t_c4e_hat, d_t_c4f_hat, d_t_t4b_hat, d_t_t4c_hat, d_t_t4d_hat, d_t_t4e_hat, d_t_t4f_hat = self.forward(x_tu)

            batch_size = len(y_s)

            loss_dmn_src, dok_src = losses.cross_entropy_logits(d_hat, torch.zeros(batch_size))
            loss_dmn_tgt, dok_tgt = losses.cross_entropy_logits(d_t_hat, torch.ones(batch_size))

            lds_c4b, doks_c4b = losses.cross_entropy_logits(d_c4b_hat, torch.zeros(batch_size))
            ldt_c4b, dokt_c4b = losses.cross_entropy_logits(d_t_c4b_hat, torch.ones(batch_size))

            lds_c4c, doks_c4c = losses.cross_entropy_logits(d_c4c_hat, torch.zeros(batch_size))
            ldt_c4c, dokt_c4c = losses.cross_entropy_logits(d_t_c4c_hat, torch.ones(batch_size))

            lds_c4d, doks_c4d = losses.cross_entropy_logits(d_c4d_hat, torch.zeros(batch_size))
            ldt_c4d, dokt_c4d = losses.cross_entropy_logits(d_t_c4d_hat, torch.ones(batch_size))

            lds_c4e, doks_c4e = losses.cross_entropy_logits(d_c4e_hat, torch.zeros(batch_size))
            ldt_c4e, dokt_c4e = losses.cross_entropy_logits(d_t_c4e_hat, torch.ones(batch_size))

            lds_c4f, doks_c4f = losses.cross_entropy_logits(d_c4f_hat, torch.zeros(batch_size))
            ldt_c4f, dokt_c4f = losses.cross_entropy_logits(d_t_c4f_hat, torch.ones(batch_size))

            lds_t4b, doks_t4b = losses.cross_entropy_logits(d_t4b_hat, torch.zeros(batch_size))
            ldt_t4b, dokt_t4b = losses.cross_entropy_logits(d_t_t4b_hat, torch.ones(batch_size))

            lds_t4c, doks_t4c = losses.cross_entropy_logits(d_t4c_hat, torch.zeros(batch_size))
            ldt_t4c, dokt_t4c = losses.cross_entropy_logits(d_t_t4c_hat, torch.ones(batch_size))

            lds_t4d, doks_t4d = losses.cross_entropy_logits(d_t4d_hat, torch.zeros(batch_size))
            ldt_t4d, dokt_t4d = losses.cross_entropy_logits(d_t_t4d_hat, torch.ones(batch_size))

            lds_t4e, doks_t4e = losses.cross_entropy_logits(d_t4e_hat, torch.zeros(batch_size))
            ldt_t4e, dokt_t4e = losses.cross_entropy_logits(d_t_t4e_hat, torch.ones(batch_size))

            lds_t4f, doks_t4f = losses.cross_entropy_logits(d_t4f_hat, torch.zeros(batch_size))
            ldt_t4f, dokt_t4f = losses.cross_entropy_logits(d_t_t4f_hat, torch.ones(batch_size))

            loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)

            _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)

            adv_loss = loss_dmn_src + loss_dmn_tgt  # adv_loss = src + tgt

            avl_c4b = lds_c4b + ldt_c4b
            avl_c4c = lds_c4c + ldt_c4c
            avl_c4d = lds_c4d + ldt_c4d
            avl_c4e = lds_c4e + ldt_c4e
            avl_c4f = lds_c4f + ldt_c4f
            avl_t4b = lds_t4b + ldt_t4b
            avl_t4c = lds_t4c + ldt_t4c
            avl_t4d = lds_t4d + ldt_t4d
            avl_t4e = lds_t4e + ldt_t4e
            avl_t4f = lds_t4f + ldt_t4f

            task_loss = loss_cls

            log_metrics = {
                f"{split_name}_source_acc": ok_src,
                f"{split_name}_target_acc": ok_tgt,
                f"{split_name}_domain_acc": torch.cat((dok_src, dok_tgt)),
                f"{split_name}_source_domain_acc": dok_src,
                f"{split_name}_target_domain_acc": dok_tgt,
                f"{split_name}_c4b_domain_acc": torch.cat((doks_c4b, dokt_c4b)),
                f"{split_name}_c4c_domain_acc": torch.cat((doks_c4c, dokt_c4c)),
                f"{split_name}_c4d_domain_acc": torch.cat((doks_c4d, dokt_c4d)),
                f"{split_name}_c4e_domain_acc": torch.cat((doks_c4e, dokt_c4e)),
                f"{split_name}_c4f_domain_acc": torch.cat((doks_c4f, dokt_c4f)),

                f"{split_name}_t4b_domain_acc": torch.cat((doks_t4b, dokt_t4b)),
                f"{split_name}_t4c_domain_acc": torch.cat((doks_t4c, dokt_t4c)),
                f"{split_name}_t4d_domain_acc": torch.cat((doks_t4d, dokt_t4d)),
                f"{split_name}_t4e_domain_acc": torch.cat((doks_t4e, dokt_t4e)),
                f"{split_name}_t4f_domain_acc": torch.cat((doks_t4f, dokt_t4f)),
            }

        else:
            raise NotImplementedError("Batch len is {}. Check the Dataloader.".format(len(batch)))

        return task_loss, adv_loss, log_metrics, avl_c4b, avl_c4c, avl_c4d, avl_c4e, avl_c4f, avl_t4b, avl_t4c, avl_t4d, avl_t4e, avl_t4f

        # learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self._init_lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def training_step(self, batch, batch_nb):
        self._update_batch_epoch_factors(batch_nb)

        task_loss, adv_loss, log_metrics, avl_c4b, avl_c4c, avl_c4d, avl_c4e, avl_c4f, avl_t4b, avl_t4c, avl_t4d, avl_t4e, avl_t4f = self.compute_loss(
            batch, split_name="T")
        c_loss = (avl_c4b + avl_c4c + avl_c4d + avl_c4e + avl_c4f) / 5
        t_loss = (avl_t4b + avl_t4c + avl_t4d + avl_t4e + avl_t4f) / 5
        # fe_loss = (c_loss + t_loss) / 2
        fe_loss = c_loss
        print("c loss")
        if self.current_epoch < self._init_epochs:
            loss = task_loss
        else:
            loss = task_loss + self.lamb_da * (adv_loss + fe_loss)

        # loss = task_loss

        log_metrics = get_aggregated_metrics_from_dict(log_metrics)
        log_metrics.update(get_metrics_from_parameter_dict(self.get_parameters_watch_list(), loss.device))
        log_metrics["T_total_loss"] = loss
        log_metrics["T_adv_loss"] = adv_loss
        log_metrics["T_fe_loss"] = fe_loss
        log_metrics["T_task_loss"] = task_loss

        for key in log_metrics:
            self.log(key, log_metrics[key])

        return {
            "loss": loss,  # required, for backward pass
        }

    def validation_step(self, batch, batch_nb):
        task_loss, adv_loss, log_metrics, avl_c4b, avl_c4c, avl_c4d, avl_c4e, avl_c4f, avl_t4b, avl_t4c, avl_t4d, avl_t4e, avl_t4f = self.compute_loss(
            batch, split_name="V")
        c_loss = (avl_c4b + avl_c4c + avl_c4d + avl_c4e + avl_c4f) / 5
        t_loss = (avl_t4b + avl_t4c + avl_t4d + avl_t4e + avl_t4f) / 5
        fe_loss = (c_loss + t_loss) / 2
        loss = task_loss + self.lamb_da * adv_loss + fe_loss

        # task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="V")
        # loss = task_loss + self.lamb_da * adv_loss
        log_metrics["val_loss"] = loss
        log_metrics["val_task_loss"] = task_loss
        log_metrics["val_adv_loss"] = adv_loss
        log_metrics["val_fe_loss"] = fe_loss
        return log_metrics

    def test_step(self, batch, batch_nb):
        # task_loss, adv_loss, log_metrics = self.compute_loss(batch, split_name="Te")
        # loss = task_loss + self.lamb_da * adv_loss
        task_loss, adv_loss, log_metrics, avl_c4b, avl_c4c, avl_c4d, avl_c4e, avl_c4f, avl_t4b, avl_t4c, avl_t4d, avl_t4e, avl_t4f = self.compute_loss(
            batch, split_name="Te")
        c_loss = (avl_c4b + avl_c4c + avl_c4d + avl_c4e + avl_c4f) / 5
        t_loss = (avl_t4b + avl_t4c + avl_t4d + avl_t4e + avl_t4f) / 5
        fe_loss = (c_loss + t_loss) / 2
        loss = task_loss + self.lamb_da * adv_loss + fe_loss

        log_metrics["test_loss"] = loss
        return log_metrics




class CDANtrainer4Video(CDANtrainer):
    """This is an implementation of CDAN for video data.
    """

    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            critic,
            use_entropy=False,
            use_random=False,
            random_dim=1024,
            **base_params,
    ):
        super(CDANtrainer4Video, self).__init__(
            dataset,
            feature_extractor,
            task_classifier,
            critic,
            use_entropy,
            use_random,
            random_dim,
            **base_params
        )
        self.image_modality = image_modality
        self.rgb_feat = self.feat['rgb']
        self.flow_feat = self.feat['flow']

    def forward(self, x):
        if self.feat is not None:
            if self.image_modality in ['rgb', 'flow']:
                if self.rgb_feat is not None:
                    x = self.rgb_feat(x)
                else:
                    x = self.flow_feat(x)
                x = x.view(x.size(0), -1)
                class_output = self.classifier(x)

                # The GRL hook is applied to all inputs to the adversary
                reverse_feature = ReverseLayerF.apply(x, self.alpha)

                softmax_output = torch.nn.Softmax(dim=1)(class_output)
                reverse_out = ReverseLayerF.apply(softmax_output, self.alpha)

                feature = torch.bmm(reverse_out.unsqueeze(2), reverse_feature.unsqueeze(1))
                feature = feature.view(-1, reverse_out.size(1) * reverse_feature.size(1))
                if self.random_layer:
                    random_out = self.random_layer.forward(feature)
                    adversarial_output = self.domain_classifier(
                        random_out.view(-1, random_out.size(1))
                    )
                else:
                    adversarial_output = self.domain_classifier(feature)

                return x, class_output, adversarial_output

            elif self.image_modality == 'joint':
                x_rgb = self.rgb_feat(x['rgb'])
                x_flow = self.flow_feat(x['flow'])
                x_rgb = x_rgb.view(x_rgb.size(0), -1)
                x_flow = x_flow.view(x_flow.size(0), -1)
                x = torch.cat((x_rgb, x_flow), dim=1)

                class_output = self.classifier(x)
                softmax_output = torch.nn.Softmax(dim=1)(class_output)
                reverse_out = ReverseLayerF.apply(softmax_output, self.alpha)

                reverse_feature_rgb = ReverseLayerF.apply(x_rgb, self.alpha)
                reverse_feature_flow = ReverseLayerF.apply(x_flow, self.alpha)

                feature_rgb = torch.bmm(reverse_out.unsqueeze(2), reverse_feature_rgb.unsqueeze(1))
                feature_rgb = feature_rgb.view(-1, reverse_out.size(1) * reverse_feature_rgb.size(1))
                feature_flow = torch.bmm(reverse_out.unsqueeze(2), reverse_feature_flow.unsqueeze(1))
                feature_flow = feature_flow.view(-1, reverse_out.size(1) * reverse_feature_flow.size(1))

                if self.random_layer:
                    random_out_rgb = self.random_layer.forward(feature_rgb)
                    random_out_flow = self.random_layer.forward(feature_flow)
                    adversarial_output_rgb = self.domain_classifier(
                        random_out_rgb.view(-1, random_out_rgb.size(1))
                    )
                    adversarial_output_flow = self.domain_classifier(
                        random_out_flow.view(-1, random_out_flow.size(1))
                    )
                else:
                    adversarial_output_rgb = self.domain_classifier(feature_rgb)
                    adversarial_output_flow = self.domain_classifier(feature_flow)
                return [x_rgb, x_flow], class_output, [adversarial_output_rgb, adversarial_output_flow]

    def compute_loss(self, batch, split_name="V"):
        if self.image_modality == 'joint' and len(batch) == 4:
            (x_s_rgb, y_s), (x_s_flow, y_s_flow), (x_tu_rgb, y_tu), (x_tu_flow, y_tu_flow) = batch
            _, y_hat, [d_hat_rgb, d_hat_flow] = self.forward({'rgb': x_s_rgb, 'flow': x_s_flow})
            _, y_t_hat, [d_t_hat_rgb, d_t_hat_flow] = self.forward({'rgb': x_tu_rgb, 'flow': x_tu_flow})
            batch_size = len(y_s)

            if self.entropy:
                e_s = self._compute_entropy_weights(y_hat)
                e_t = self._compute_entropy_weights(y_t_hat)
                source_weight = e_s / torch.sum(e_s)
                target_weight = e_t / torch.sum(e_t)
            else:
                source_weight = None
                target_weight = None

            loss_dmn_src_rgb, dok_src_rgb = losses.cross_entropy_logits(
                d_hat_rgb, torch.zeros(batch_size), source_weight)
            loss_dmn_src_flow, dok_src_flow = losses.cross_entropy_logits(
                d_hat_flow, torch.zeros(batch_size), source_weight)
            loss_dmn_tgt_rgb, dok_tgt_rgb = losses.cross_entropy_logits(
                d_t_hat_rgb, torch.ones(len(d_t_hat_rgb)), target_weight)
            loss_dmn_tgt_flow, dok_tgt_flow = losses.cross_entropy_logits(
                d_t_hat_flow, torch.ones(len(d_t_hat_flow)), target_weight)
            loss_dmn_src = loss_dmn_src_rgb + loss_dmn_src_flow
            loss_dmn_tgt = loss_dmn_tgt_rgb + loss_dmn_tgt_flow
            # Sum rgb and flow results(True/False) to get the domain accuracy result.
            dok_src = dok_src_rgb + dok_src_flow
            dok_tgt = dok_tgt_rgb + dok_tgt_flow

        elif self.image_modality in ['rgb', 'flow'] and len(batch) == 2:
            (x_s, y_s), (x_tu, y_tu) = batch
            _, y_hat, d_hat = self.forward(x_s)
            _, y_t_hat, d_t_hat = self.forward(x_tu)
            batch_size = len(y_s)

            if self.entropy:
                e_s = self._compute_entropy_weights(y_hat)
                e_t = self._compute_entropy_weights(y_t_hat)
                source_weight = e_s / torch.sum(e_s)
                target_weight = e_t / torch.sum(e_t)
            else:
                source_weight = None
                target_weight = None

            loss_dmn_src, dok_src = losses.cross_entropy_logits(
                d_hat, torch.zeros(batch_size), source_weight
            )
            loss_dmn_tgt, dok_tgt = losses.cross_entropy_logits(
                d_t_hat, torch.ones(len(d_t_hat)), target_weight
            )

        else:
            raise NotImplementedError("Batch len is {}. Check the Dataloader.".format(len(batch)))

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)

        adv_loss = loss_dmn_src + loss_dmn_tgt
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": torch.cat((dok_src, dok_tgt)),
            f"{split_name}_source_domain_acc": dok_src,
            f"{split_name}_target_domain_acc": dok_tgt,
        }
        return task_loss, adv_loss, log_metrics


class WDGRLtrainer4Video(WDGRLtrainer):
    """This is an implementation of WDGRL for video data.
    """

    def __init__(
            self,
            dataset,
            image_modality,
            feature_extractor,
            task_classifier,
            critic,
            k_critic=5,
            gamma=10,
            beta_ratio=0,
            **base_params,
    ):
        super(WDGRLtrainer4Video, self).__init__(
            dataset,
            feature_extractor,
            task_classifier,
            critic,
            k_critic,
            gamma,
            beta_ratio,
            **base_params
        )
        self.image_modality = image_modality
        self.rgb_feat = self.feat['rgb']
        self.flow_feat = self.feat['flow']

    def forward(self, x):
        if self.feat is not None:
            if self.image_modality in ['rgb', 'flow']:
                if self.rgb_feat is not None:
                    x = self.rgb_feat(x)
                else:
                    x = self.flow_feat(x)
            elif self.image_modality == 'joint':
                x_rgb = self.rgb_feat(x['rgb'])
                x_flow = self.flow_feat(x['flow'])
                x = torch.cat((x_rgb, x_flow), dim=1)
        x = x.view(x.size(0), -1)

        class_output = self.classifier(x)
        adversarial_output = self.domain_classifier(x)
        return x, class_output, adversarial_output

    def compute_loss(self, batch, split_name="V"):
        if len(batch) == 4:
            (x_s_rgb, y_s), (x_s_flow, y_s_flow), (x_tu_rgb, y_tu), (x_tu_flow, y_tu_flow) = batch
            _, y_hat, d_hat = self.forward({'rgb': x_s_rgb, 'flow': x_s_flow})
            _, y_t_hat, d_t_hat = self.forward({'rgb': x_tu_rgb, 'flow': x_tu_flow})
        elif len(batch) == 2:
            (x_s, y_s), (x_tu, y_tu) = batch
            _, y_hat, d_hat = self.forward(x_s)
            _, y_t_hat, d_t_hat = self.forward(x_tu)
        else:
            raise NotImplementedError("Batch len is {}. Check the Dataloader.".format(len(batch)))
        batch_size = len(y_s)

        loss_cls, ok_src = losses.cross_entropy_logits(y_hat, y_s)
        _, ok_tgt = losses.cross_entropy_logits(y_t_hat, y_tu)

        _, dok_src = losses.cross_entropy_logits(d_hat, torch.zeros(batch_size))
        _, dok_tgt = losses.cross_entropy_logits(d_t_hat, torch.ones(len(d_t_hat)))

        wasserstein_distance = d_hat.mean() - (1 + self._beta_ratio) * d_t_hat.mean()
        adv_loss = wasserstein_distance
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": torch.cat((dok_src, dok_tgt)),
            f"{split_name}_source_domain_acc": dok_src,
            f"{split_name}_target_domain_acc": dok_tgt,
            f"{split_name}_wasserstein_dist": wasserstein_distance,
        }
        return task_loss, adv_loss, log_metrics

    def configure_optimizers(self):
        if self.image_modality in ['rgb', 'flow']:
            if self.rgb_feat is not None:
                nets = [self.rgb_feat, self.classifier]
            else:
                nets = [self.flow_feat, self.classifier]
        elif self.image_modality == 'joint':
            nets = [self.rgb_feat, self.flow_feat, self.classifier]
        parameters = set()

        for net in nets:
            parameters |= set(net.parameters())

        if self._adapt_lr:
            task_feat_optimizer, task_feat_sched = self._configure_optimizer(parameters)
            self.critic_opt, self.critic_sched = self._configure_optimizer(
                self.domain_classifier.parameters()
            )
            self.critic_opt = self.critic_opt[0]
            self.critic_sched = self.critic_sched[0]
            return task_feat_optimizer, task_feat_sched
        else:
            task_feat_optimizer = self._configure_optimizer(parameters)
            self.critic_opt = self._configure_optimizer(
                self.domain_classifier.parameters()
            )
            self.critic_sched = None
            self.critic_opt = self.critic_opt[0]
        return task_feat_optimizer

    def critic_update_steps(self, batch):
        # if self.current_epoch < self._init_epochs:
        #     return

        set_requires_grad(self.domain_classifier, requires_grad=True)

        if self.image_modality in ['rgb', 'flow']:
            if self.rgb_feat is not None:
                set_requires_grad(self.rgb_feat, requires_grad=False)
                (x_s, y_s), (x_tu, _) = batch
                with torch.no_grad():
                    h_s = self.rgb_feat(x_s).data.view(x_s.shape[0], -1)
                    h_t = self.rgb_feat(x_tu).data.view(x_tu.shape[0], -1)
            else:
                set_requires_grad(self.flow_feat, requires_grad=False)
                (x_s, y_s), (x_tu, _) = batch
                with torch.no_grad():
                    h_s = self.flow_feat(x_s).data.view(x_s.shape[0], -1)
                    h_t = self.flow_feat(x_tu).data.view(x_tu.shape[0], -1)

            for _ in range(self._k_critic):
                gp = losses.gradient_penalty(self.domain_classifier, h_s, h_t)

                critic_s = self.domain_classifier(h_s)
                critic_t = self.domain_classifier(h_t)
                wasserstein_distance = (critic_s.mean() - (1 + self._beta_ratio) * critic_t.mean())

                critic_cost = -wasserstein_distance + self._gamma * gp

                self.critic_opt.zero_grad()
                critic_cost.backward()
                self.critic_opt.step()
                if self.critic_sched:
                    self.critic_sched.step()

            if self.rgb_feat is not None:
                set_requires_grad(self.rgb_feat, requires_grad=True)
            else:
                set_requires_grad(self.flow_feat, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=False)

        elif self.image_modality == 'joint':
            set_requires_grad(self.rgb_feat, requires_grad=False)
            set_requires_grad(self.flow_feat, requires_grad=False)
            (x_s_rgb, y_s), (x_s_flow, _), (x_tu_rgb, _), (x_tu_flow, _) = batch
            with torch.no_grad():
                h_s_rgb = self.rgb_feat(x_s_rgb).data.view(x_s_rgb.shape[0], -1)
                h_t_rgb = self.rgb_feat(x_tu_rgb).data.view(x_tu_rgb.shape[0], -1)
                h_s_flow = self.flow_feat(x_s_flow).data.view(x_s_flow.shape[0], -1)
                h_t_flow = self.flow_feat(x_tu_flow).data.view(x_tu_flow.shape[0], -1)
                h_s = torch.cat((h_s_rgb, h_s_flow), dim=1)
                h_t = torch.cat((h_t_rgb, h_t_flow), dim=1)

            for _ in range(self._k_critic):
                gp = losses.gradient_penalty(self.domain_classifier, h_s, h_t)

                critic_s = self.domain_classifier(h_s)
                critic_t = self.domain_classifier(h_t)
                wasserstein_distance = (critic_s.mean() - (1 + self._beta_ratio) * critic_t.mean())

                critic_cost = -wasserstein_distance + self._gamma * gp

                self.critic_opt.zero_grad()
                critic_cost.backward()
                self.critic_opt.step()
                if self.critic_sched:
                    self.critic_sched.step()

                # Uncomment for later work. Process rgb and flow dividedly.
                # for _ in range(self._k_critic):
                #     gp_rgb = losses.gradient_penalty(self.domain_classifier, h_s_rgb, h_t_rgb)
                #     gp_flow = losses.gradient_penalty(self.domain_classifier, h_s_flow, h_t_flow)
                #
                #     critic_s_rgb = self.domain_classifier(h_s_rgb)
                #     critic_s_flow = self.domain_classifier(h_s_flow)
                #     critic_t_rgb = self.domain_classifier(h_t_rgb)
                #     critic_t_flow = self.domain_classifier(h_t_flow)
                #     wasserstein_distance_rgb = (
                #             critic_s_rgb.mean() - (1 + self._beta_ratio) * critic_t_rgb.mean()
                #     )
                #     wasserstein_distance_flow = (
                #             critic_s_flow.mean() - (1 + self._beta_ratio) * critic_t_flow.mean()
                #     )
                #
                #     critic_cost = (-wasserstein_distance_rgb + -wasserstein_distance_flow +
                #                    self._gamma * gp_rgb + self._gamma * gp_flow) * 0.5

                self.critic_opt.zero_grad()
                critic_cost.backward()
                self.critic_opt.step()
                if self.critic_sched:
                    self.critic_sched.step()

            set_requires_grad(self.rgb_feat, requires_grad=True)
            set_requires_grad(self.flow_feat, requires_grad=True)
            set_requires_grad(self.domain_classifier, requires_grad=False)
