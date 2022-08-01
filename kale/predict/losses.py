"""Commonly used losses, from domain adaptation package
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/losses.py
"""
# import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.nn import functional as F


def cross_entropy_logits(linear_output, label, weights=None):
    """Computes cross entropy with logits

    Examples:
        See DANN, WDGRL, and MMD trainers in kale.pipeline.domain_adapter
    """

    class_output = F.log_softmax(linear_output, dim=1)
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    correct = y_hat.eq(label.view(label.size(0)).type_as(y_hat))
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return loss, correct


def topk_accuracy(output, target, topk=(1,)):
    """Computes the top-k accuracy for the specified values of k.

    Args:
        output (Tensor): Generated predictions. Shape: (batch_size, class_count).
        target (Tensor): Ground truth. Shape: (batch_size)
        topk (tuple(int)): Compute accuracy at top-k for the values of k specified in this parameter.
    Returns:
        list(Tensor): A list of tensors of the same length as topk.
        Each tensor consists of boolean variables to show if this prediction ranks top k with each value of k.
        True means the prediction ranks top k and False means not.
        The shape of tensor is batch_size, i.e. the number of predictions.

    Examples:
        >>> output = torch.tensor(([0.3, 0.2, 0.1], [0.3, 0.2, 0.1]))
        >>> target = torch.tensor((0, 1))
        >>> top1, top2 = topk_accuracy(output, target, topk=(1, 2)) # get the boolean value
        >>> top1_value = top1.double().mean() # get the top 1 accuracy score
        >>> top2_value = top2.double().mean() # get the top 2 accuracy score
    """

    maxk = max(topk)

    # returns the k largest elements and their indexes of inputs along a given dimension.
    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = torch.ge(correct[:k].float().sum(0), 1)
        result.append(correct_k)
    return result


def multitask_topk_accuracy(output, target, topk=(1,)):
    """Computes the top-k accuracy for the specified values of k for multitask input.

    Args:
        output (tuple(Tensor)): A tuple of generated predictions. Each tensor is of shape [batch_size, class_count],
            class_count can vary per task basis, i.e. outputs[i].shape[1] can differ from outputs[j].shape[1].
        target (tuple(Tensor)): A tuple of ground truth. Each tensor is of shape [batch_size]
        topk (tuple(int)): Compute accuracy at top-k for the values of k specified in this parameter.
    Returns:
        list(Tensor): A list of tensors of the same length as topk.
        Each tensor consists of boolean variables to show if predictions of multitask ranks top k with each value of k.
        True means predictions of this output for all tasks ranks top k and False means not.
        The shape of tensor is batch_size, i.e. the number of predictions.

        Examples:
            >>> first_output = torch.tensor(([0.3, 0.2, 0.1], [0.3, 0.2, 0.1]))
            >>> first_target = torch.tensor((0, 2))
            >>> second_output = torch.tensor(([0.2, 0.1], [0.2, 0.1]))
            >>> second_target = torch.tensor((0, 1))
            >>> output = (first_output, second_output)
            >>> target = (first_target, second_target)
            >>> top1, top2 = multitask_topk_accuracy(output, target, topk=(1, 2)) # get the boolean value
            >>> top1_value = top1.double().mean() # get the top 1 accuracy score
            >>> top2_value = top2.double().mean() # get the top 2 accuracy score
    """

    maxk = max(topk)
    batch_size = target[0].size(0)
    task_count = len(output)
    all_correct = torch.zeros(maxk, batch_size).type(torch.ByteTensor).to(output[0].device)

    for output, target in zip(output, target):
        # returns the k largest elements and their indexes of inputs along a given dimension.
        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        all_correct.add_(correct)

    result = []
    for k in topk:
        all_correct_k = torch.ge(all_correct[:k].float().sum(0), task_count)
        result.append(all_correct_k)
    return result


def entropy_logits(linear_output):
    """Computes entropy logits in CDAN with entropy conditioning (CDAN+E)

    Examples:
        See CDANTrainer in kale.pipeline.domain_adapter
    """
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


def entropy_logits_loss(linear_output):
    """Computes entropy logits loss in semi-supervised or few-shot domain adaptation

    Examples:
        See FewShotDANNTrainer in kale.pipeline.domain_adapter
    """
    return torch.mean(entropy_logits(linear_output))


def gradient_penalty(critic, h_s, h_t):
    """Computes gradient penalty in Wasserstein distance guided representation learning

    Examples:
        See WDGRLTrainer and WDGRLTrainerMod in kale.pipeline.domain_adapter
    """

    alpha = torch.rand(h_s.size(0), 1)
    alpha = alpha.expand(h_s.size()).type_as(h_s)
    # try:
    differences = h_t - h_s

    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat((interpolates, h_s, h_t), dim=0).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates, grad_outputs=torch.ones_like(preds), retain_graph=True, create_graph=True,)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    # except:
    #     gradient_penalty = 0

    return gradient_penalty


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Code from XLearn: computes the full kernel matrix, which is less than optimal since we don't use all of it
    with the linear MMD estimate.

    Examples:
        See DANTrainer and JANTrainer in kale.pipeline.domain_adapter
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    l2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def compute_mmd_loss(kernel_values, batch_size):
    """Computes the Maximum Mean Discrepancy (MMD) between domains.

    Examples:
        See DANTrainer and JANTrainer in kale.pipeline.domain_adapter
    """
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernel_values[s1, s2] + kernel_values[t1, t2]
        loss -= kernel_values[s1, t2] + kernel_values[s2, t1]
    return loss / float(batch_size)


def attentive_entropy(pred, pred_domain):
    """Computes the Attentive entropy loss.

    Examples:
    See TA3NTrainer in kale.pipeline.domain_adapter
    """
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)

    # attention weight
    entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
    weights = 1 + entropy

    # attentive entropy
    loss = torch.mean(weights * torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss


def cross_entropy_soft(pred):
    """Computes the Cross entropy softmax loss.

    Examples:
    See TA3NTrainer in kale.pipeline.domain_adapter
    """
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    loss = torch.mean(torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss


def dis_MCD(out1, out2):
    """discrepancy loss used in MCD (CVPR 18).

    Examples:
    See TA3NTrainer in kale.pipeline.domain_adapter
    """
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))


def mmd_linear(f_of_X, f_of_Y):
    # Consider linear time MMD with a linear kernel:
    # K(f(x), f(y)) = f(x)^Tf(y)
    # h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
    #             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
    #
    # f_of_X: batch_size * k
    # f_of_Y: batch_size * k

    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError("ver == 1 or 2")

    return loss


def JAN(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[2, 5], fix_sigma_list=[None, None], ver=2):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
            loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = joint_kernels[:batch_size, :batch_size]
        YY = joint_kernels[batch_size:, batch_size:]
        XY = joint_kernels[:batch_size, batch_size:]
        YX = joint_kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError("ver == 1 or 2")

    return loss


def hsic(kx, ky, device):
    """
    Perform independent test with Hilbert-Schmidt Independence Criterion (HSIC) between two sets of variables x and y.

    Args:
        kx (2-D tensor): kernel matrix of x, shape (n_samples, n_samples)
        ky (2-D tensor): kernel matrix of y, shape (n_samples, n_samples)
        device (torch.device): the desired device of returned tensor

    Returns:
        [tensor]: Independent test score >= 0

    Reference:
        [1] Gretton, Arthur, Bousquet, Olivier, Smola, Alex, and Schölkopf, Bernhard. Measuring Statistical Dependence
            with Hilbert-Schmidt Norms. In Algorithmic Learning Theory (ALT), pp. 63–77. 2005.
        [2] Gretton, Arthur, Fukumizu, Kenji, Teo, Choon H., Song, Le, Schölkopf, Bernhard, and Smola, Alex J. A Kernel
            Statistical Test of Independence. In Advances in Neural Information Processing Systems, pp. 585–592. 2008.
    """

    n = kx.shape[0]
    if ky.shape[0] != n:
        raise ValueError("kx and ky are expected to have the same sample sizes.")
    ctr_mat = torch.eye(n, device=device) - torch.ones((n, n), device=device) / n
    return torch.trace(torch.mm(torch.mm(torch.mm(kx, ctr_mat), ky), ctr_mat)) / (n ** 2)


def euclidean(x1, x2):
    """Compute the Euclidean distance

    Args:
        x1 (torch.Tensor): variables set 1
        x2 (torch.Tensor): variables set 2

    Returns:
        torch.Tensor: Euclidean distance
    """
    return ((x1 - x2) ** 2).sum().sqrt()


def _moment_k(x: torch.Tensor, domain_labels: torch.Tensor, k_order=2):
    """Compute the k-th moment distance

    Args:
        x (torch.Tensor): input data, shape (n_samples, n_features)
        domain_labels (torch.Tensor): labels indicating which domain the instance is from, shape (n_samples,)
        k_order (int, optional): moment order. Defaults to 2.

    Returns:
        torch.Tensor: the k-th moment distance

    The code is based on:
        https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/engine/da/m3sda.py#L153
        https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/blob/master/M3SDA/code_MSDA_digit/metric/msda.py#L6
    """
    unique_domain_ = torch.unique(domain_labels)
    n_unique_domain_ = len(unique_domain_)
    x_k_order = []
    for domain_label_ in unique_domain_:
        domain_idx = torch.where(domain_labels == domain_label_)[0]
        x_mean = x[domain_idx].mean(0)
        if k_order == 1:
            x_k_order.append(x_mean)
        else:
            x_k_order.append(((x[domain_idx] - x_mean) ** k_order).mean(0))
    moment_sum = 0
    n_pair = 0
    for i in range(n_unique_domain_):
        for j in range(i + 1, n_unique_domain_):
            moment_sum += euclidean(x_k_order[i], x_k_order[j])
            n_pair += 1
    return moment_sum / n_pair
