import torch
import torch.nn.functional as F


def get_loss_fn(name):
    if name == "corrected":
        return _dco_corrected_paper_loss
    if name == "corrected-max":
        return _dco_corrected_paper_loss_with_max
    if name == "uncorrected":
        return _dco_un_corrected_paper_loss
    if name == "uncorrected-code":
        return _dco_un_corrected_code_loss
    if name == "var":
        return _dco_corrected_paper_loss_with_var
    if name == "var-max":
        return _dco_corrected_paper_loss_with_max_and_var
    raise NotImplementedError(name)


def _dco_corrected_paper_loss(f_s1, f_sp, f_s2, beta, delta):
    term1 = F.mse_loss(f_s1, f_sp)
    term2 = torch.mean((f_s1 - delta) * (f_s2 - delta))
    term3 = torch.mean((f_s1 ** 2) * (f_s2 ** 2))
    loss = term1 + (beta * (term2 + term3))
    return loss


def _dco_corrected_paper_loss_with_max(f_s1, f_sp, f_s2, beta, delta):
    term1 = F.mse_loss(f_s1, f_sp)
    term2 = torch.mean((f_s1 - delta) * (f_s2 - delta))
    term3 = torch.mean((f_s1 ** 2) * (f_s2 ** 2))
    term4 = torch.max(f_s1 - f_sp, 0)[0]
    loss = term1 + (beta * (term2 + term3 + term4))
    return loss


def _dco_corrected_paper_loss_with_var(f_s1, f_sp, f_s2, beta, delta):
    term1 = F.mse_loss(f_s1, f_sp)
    term2 = torch.mean((f_s1 - delta) * (f_s2 - delta))
    term3 = torch.mean((f_s1 ** 2) * (f_s2 ** 2))
    term4 = -torch.var(f_s1) - torch.var(f_s2)
    loss = term1 + (beta * (term2 + term3 + term4))
    return loss


def _dco_corrected_paper_loss_with_max_and_var(f_s1, f_sp, f_s2, beta, delta):
    term1 = F.mse_loss(f_s1, f_sp)
    term2 = torch.mean((f_s1 - delta) * (f_s2 - delta))
    term3 = torch.mean((f_s1 ** 2) * (f_s2 ** 2))
    term4 = torch.max(f_s1 - f_sp, 0)[0]
    term5 = -torch.var(f_s1) - torch.var(f_s2)
    loss = term1 + (beta * (term2 + term3 + term4 + term5))
    return loss


def _dco_un_corrected_paper_loss(f_s1, f_sp, f_s2, beta, delta):
    term1 = F.mse_loss(f_s1, f_sp)
    term2 = torch.mean((f_s1**2 - delta) * (f_s2**2 - delta))
    term3 = torch.mean((f_s1 ** 2) * (f_s2 ** 2))
    loss = term1 + (beta * (term2 + term3))
    return loss


def _dco_un_corrected_code_loss(f_s1, f_sp, f_s2, beta, delta):
    """
    self.loss = tflearn.mean_square(self.f_value, self.next_f_value) \
                + self.beta * tf.reduce_mean(tf.multiply(self.f_value - self.delta, self.next_f_value - self.delta)) \
                + self.beta * tf.reduce_mean(self.f_value * self.f_value * self.next_f_value * self.next_f_value) \
                + self.beta * tf.math.maximum((self.f_value - self.next_f_value),
                                              0.0)

    """
    beta = 0.1
    delta = 0.05
    term1 = F.mse_loss(f_s1, f_sp)
    term2 = torch.mean((f_s1 - delta) * (f_sp - delta))
    term3 = torch.mean((f_s1 ** 2) * (f_sp ** 2))
    term4 = torch.max(f_s1 - f_sp, 0)[0]
    loss = term1 + (beta * (term2 + term3 + term4))
    return loss


def _dco_unit_norm_loss(f_s1, f_sp, f_s2, beta, delta):
    """ Rather than maximizing delta*||f(s)||, maximize delta*||f(s) - 1||. """
    pass


def _dco_lagrangian_loss(f_s1, f_sp, f_s2, beta, delta):
    pass


def _dco_projected_gradient_loss(f_s1, f_sp, f_s2, beta, delta):
    pass

