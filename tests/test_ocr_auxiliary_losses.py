"""Guard retention-loss scope with synthetic logits, no images/checkpoints."""
import pytest
import torch
from tools.ocr_auxiliary_losses import masked_teacher_kl, freeze_batch_norm_stats


def test_auxiliary_forward_freezes_bn_stats_but_keeps_recurrent_backward():
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(3)
            self.rnn = torch.nn.LSTM(3,4,batch_first=True)

        def forward(self,x):
            return self.rnn(self.norm(x.transpose(1,2)).transpose(1,2))[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Network().to(device).train()
    mean = model.norm.running_mean.clone()
    variance = model.norm.running_var.clone()
    freeze_batch_norm_stats(model)
    model(torch.randn(2,5,3,device=device)).sum().backward()
    assert model.rnn.training and not model.norm.training
    assert torch.equal(model.norm.running_mean,mean)
    assert torch.equal(model.norm.running_var,variance)
    assert model.norm.weight.grad is not None
    assert model.rnn.weight_ih_l0.grad is not None
    model.train()
    assert model.norm.training


def test_padding_unselected_rows_and_teacher_have_no_gradient():
    student = torch.zeros(2, 3, 3, requires_grad=True)
    teacher = torch.tensor([[[4.,0.,0.]]*3, [[0.,4.,0.]]*3], requires_grad=True)
    loss = masked_teacher_kl(student, teacher, torch.tensor([1,3]), torch.tensor([True,False]))
    loss.backward()
    assert loss.item() > 0
    assert student.grad[0,0].abs().sum() > 0
    assert student.grad[0,1:].abs().sum() == 0
    assert student.grad[1].abs().sum() == 0
    assert teacher.grad is None


def test_invalid_padding_values_cannot_contaminate_selected_loss():
    student = torch.tensor([[[0.,0.],[float('nan'),float('inf')]]], requires_grad=True)
    teacher = torch.tensor([[[3.,0.],[float('nan'),float('inf')]]])
    loss = masked_teacher_kl(student, teacher, torch.tensor([1]), torch.tensor([True]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad[0,1].abs().sum() == 0


def test_empty_selection_returns_differentiable_zero():
    student = torch.full((2,3,4),float('nan'),requires_grad=True)
    loss = masked_teacher_kl(student, torch.randn_like(student), torch.tensor([2,3]), torch.tensor([False,False]))
    loss.backward()
    assert loss.item() == 0 and student.grad.abs().sum() == 0


def test_equal_logits_have_zero_loss_and_ignore_longer_padding():
    logits = torch.randn(2,4,3)
    assert abs(masked_teacher_kl(logits, logits, torch.tensor([2,4]), torch.tensor([True,True])).item()) < 1e-6


@pytest.mark.parametrize('temperature', [0.,-1.,float('nan'),float('inf')])
def test_bad_temperature_is_rejected(temperature):
    with pytest.raises(ValueError,match='temperature'):
        masked_teacher_kl(torch.zeros(1,2,3),torch.zeros(1,2,3),torch.tensor([2]),torch.tensor([True]),temperature=temperature)


def test_bad_dimensions_and_lengths_are_rejected():
    logits = torch.zeros(1,2,3)
    with pytest.raises(ValueError,match='matching'):
        masked_teacher_kl(logits,torch.zeros(1,1,3),torch.tensor([2]),torch.tensor([True]))
    with pytest.raises(ValueError,match='integer lengths'):
        masked_teacher_kl(logits,logits,torch.tensor([3]),torch.tensor([True]))
    with pytest.raises(ValueError,match='boolean selection'):
        masked_teacher_kl(logits,logits,torch.tensor([2]),torch.tensor([1]))
