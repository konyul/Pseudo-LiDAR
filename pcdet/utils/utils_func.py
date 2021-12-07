import torch


class ModularizedFunction(torch.nn.Module):
    """
    A Module which calls the specified function in place of the forward pass.
    Useful when your existing loss is functional and you need it to be a Module.
    """
    def __init__(self, forward_op):
        super().__init__()
        self.forward_op = forward_op

    def forward(self, *args, **kwargs):
        return self.forward_op(*args, **kwargs)


class CriterionParallel(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        if not isinstance(criterion, torch.nn.Module):
            criterion = ModularizedFunction(criterion)
        self.criterion = torch.nn.DataParallel(criterion)

    def forward(self, *args, **kwargs):
        """
        Note the .mean() here, which is required since DataParallel 
        gathers any scalar outputs of forward() into a vector with 
        one item per GPU (See DataParallel docs).
        """
        return self.criterion(*args, **kwargs).mean()
class Metric(object):
    def __init__(self):
        self.EPE = AverageMeter()
        self.RMSELIs = AverageMeter()
        self.RMSELGs = AverageMeter()
        self.ABSRs = AverageMeter()
        self.SQRs = AverageMeter()
        self.DELTA = AverageMeter()
        self.DELTASQ = AverageMeter()
        self.DELTACU = AverageMeter()
        self.losses = AverageMeter()

    def update(self, loss, EPE, RMSE_Linear, RMSE_Log, abs_relative, sq_relative, delta, delta_sq, delta_cu):
        if loss:
            self.losses.update(loss)
        self.EPE.update(EPE)
        self.RMSELIs.update(RMSE_Linear)
        self.RMSELGs.update(RMSE_Log)
        self.ABSRs.update(abs_relative)
        self.SQRs.update(sq_relative)
        self.DELTA.update(delta)
        self.DELTASQ.update(delta_sq)
        self.DELTACU.update(delta_cu)

    def get_info(self):
        return [self.losses.avg, self.EPE.avg, self.RMSELIs.avg, self.RMSELGs.avg, self.ABSRs.avg, self.SQRs.avg, self.DELTA.avg,
                self.DELTASQ.avg, self.DELTACU.avg]

    def calculate(self, depth, predict, loss=None):
        # only consider 1~80 meters
        mask = (depth >= 1) * (depth <= 80)
        EPE = torch.abs(predict[mask] - depth[mask]).float().mean().cpu().data
        RMSE_Linear = ((((predict[mask] - depth[mask]) ** 2).mean()) ** 0.5).cpu().data
        RMSE_Log = ((((torch.log(predict[mask]) - torch.log(depth[mask])) ** 2).mean()) ** 0.5).cpu().data
        abs_relative = (torch.abs(predict[mask] - depth[mask]) / depth[mask]).mean().cpu().data
        sq_relative = ((predict[mask] - depth[mask]) ** 2 / depth[mask]).mean().cpu().data
        delta = (torch.max(predict[mask] / depth[mask], depth[mask] / predict[mask]) < 1.25).float().mean().cpu().data
        delta_sq = (torch.max(predict[mask] / depth[mask],
                              depth[mask] / predict[mask]) < 1.25 ** 2).float().mean().cpu().data
        delta_cu = (torch.max(predict[mask] / depth[mask],
                              depth[mask] / predict[mask]) < 1.25 ** 3).float().mean().cpu().data
        self.update(loss, EPE, RMSE_Linear, RMSE_Log, abs_relative, sq_relative, delta, delta_sq, delta_cu)

    def tensorboard(self, writer, epoch, token='train'):
        writer.add_scalar(token + '/EPEs', self.EPE.avg, epoch)
        writer.add_scalar(token + '/RMSELIs', self.RMSELIs.avg, epoch)
        writer.add_scalar(token + '/RMSELGs', self.RMSELGs.avg, epoch)
        writer.add_scalar(token + '/ABSRs', self.ABSRs.avg, epoch)
        writer.add_scalar(token + '/SQRs', self.SQRs.avg, epoch)
        writer.add_scalar(token + '/DELTA', self.DELTA.avg, epoch)
        writer.add_scalar(token + '/DELTASQ', self.DELTASQ.avg, epoch)
        writer.add_scalar(token + '/DELTACU', self.DELTACU.avg, epoch)

    def print(self, iter, token):
        string = '{}:{}\tL {:.3f} EPE {:.3f} RLI {:.3f} RLO {:.3f} ABS {:.3f} SQ {:.3f} DEL {:.3f} DELQ {:.3f} DELC {:.3f}'.format(token, iter, *self.get_info())
        return string

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

