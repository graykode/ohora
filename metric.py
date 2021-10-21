import torch
import torchmetrics


class Metric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):  # type: ignore
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("num_total", default=torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("num_correct", default=torch.zeros(()), dist_reduce_fx="sum")

    def update(self, num_total, num_correct):  # type: ignore
        self.num_total += torch.tensor(num_total).type_as(self.num_total)  # type: ignore
        self.num_correct += torch.tensor(num_correct).type_as(self.num_correct)  # type: ignore

    def compute(self):  # type: ignore
        return (self.num_correct / self.num_total) * 100
