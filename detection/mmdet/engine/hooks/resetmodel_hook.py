import torch
from spikingjelly.clock_driven import functional
from typing import Optional, Sequence
from mmengine.hooks import Hook

from mmdet.registry import HOOKS

global_idx = 0


@HOOKS.register_module()
class ResetModelHook(Hook):
    """Docstring for NewHook.
    """

    def __init__(self, **kwargs):
        super(ResetModelHook, self).__init__(
            **kwargs)
        self.iters = 0

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        # import pdb; pdb.set_trace()
        torch.cuda.synchronize()
        functional.reset_net(runner.model)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         outputs: None,
                         data_batch: Optional[Sequence[dict]] = None) -> None:
        self.iters += 1
        if self.iters % 50 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        functional.reset_net(runner.model)

    def after_train_epoch(self,
                         runner,
                         batch_idx: int,
                         outputs: None,
                         data_batch: Optional[Sequence[dict]] = None) -> None:

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        functional.reset_net(runner.model)

    def before_val_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: Optional[Sequence[dict]] = None) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        functional.reset_net(runner.model)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       outputs: None,
                       data_batch: Optional[Sequence[dict]] = None) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        functional.reset_net(runner.model)

    def before_val_epoch(self, runner) -> None:
        # import pdb; pdb.set_trace()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        functional.reset_net(runner.model)

    # def after_val_epoch(self,
    #                       runner,
    #                       batch_idx: int,
    #                       data_batch: Optional[Sequence[dict]] = None) -> None:
    #     # import pdb; pdb.set_trace()
    #     torch.cuda.synchronize()
    #     torch.cuda.empty_cache()
    #     functional.reset_net(runner.model)

    def before_test_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: Optional[Sequence[dict]] = None) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        functional.reset_net(runner.model)

    def before_test_epoch(self, runner) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        functional.reset_net(runner.model)

    # def after_test_epoch(self,
    #                      runner,
    #                      batch_idx: int,
    #                      data_batch: Optional[Sequence[dict]] = None) -> None:
    #     torch.cuda.synchronize()
    #     torch.cuda.empty_cache()
    #     functional.reset_net(runner.model)
