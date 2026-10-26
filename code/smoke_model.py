# -*- coding: utf-8 -*-
"""Smoke test: full forward + backward through the prototype with the
Section III-G objective (autograd audit required by Section III-H)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from eatqi.losses import total_loss       # noqa: E402
from eatqi.model import PrototypeEAQI     # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    net = PrototypeEAQI()
    imgs = torch.rand(3, 3, 128, 128)
    y = torch.tensor([0.1, 0.5, 0.9])
    out = net(imgs)
    targets = {
        "family": torch.tensor([0, 1, 2]),
        "severity": torch.tensor([0, 1, 2]),
        "spatial": torch.rand(3, 1, 32, 32),
        "edge_interaction": torch.tensor([0.0, 1.0, 1.0]),
    }
    loss, log = total_loss(out["severity"], y, out, targets)
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in net.parameters()
            if p.grad is not None)
    print("loss %.4f  grad_sum %.4f  (autograd OK)" % (loss.item(), g))
    print(log)


if __name__ == "__main__":
    main()
