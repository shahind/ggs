"""8-bit float storage for frozen Linear layers (needs torch).

Like kohya's ``--fp8_base``: the weight is a plain cast to float8_e4m3fn (no scaling) and is cast back to the
compute dtype on every call. It halves the VRAM of the frozen base model during LoRA training. Only for frozen
weights; LoRA layers (any module path containing ``lora_``) are never converted.
"""

from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn


class Fp8Linear(torch.nn.Linear):
    @classmethod
    def from_linear(cls, lin: torch.nn.Linear) -> Fp8Linear:
        new = cls.__new__(cls)
        torch.nn.Module.__init__(new)
        new.in_features, new.out_features = lin.in_features, lin.out_features
        new.weight = torch.nn.Parameter(lin.weight.detach().to(FP8), requires_grad=False)
        new.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach(), requires_grad=False)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") and x.is_cuda else x.dtype
        bias = None if self.bias is None else self.bias.to(dtype)
        return torch.nn.functional.linear(x.to(dtype), self.weight.to(dtype), bias)


def linears_to_fp8(model: torch.nn.Module, skip: tuple[str, ...] = ("lora_",)) -> int:
    """Swap every plain ``nn.Linear`` (also a PEFT ``base_layer``) for an Fp8Linear. Returns weights converted."""
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if type(child) is torch.nn.Linear and not any(s in full for s in skip):
                setattr(module, child_name, Fp8Linear.from_linear(child))
                n += child.weight.numel()
    return n
