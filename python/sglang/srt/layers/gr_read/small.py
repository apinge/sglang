# Kernel source: pyhip d3510c6 (final algorithm a716010).
# Only the final down/up launch path is retained; no test/control imports.

from dataclasses import dataclass, replace

from . import up as prefetch_up
from .down import DownConfig


@dataclass(frozen=True)
class SmallConfig:
    down: DownConfig = DownConfig(block_k=128, global_split=4, prefetch_unroll=2)
    up_n: int = 128
    up_waves: int = 4

    @property
    def name(self):
        return f"small_{self.down.name}_un{self.up_n}_uw{self.up_waves}"

    def validate(self):
        self.down.validate()
        if self.down.block_m != 16:
            raise ValueError("small-batch down must retain BM16")
        self.up_config(
            prefetch_up.Config(compensate_hidden=True, hidden_pad=4)
        ).validate()

    def up_config(self, base):
        return replace(
            base,
            down_mode="partial" if self.down.global_split > 1 else "wave_splitk",
            split_k=self.down.global_split,
            up_n=self.up_n,
            waves=self.up_waves,
            compensate_hidden=True,
            prefetch_low=False,
        )


def default_config(rows):
    if not 1 <= rows <= 16:
        raise ValueError("small-batch pipeline requires T=1..16")
    # Full-call graph timings choose BN128 at both ends of the small-T range.
    return SmallConfig(up_n=128 if rows <= 6 or rows == 16 else 64)
