# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import MaxNLocator


class WaveformTimeScale(mscale.ScaleBase):
    name = "merger_zoom"

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis)
        self.t_initial = float(kwargs.pop("t_initial", 0.0))
        self.t_merger = float(kwargs.pop("t_merger"))
        self.t_ringdown = float(kwargs.pop("t_ringdown"))
        self.t_final = float(kwargs.pop("t_final"))
        self.merger_stretch = float(
            kwargs.pop("merger_stretch", 0.75 * (self.t_final - self.t_initial) / (self.t_ringdown - self.t_merger))
        )

    def get_transform(self):
        return self.WaveformTimeTransform(
            self.merger_stretch, self.t_initial, self.t_merger, self.t_ringdown, self.t_final
        )

    def set_default_locators_and_formatters(self, axis):
        class StretchingLocator(MaxNLocator):
            def __init__(self, t_initial, t_merger, t_ringdown, t_final):
                MaxNLocator.__init__(self, nbins=9, steps=[1, 2, 5, 10])
                self.t_initial = t_initial
                self.t_merger = t_merger
                self.t_ringdown = t_ringdown
                self.t_final = t_final

            def tick_values(self, vmin, vmax):
                self.set_params(nbins=6, prune=None)
                ticks = list(super().tick_values(max(vmin, self.t_initial), min(vmax, self.t_merger)))
                self.set_params(nbins=11 - len(ticks), prune="both")
                ticks += list(super().tick_values(max(vmin, self.t_merger), min(vmax, self.t_ringdown)))
                self.set_params(nbins=9, prune=None)
                return np.array(ticks)

        axis.set_major_locator(StretchingLocator(self.t_initial, self.t_merger, self.t_ringdown, self.t_final))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, self.t_initial), min(vmax, self.t_final)

    class WaveformTimeTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, merger_stretch, t_initial, t_merger, t_ringdown, t_final):
            mtransforms.Transform.__init__(self)
            self.merger_stretch, self.t_initial, self.t_merger, self.t_ringdown, self.t_final = (
                merger_stretch,
                t_initial,
                t_merger,
                t_ringdown,
                t_final,
            )

        def transform_non_affine(self, a):
            return np.piecewise(
                a,
                [(a < self.t_merger), (a >= self.t_merger) & (a < self.t_ringdown), a >= self.t_ringdown],
                [
                    lambda x: x,
                    lambda x: self.t_merger + self.merger_stretch * (x - self.t_merger),
                    lambda x: (self.merger_stretch - 1) * (self.t_ringdown - self.t_merger) + x,
                ],
            )

        def inverted(self):
            return WaveformTimeScale.InvertedWaveformTimeTransform(
                self.merger_stretch, self.t_initial, self.t_merger, self.t_ringdown, self.t_final
            )

    class InvertedWaveformTimeTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, merger_stretch, t_initial, t_merger, t_ringdown, t_final):
            mtransforms.Transform.__init__(self)
            self.merger_stretch, self.t_initial, self.t_merger, self.t_ringdown, self.t_final = (
                merger_stretch,
                t_initial,
                t_merger,
                t_ringdown,
                t_final,
            )

        def transform_non_affine(self, f_a):
            return np.piecewise(
                f_a,
                [
                    (f_a < self.t_merger),
                    (
                        (f_a >= (1 - self.merger_stretch) * self.t_merger + self.merger_stretch * self.t_merger)
                        & (f_a < (1 - self.merger_stretch) * self.t_merger + self.merger_stretch * self.t_ringdown)
                    ),
                    f_a >= self.merger_stretch * self.t_ringdown - (self.merger_stretch - 1) * (self.t_merger),
                ],
                [
                    lambda x: x,
                    lambda x: (x - (1 - self.merger_stretch) * self.t_merger) / self.merger_stretch,
                    lambda x: x - (self.merger_stretch - 1) * (self.t_ringdown - self.t_merger),
                ],
            )

        def inverted(self):
            return WaveformTimeScale.WaveformTimeTransform(
                self.merger_stretch, self.t_initial, self.t_merger, self.t_ringdown, self.t_final
            )


mscale.register_scale(WaveformTimeScale)
