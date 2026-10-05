from pydantic import BaseModel, Field

from pquant.core.constants import QuantizationGranularity


class BaseQuantizationModel(BaseModel):
    default_weight_keep_negatives: float = Field(default=1.0)
    default_weight_integer_bits: float = Field(default=0.0)
    default_weight_fractional_bits: float = Field(default=7.0)
    default_data_keep_negatives: float = Field(default=0.0)
    default_data_integer_bits: float = Field(default=0.0)
    default_data_fractional_bits: float = Field(default=7.0)
    quantize_input: bool = Field(default=True)
    quantize_output: bool = Field(default=False)
    granularity: QuantizationGranularity = Field(default=QuantizationGranularity.PER_TENSOR)
    dynamic_data_quantization: bool = Field(default=False)
    enable_quantization: bool = Field(default=True)
    hgq_gamma: float = Field(default=0.0003)
    hgq_beta: float = Field(default=1e-5)
    hgq_heterogeneous: bool = Field(default=True)
    layer_specific: dict[str, dict] = Field(default_factory=dict)
    use_high_granularity_quantization: bool = Field(default=False)
    # (min, max) bounds on the total learned bitwidth k+i+f of every HGQ quantizer, or None for no bounds. A
    # non-zero minimum stops HGQ from pruning a tensor away (a zero-bit tensor outputs zeros), which matters when
    # per-tensor HGQ is combined with pruning. Layers take a `bitwidth_minmax` argument that overrides this.
    hgq_bitwidth_minmax: tuple[float, float] | None = Field(default=None)
    # Floor on learned INTEGER bits i, i.e. on each lane's representable range (2**i).
    # Distinct from hgq_bitwidth_minmax, which floors the total k+i+f and is not
    # sufficient on its own: i=-23 with f=24 sums to 2, satisfying a 2-bit total floor
    # while leaving the lane a range of ~1e-7 that any real weight saturates against.
    hgq_integer_bits_min: float | None = Field(default=None)
    # PQSoftmax floors the integer bits of its inverse-table input lane at ceil(log2(keys + 1)): that lane carries
    # the sum of the exponentials, which lies in [0, keys], and without the floor the bitwidth pressure can
    # saturate the sum to zero and switch the attention off (a total-bitwidth bound cannot prevent this). Off
    # only for experiments.
    softmax_inverse_lane_floor: bool = Field(default=True)
    # Range floors on every PQSoftmax lane whose range is known analytically (the inverse floor above included):
    # exp output i >= 1 (holds exp(0) = 1), inverse input i >= ceil(log2(keys + 1)) with f >= 0, inverse output
    # i >= 1 with f >= ceil(log2 keys) (holds 1 and resolves 1/keys), output i >= 1 with f >= ceil(log2 keys)
    # (holds a weight of 1 and of 1/keys). Each floor also raises the lane's minimum total width so the
    # fractional bits cannot be traded away. The score lane is left free: attention may still go uniform under
    # pressure, but as a normalized softmax.
    softmax_range_floors: bool = Field(default=False)
    use_real_tanh: bool = Field(default=False)
    overflow_mode_parameters: str = Field(default="SAT")
    overflow_mode_data: str = Field(default="SAT")
    round_mode: str = Field(default="RND")
    use_relu_multiplier: bool = Field(default=True)
