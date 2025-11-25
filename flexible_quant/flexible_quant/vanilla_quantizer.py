
import torch


def quant_sym(x: torch.tensor, scaling: torch.tensor, nbits: int):
    q_max, q_min = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    return torch.round(x / scaling.unsqueeze(1)).clip(q_min, q_max).to(torch.int8)

def dequant_sym(x: torch.tensor, scaling: torch.tensor, target_dtype: torch.dtype):
    return x * scaling.unsqueeze(1).to(target_dtype)

def quant_asym(x: torch.tensor, scaling: torch.tensor, zeros: torch.tensor, nbits: int):
    q_max, q_min = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    return (torch.round(x / scaling.unsqueeze(1) - zeros.unsqueeze(1))).clip(q_min, q_max).to(torch.int8)
    
def dequant_asym(x: torch.tensor, scaling: torch.tensor, zeros: torch.tensor, target_dtype: torch.dtype):
    return (x + zeros.unsqueeze(1)) * scaling.unsqueeze(1).to(target_dtype)

def quant_fp(x: torch.tensor, scaling: torch.tensor, nbits: int, fp_format: str = None, zeros: torch.tensor = None):
    """
    Quantize tensor to floating-point format using block-wise quantization.
    Currently only FP8 is supported via PyTorch's native support.
    Supports both symmetric and asymmetric quantization.
    
    Args:
        x: Input tensor of shape (num_groups, group_size)
        scaling: Scale factors of shape (num_groups,)
        nbits: Number of bits (must be 8 for FP8)
        fp_format: Format string for FP8 (e.g., "e4m3", "e5m2")
        zeros: Zero-point offsets of shape (num_groups,) for asymmetric quantization. If None, uses symmetric quantization.
    
    Returns:
        Quantized tensor in FP8 format
    
    Raises:
        NotImplementedError: If nbits is not 8 (FP8 is the only supported floating-point format)
        ValueError: If fp_format is not "e4m3" or "e5m2"
    """
    if nbits != 8:
        raise NotImplementedError(
            f"Floating-point quantization with nbits={nbits} is not supported. "
            "Only FP8 (nbits=8) is currently supported via PyTorch's native float8 types."
        )
    
    # FP8: Use PyTorch's native support
    if fp_format == "e4m3":
        fp_dtype = torch.float8_e4m3fn
    elif fp_format == "e5m2":
        fp_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {fp_format}. Currently supported: e4m3, e5m2")
    
    fp_min = torch.finfo(fp_dtype).min
    fp_max = torch.finfo(fp_dtype).max
    
    # Scale the input - follow same pattern as integer quantization: x / scale - zeros
    if zeros is not None:
        # Asymmetric: same pattern as quant_asym: x / scale - zeros
        # This maps input range to [fp_min, fp_max] range
        # Convert FP8 zeros back to float for computation (like int zeros are used as-is)
        zeros_float = zeros.float() if zeros.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else zeros
        scaled = x / scaling.unsqueeze(1) - zeros_float.unsqueeze(1)
    else:
        # Symmetric: scale to fit in [-fp_max, fp_max] range
        scaled = x / scaling.unsqueeze(1)
    
    # Clamp to FP8 range and convert
    quantized = torch.clamp(scaled, min=fp_min, max=fp_max).to(fp_dtype)
    return quantized

def dequant_fp(x, scaling: torch.tensor, target_dtype: torch.dtype, nbits: int, fp_format: str = None, zeros: torch.tensor = None):
    """
    Dequantize floating-point tensor back to target dtype.
    Currently only FP8 is supported via PyTorch's native support.
    Supports both symmetric and asymmetric quantization.
    
    Args:
        x: FP8 quantized tensor
        scaling: Scale factors of shape (num_groups,)
        target_dtype: Target dtype for dequantization
        nbits: Number of bits used (must be 8 for FP8)
        fp_format: Format string for FP8 (unused, kept for API consistency)
        zeros: Zero-point offsets of shape (num_groups,) for asymmetric quantization. If None, uses symmetric dequantization.
    
    Returns:
        Dequantized tensor in target_dtype
    
    Raises:
        NotImplementedError: If nbits is not 8 (FP8 is the only supported floating-point format)
    """
    if nbits != 8:
        raise NotImplementedError(
            f"Floating-point dequantization with nbits={nbits} is not supported. "
            "Only FP8 (nbits=8) is currently supported via PyTorch's native float8 types."
        )
    
    # FP8: Use PyTorch's native support
    # Convert FP8 to float32 first
    dequantized = x.float()
    
    # Get FP8 range for dequantization
    if fp_format == "e4m3":
        fp_dtype = torch.float8_e4m3fn
    elif fp_format == "e5m2":
        fp_dtype = torch.float8_e5m2
    else:
        fp_dtype = torch.float8_e4m3fn  # default
    
    fp_min = torch.finfo(fp_dtype).min
    fp_max = torch.finfo(fp_dtype).max
    
    # Apply inverse scaling - follow same pattern as integer dequantization: (x + zeros) * scale
    if zeros is not None:
        # Asymmetric: same pattern as dequant_asym: (x + zeros) * scale
        # Convert FP8 zeros back to float for computation (like int zeros are used as-is)
        zeros_float = zeros.float() if zeros.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else zeros
        dequantized = (dequantized + zeros_float.unsqueeze(1)) * scaling.unsqueeze(1)
    else:
        # Symmetric: multiply by scale to restore original range
        dequantized = dequantized * scaling.unsqueeze(1)
    
    return dequantized.to(target_dtype)


class VanillaQuantizeMeta:
    def __init__(self, nbits, asym, compute_dtype, quant_dtype="int", fp_format="e4m3"):
        self.nbits = nbits
        # self.group_size = group_size
        # self.axis = axis # 1 for per-channel, 0 for per-token
        self.asym = asym
        self.compute_dtype = compute_dtype
        self.quant_dtype = quant_dtype  # "int" or "fp" (generalized floating-point)
        self.fp_format = fp_format  # "e4m3" or "e5m2" for FP8, None for auto
        
        # Set FP dtype if using floating-point quantization
        if self.quant_dtype == "fp":
            if self.nbits != 8:
                raise NotImplementedError(
                    f"Floating-point quantization with nbits={self.nbits} is not supported. "
                    "Only FP8 (nbits=8) is currently supported via PyTorch's native float8 types."
                )
            # FP8: Use PyTorch's native support
            if self.fp_format == "e4m3":
                self.fp_dtype = torch.float8_e4m3fn
            elif self.fp_format == "e5m2":
                self.fp_dtype = torch.float8_e5m2
            else:
                raise ValueError(f"Unsupported FP8 format: {self.fp_format}. Currently supported: e4m3, e5m2")
        elif self.quant_dtype == "int":
            self.fp_dtype = None
        else:
            raise ValueError(f"Unsupported quantization dtype: {self.quant_dtype}. Currently supported: int, fp")

    
class VanillaQuantizedTensor:
    def __init__(self, tensor, scaling, zeros, original_shape, axis, meta: VanillaQuantizeMeta):
        self.tensor = tensor
        self.scaling = scaling
        self.zeros = zeros
        self.original_shape = original_shape
        self.axis = axis
        self.meta = meta

    def dequantize(self):
        if self.meta.quant_dtype == "fp":
            # Floating-point dequantization (FP8 only)
            dequant = dequant_fp(self.tensor, self.scaling, self.meta.compute_dtype, 
                                self.meta.nbits, self.meta.fp_format, self.zeros)
        elif self.meta.asym:
            dequant = dequant_asym(self.tensor, self.scaling, self.zeros, self.meta.compute_dtype)
        else:
            dequant = dequant_sym(self.tensor, self.scaling, self.meta.compute_dtype)

        dequant = dequant.view(self.original_shape)

        if self.axis == 1:
            # axis=1: usually last dimension was transposed with penultimate
            max_dim = len(self.original_shape) - 1
            dequant = dequant.transpose(max_dim - 1, max_dim)
        elif self.axis == 2:
            # axis=2: per-token quantization
            # original shape assumed: (B, H, T, D)
            # after quantization we permuted: (B, T, H, D)
            B, H, T, D = self.original_shape
            dequant = dequant.view(B, T, H, D).permute(0, 2, 1, 3)
        
        return dequant

class VanillaQuantizer:
    def __init__(self, nbits, asym, compute_dtype, quant_dtype="int", fp_format="e4m3"):
        self.meta = VanillaQuantizeMeta(nbits, asym, compute_dtype, quant_dtype, fp_format)
    
    def quantize(self, tensor, q_group_size, axis):
        if axis == 1:
            max_dim = len(tensor.shape) - 1
            tensor = tensor.transpose(max_dim - 1, max_dim)
        if q_group_size == -1:
            #  [batch, num_heads, n_tokens, dim_per_head]
            if axis == 0:
                # [batch, num_heads, n_tokens, dim_per_head]
                # per-token & per-head
                # assert axis == 0 # must be per-token
                q_group_size = tensor.shape[-1] # take the last dimension

            elif axis == 2:
                # per-token only quantization
                # input shape assumed: (B, H, T, D)
                B, H, T, D = tensor.shape
                q_group_size = H * D
                rs = tensor.permute(0,2,1,3).reshape(B*T, q_group_size)
            rs = tensor.reshape(-1, q_group_size)
        
        # Handle floating-point quantization (FP8 only)
        if self.meta.quant_dtype == "fp":
            # Floating-point uses block-wise quantization with scaling
            fp_max = torch.finfo(self.meta.fp_dtype).max
            fp_min = torch.finfo(self.meta.fp_dtype).min
            fp_range = fp_max - fp_min
            
            if self.meta.asym:
                # Asymmetric quantization: compute scale based on range (max - min)
                # Follow same pattern as integer quantization: scale = range / quant_range
                _max = rs.max(dim=1).values
                _min = rs.min(dim=1).values
                range_vals = (_max - _min).clamp(min=1e-5)
                scale = range_vals / fp_range
                # Compute zero-point offset (same pattern as integer: _min / scale - fp_min)
                # This ensures _min maps to fp_min: _min / scale - zeros = fp_min
                # Quantize zeros to FP8 (like integer quantization quantizes zeros to int)
                zeros_float = (_min / scale) - fp_min
                zeros = torch.clamp(zeros_float, min=fp_min, max=fp_max).to(self.meta.fp_dtype)
            else:
                # Symmetric quantization: compute scale based on max absolute value
                # Same pattern as example: scale = max_abs / fp_max, then quantize as x / scale
                max_abs = rs.abs().max(dim=1).values
                scale = max_abs.clamp(min=1e-5) / fp_max
                zeros = None
            
            quant = quant_fp(rs, scale, self.meta.nbits, self.meta.fp_format, zeros)
        else:
            # Integer quantization
            q_max, q_min = 2 ** (self.meta.nbits - 1) - 1, -2 ** (self.meta.nbits - 1)
            
            if self.meta.asym:
                _max, _min = rs.max(dim=1).values, rs.min(dim=1).values
                scale = (_max - _min).clamp(min=1e-5).div(q_max - q_min)
                zeros = (_min / scale).round() - q_min
                quant = quant_asym(rs, scale, zeros, self.meta.nbits)
            else:
                scale = rs.abs().max(dim=1).values.clamp(min=1e-5).div(q_max)
                zeros = None
                quant = quant_sym(rs, scale, self.meta.nbits)
        
        return VanillaQuantizedTensor(quant, scale, zeros, tensor.shape, axis, self.meta)
