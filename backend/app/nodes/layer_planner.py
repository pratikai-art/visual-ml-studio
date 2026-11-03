"""Pure-Python shape planner for visual deep-learning layers."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from ..workflow.spec import NodeError


def _prod(shape: tuple[int, ...]) -> int:
    return int(math.prod(shape))


def _bad(name: str, shape: tuple[int, ...], expected: str, hint: str) -> None:
    raise NodeError(
        f"{name} needs a {expected} tensor but receives shape {shape}. {hint}",
        f"Shapes exclude the batch dimension. Current per-sample shape: {shape}",
    )


def _conv_out(length: int, kernel: int, stride: int, padding: int) -> int:
    output = (length + 2 * padding - kernel) // stride + 1
    if output < 1:
        raise NodeError(
            f"The kernel ({kernel}) is larger than the input length ({length}); "
            "the output would be empty. Use a smaller kernel or padding='same'."
        )
    return output


def plan(
    kind: str,
    params: dict[str, Any],
    shape: Iterable[int],
) -> tuple[str, tuple[int, ...], list[str]]:
    """Return a constructor expression, output shape, and warnings.

    Shapes are per-sample and therefore exclude the batch dimension. The
    expression is deliberately kept as the single representation used by the
    runtime and exported code.
    """
    current = tuple(int(value) for value in shape)
    warnings: list[str] = []

    if kind == "reshape":
        total = _prod(current)
        mode = params["mode"]
        if mode == "sequence":
            output = (total, 1)
        elif mode == "channels_1d":
            output = (1, total)
        elif mode == "sequence_chunks":
            features = int(params["features_per_step"])
            if features <= 0:
                raise NodeError("Features per step must be greater than zero.")
            if total % features:
                raise NodeError(
                    f"{total} features cannot be split into steps of {features} "
                    f"features ({total} is not divisible by {features})."
                )
            output = (total // features, features)
        else:
            channels, height, width = (
                int(params["channels"]),
                int(params["height"]),
                int(params["width"]),
            )
            if min(channels, height, width) <= 0:
                raise NodeError("Image channels, height, and width must be greater than zero.")
            if channels * height * width != total:
                raise NodeError(
                    f"An image of {channels}x{height}x{width} has "
                    f"{channels * height * width} values, but each row has {total} "
                    "features. Adjust channels/height/width so they multiply to "
                    f"{total}."
                )
            output = (channels, height, width)
        return f"Reshape({output!r})", output, warnings

    if kind == "linear":
        output_features = int(params["out_features"])
        if output_features <= 0:
            raise NodeError("Linear output features must be greater than zero.")
        if not current:
            _bad("Linear", current, "at least 1-D", "Add a Reshape layer first.")
        return f"nn.Linear({current[-1]}, {output_features})", current[:-1] + (output_features,), warnings

    if kind == "activation":
        function = params["function"]
        expressions = {
            "relu": "nn.ReLU()",
            "gelu": "nn.GELU()",
            "sigmoid": "nn.Sigmoid()",
            "tanh": "nn.Tanh()",
            "elu": "nn.ELU()",
            "softmax": "nn.Softmax(dim=-1)",
        }
        if function == "leaky_relu":
            expression = f"nn.LeakyReLU({params['negative_slope']})"
        else:
            expression = expressions.get(function)
        if expression is None:
            raise NodeError(f"Unknown activation function '{function}'.")
        if function == "softmax":
            warnings.append(
                "Softmax before the final layer is usually wrong: the trainer applies "
                "CrossEntropy, which already includes softmax. Use Softmax only for experiments."
            )
        return expression, current, warnings

    if kind == "dropout":
        return f"nn.Dropout({params['p']})", current, warnings

    if kind == "batchnorm":
        if len(current) == 3:
            return f"nn.BatchNorm2d({current[0]})", current, warnings
        if len(current) == 2:
            warnings.append(
                "BatchNorm1d normalises axis 0 of (channels, length). For "
                "(steps, features) sequences use Layer Normalization instead."
            )
        if not current:
            _bad("BatchNorm", current, "at least 1-D", "Add a Reshape layer first.")
        return f"nn.BatchNorm1d({current[0]})", current, warnings

    if kind == "layernorm":
        if not current:
            _bad("LayerNorm", current, "at least 1-D", "Add a Reshape layer first.")
        return f"nn.LayerNorm({current[-1]})", current, warnings

    if kind == "flatten":
        return "nn.Flatten()", (_prod(current),), warnings

    if kind == "transpose":
        if len(current) != 2:
            _bad("Transpose", current, "2-D (a, b)", "Use Reshape first.")
        return "Transpose()", (current[1], current[0]), warnings

    if kind == "conv1d":
        if len(current) != 2:
            _bad(
                "Conv1D",
                current,
                "(channels, length)",
                "Add a Reshape layer with mode 'channels_1d' "
                "(or Transpose a (steps, features) sequence).",
            )
        kernel, stride = int(params["kernel_size"]), int(params["stride"])
        padding = kernel // 2 if params["padding"] == "same" else 0
        return (
            f"nn.Conv1d({current[0]}, {params['out_channels']}, {kernel}, "
            f"stride={stride}, padding={padding})",
            (int(params["out_channels"]), _conv_out(current[1], kernel, stride, padding)),
            warnings,
        )

    if kind == "conv2d":
        if len(current) != 3:
            _bad("Conv2D", current, "(channels, height, width)", "Add a Reshape layer with mode 'image'.")
        kernel, stride = int(params["kernel_size"]), int(params["stride"])
        padding = kernel // 2 if params["padding"] == "same" else 0
        return (
            f"nn.Conv2d({current[0]}, {params['out_channels']}, {kernel}, "
            f"stride={stride}, padding={padding})",
            (
                int(params["out_channels"]),
                _conv_out(current[1], kernel, stride, padding),
                _conv_out(current[2], kernel, stride, padding),
            ),
            warnings,
        )

    if kind == "pool":
        if len(current) not in (2, 3):
            _bad(
                "Pooling",
                current,
                "(channels, length) or (channels, height, width)",
                "Add Reshape + Convolution first.",
            )
        kernel = int(params["kernel_size"])
        stride = int(params["stride"]) or kernel
        pool_class = ("Max" if params["pool_type"] == "max" else "Avg") + f"Pool{len(current) - 1}d"
        dimensions = tuple(_conv_out(value, kernel, stride, 0) for value in current[1:])
        return f"nn.{pool_class}({kernel}, stride={stride})", (current[0],) + dimensions, warnings

    if kind == "global_pool":
        if len(current) not in (2, 3):
            _bad(
                "Global Average Pooling",
                current,
                "(channels, length) or (channels, height, width)",
                "Use Flatten for plain vectors.",
            )
        return "GlobalAvgPool()", (current[0],), warnings

    if kind == "sequence_mean":
        if len(current) != 2:
            _bad("Sequence Mean", current, "(steps, features)", "Add a Reshape layer with mode 'sequence'.")
        return "SequenceMean()", (current[1],), warnings

    if kind in ("lstm", "gru"):
        if len(current) != 2:
            _bad(
                kind.upper(),
                current,
                "(steps, features)",
                "Add a Reshape layer with mode 'sequence' (each feature becomes one "
                "time step) or 'sequence_chunks'.",
            )
        direction_multiplier = 2 if params["bidirectional"] else 1
        output = (
            (int(params["hidden_size"]) * direction_multiplier,)
            if params["return_sequences"] == "last"
            else (current[0], int(params["hidden_size"]) * direction_multiplier)
        )
        expression = (
            f"RNNBlock({kind!r}, {current[1]}, {params['hidden_size']}, "
            f"{params['num_layers']}, {params['bidirectional']}, {params['dropout']}, "
            f"{params['return_sequences']!r})"
        )
        return expression, output, warnings

    if kind in ("attention", "transformer_encoder"):
        if len(current) != 2:
            _bad(
                "Attention/Transformer",
                current,
                "(steps, features)",
                "Add a Reshape (sequence) and a Linear layer to create token embeddings first.",
            )
        if current[1] % int(params["num_heads"]):
            raise NodeError(
                f"The feature size ({current[1]}) must be divisible by the number of heads "
                f"({params['num_heads']}). Add a Linear layer to change the embedding size, "
                "or change the heads."
            )
        if kind == "attention":
            return (
                f"SelfAttention({current[1]}, {params['num_heads']}, "
                f"{params['dropout']}, {params['residual']})",
                current,
                warnings,
            )
        return (
            f"TransformerEncoderBlock({current[0]}, {current[1]}, {params['num_heads']}, "
            f"{params['ff_dim']}, {params['num_layers']}, {params['dropout']}, "
            f"{params['positional']})",
            current,
            warnings,
        )

    raise NodeError(f"Unknown layer '{kind}'.")
