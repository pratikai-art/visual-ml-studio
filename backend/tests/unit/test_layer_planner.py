import pytest

from app.nodes.layer_planner import plan
from app.workflow.spec import NodeError


def test_plan_reshape_sequence_chunks():
    expression, shape, warnings = plan(
        "reshape",
        {"mode": "sequence_chunks", "features_per_step": 3},
        (12,),
    )

    assert expression == "Reshape((4, 3))"
    assert shape == (4, 3)
    assert warnings == []


def test_plan_conv2d_same_padding():
    expression, shape, warnings = plan(
        "conv2d",
        {"out_channels": 8, "kernel_size": 3, "stride": 1, "padding": "same"},
        (3, 28, 28),
    )

    assert expression == "nn.Conv2d(3, 8, 3, stride=1, padding=1)"
    assert shape == (8, 28, 28)
    assert warnings == []


def test_plan_attention_requires_divisible_embedding_size():
    with pytest.raises(NodeError, match="must be divisible"):
        plan("attention", {"num_heads": 3, "dropout": 0.1, "residual": True}, (5, 10))


def test_plan_softmax_returns_cross_entropy_warning():
    _, _, warnings = plan("activation", {"function": "softmax"}, (4,))

    assert warnings and "CrossEntropy" in warnings[0]
