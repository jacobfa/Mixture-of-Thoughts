import pytest
import torch

from mot.experts.backbone import layer_segments
from mot.experts.streams import IGNORE_INDEX, PAD_TIME, encode
from mot.interaction.attention import alignment_mask
from mot.losses import routing_loss
from mot.routing.router import Router


@pytest.mark.parametrize(
    ("layers", "q", "placement", "expected"),
    [
        (32, 8, "uniform", [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 32)]),
        (32, 4, "shallow", [(0, 4), (4, 8), (8, 12), (12, 16), (16, 32)]),
        (32, 4, "deep", [(0, 20), (20, 24), (24, 28), (28, 32), (32, 32)]),
        (32, 4, "intermediate", [(0, 12), (12, 16), (16, 20), (20, 24), (24, 32)]),
    ],
)
def test_layer_segments(layers, q, placement, expected):
    segs = layer_segments(layers, q, placement)
    assert segs == expected
    assert segs[0][0] == 0 and segs[-1][1] == layers
    assert all(a[1] == b[0] for a, b in zip(segs, segs[1:], strict=False))


def test_layer_segments_rejects_overfull():
    with pytest.raises(ValueError):
        layer_segments(4, 8, "uniform")


def test_encode_labels_and_time(tiny_model):
    tok = tiny_model.pool[0].tokenizer
    prompt, completion = "Question: 2+2? Answer:", " 4"
    ids, time, labels = encode(tok, [prompt], [completion], max_length=64, add_eos=True)
    ids, time, labels = ids[0], time[0], labels[0]
    assert ids[-1] == tok.eos_token_id and labels[-1] == tok.eos_token_id
    assert (time[1:] >= time[:-1]).all()
    supervised = labels != IGNORE_INDEX
    assert supervised.sum() >= 2
    assert tok.decode(ids[supervised][:-1]).strip() == completion.strip()


def test_alignment_mask_blocks_peer_future(tiny_model):
    """With heterogeneous tokenizers a primary token must never see a peer token ending after it."""
    text = "The quick brown fox jumps over the lazy dog."
    a, b = tiny_model.pool[0].tokenizer, tiny_model.pool[1].tokenizer
    _, ta, _ = encode(a, [text], None, 64, add_eos=False)
    _, tb, _ = encode(b, [text], None, 64, add_eos=False)
    mask = alignment_mask(ta[0][None], tb[0][None])[0, 0]
    qa, kb = ta[0][:, None], tb[0][None, :]
    assert torch.equal(mask, kb <= qa)
    assert not mask[0, -1]
    assert mask[-1].all()
    # the primary's own stream is always among the keys, so no query row can be fully masked
    own = alignment_mask(ta[0][None], torch.cat([ta[0], tb[0]])[None])[0, 0]
    assert own.any(-1).all()


def test_alignment_mask_hides_padding():
    q = torch.tensor([[3, 5, PAD_TIME]])
    k = torch.tensor([[PAD_TIME, 2, 4, 6]])
    mask = alignment_mask(q, k)[0, 0]
    assert mask.tolist() == [[False, True, False, False], [False, True, True, False], [True, True, True, True]]


def test_router_selection_and_gate():
    torch.manual_seed(0)
    router = Router(8, 16, num_experts=5, top_k=2, temperature=1.0, dropout=0.0)
    scores = router(torch.randn(4, 8))
    hard = router.select(scores, noisy=False, straight_through=False)
    assert hard.active.shape == (4, 2)
    assert torch.equal(hard.primary, scores.argmax(-1))
    assert torch.equal(hard.gate.sum(-1), torch.full((4,), 2.0))
    assert not hard.gate.requires_grad

    st = router.select(scores, noisy=True, straight_through=True)
    assert torch.equal(st.gate.detach().sum(-1), torch.full((4,), 2.0))
    st.gate.sum().backward()
    assert router.mlp[0].weight.grad is not None


def test_routing_losses():
    scores = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, 0.0]])
    onehot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert routing_loss(scores, onehot, "ce") < routing_loss(scores, onehot.flip(0), "ce")
    assert routing_loss(scores, onehot, "bce") < routing_loss(scores, onehot.flip(0), "bce")
    assert routing_loss(scores, onehot, "none") == 0
    soft = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert torch.isfinite(routing_loss(scores, soft, "ce"))
    with pytest.raises(ValueError):
        routing_loss(scores, onehot, "hinge")
