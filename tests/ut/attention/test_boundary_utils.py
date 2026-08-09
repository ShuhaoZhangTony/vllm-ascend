import pytest
import torch

from vllm_ascend.attention.utils import _split_decode_prefill_boundary


@pytest.mark.parametrize("num_reqs", [256, 257])
def test_boundary_preserves_pcp_graph_padding(num_reqs):
    query_start_loc = torch.tensor([0, 1, 6, *([6] * (num_reqs - 1))])
    query_lens = torch.tensor([1, 5])
    is_prefilling = torch.tensor([False, True, *([False] * (num_reqs - 2))])

    result = _split_decode_prefill_boundary(
        query_start_loc,
        num_reqs,
        num_tokens=6,
        max_query_len=5,
        query_lens=query_lens,
        treat_short_extends_as_decodes=False,
        is_prefilling=is_prefilling,
    )

    assert result == (1, num_reqs - 1, 1, 5)


@pytest.mark.parametrize(
    ("query_lens", "kwargs", "expected"),
    [
        ([1, 1, 5], {}, (2, 1, 2, 5)),
        ([5, 5, 5], {}, (0, 3, 0, 15)),
        ([2, 2, 0, 0], {"decode_threshold": 4, "require_uniform": True}, (4, 0, 4, 0)),
        (
            [1, 1, 5],
            {
                "treat_short_extends_as_decodes": False,
                "is_prefilling": torch.tensor([False, True, True]),
            },
            (1, 2, 1, 6),
        ),
    ],
)
def test_boundary_common_scheduler_cases(query_lens, kwargs, expected):
    query_start_loc = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()])

    result = _split_decode_prefill_boundary(
        query_start_loc,
        len(query_lens),
        num_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        **kwargs,
    )

    assert result == expected


def test_boundary_empty_batch():
    result = _split_decode_prefill_boundary(
        torch.tensor([0]),
        num_reqs=0,
        num_tokens=0,
        max_query_len=0,
    )

    assert result == (0, 0, 0, 0)
