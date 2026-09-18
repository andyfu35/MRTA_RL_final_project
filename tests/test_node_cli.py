from marl2d.node_cli import build_parser


def test_node_cli_defaults_to_three_file_deployment_layout():
    args = build_parser().parse_args([])
    assert args.config == "config.json"
    assert args.node_id is None
    assert args.device == "cpu"


def test_node_cli_allows_manual_node_identity_for_offline_labs():
    args = build_parser().parse_args(
        ["--node-id", "pc_runner_0", "--check"]
    )
    assert args.node_id == "pc_runner_0"
    assert args.check is True
