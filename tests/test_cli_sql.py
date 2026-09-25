from scoring_kit.cli import main
from scoring_kit.spec import load_pipeline
from scoring_kit.sql import compare_sql, ddl
from tests.conftest import DEMO


def test_ddl():
    p = load_pipeline(DEMO)
    text = ddl(p.sinks[0], p)
    assert text.startswith("create table sandbox.demo_scores_fresh (")
    assert "client_id  bigint," in text
    assert text.endswith("distributed randomly;")


def test_ddl_with_service_columns_and_distribution():
    p = load_pipeline(DEMO)
    sink = p.sinks[0].model_copy(update={"add_scored_at": True, "distributed_by": ["client_id"]})
    text = ddl(sink, p)
    assert "scored_at" in text and "timestamp" in text
    assert text.endswith("distributed by (client_id);")


def test_compare_sql():
    q = compare_sql("s.a", "s.a_shadow", ["k1", "k2"], "score", 1e-9)
    assert "full outer join s.a_shadow s using (k1, k2)" in q
    assert "max(abs(p.score - s.score))" in q


def test_cli_full_cycle(tmp_path, capsys):
    target = tmp_path / "my_model"
    assert main(["new", str(target)]) == 0
    assert main(["validate", str(target)]) == 0
    out = tmp_path / "dags"
    assert main(["render", str(target), str(DEMO), "-o", str(out), "--shadow"]) == 0
    assert (out / "my_model_shadow.py").exists()
    assert (out / "demo_scoring_shadow.py").exists()
    assert main(["ddl", str(DEMO), "--shadow"]) == 0
    assert main(["compare-sql", str(DEMO)]) == 0
    printed = capsys.readouterr().out
    assert "create table sandbox.demo_scores_fresh_shadow (" in printed
    assert "full outer join sandbox.demo_scores_fresh_shadow s using (client_id, account_id)" in printed


def test_cli_reports_config_errors(tmp_path, capsys):
    assert main(["validate", str(tmp_path)]) == 1
    assert "Не найден" in capsys.readouterr().err
