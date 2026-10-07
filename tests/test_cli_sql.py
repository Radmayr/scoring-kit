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


def test_catalog_dependencies(tmp_path):
    import shutil

    from scoring_kit.catalog import build_catalog
    from tests.conftest import ROOT

    # calibration ждёт таблицу, в которую пишет demo_scoring → зависимость demo_scoring → demo_calibration
    d = tmp_path / "calib"
    shutil.copytree(ROOT / "examples" / "demo_calibration", d)
    text = (d / "pipeline.yaml").read_text(encoding="utf-8").replace(
        "tables: [sandbox.dev_calib_in, sandbox.apply_calib_in]", "tables: [sandbox.demo_scores_fresh]"
    )
    (d / "pipeline.yaml").write_text(text, encoding="utf-8")
    catalog = build_catalog([DEMO, d, ROOT / "examples" / "demo_multi", tmp_path / "missing"])
    assert "demo_scoring --> demo_calibration" in catalog
    assert "ежедневно 07:30 МСК" in catalog
    assert "| `demo_tenant/demo_d1/0.0.1` | demo_multi |" in catalog
    assert "Невалидные конфиги" in catalog


def test_cli_bundle_and_dlh_debug(tmp_path, model_path, capsys):
    import pickle

    from tests.conftest import ROOT, make_frame

    dlh = ROOT / "examples" / "demo_scoring_dlh"
    out = tmp_path / "bundle.pkl"
    assert main(["bundle", str(dlh), "--model", str(model_path), "-o", str(out)]) == 0
    bundle = pickle.loads(out.read_bytes())
    assert bundle.features[0] == "segment"
    csv = tmp_path / "d.csv"
    make_frame(300, seed=3).to_csv(csv, index=False)
    assert main(["debug", str(dlh), "--data", str(csv), "--model", str(out), "--out", str(tmp_path / "o")]) == 0
    assert "sandbox.demo_scores_dlh: 300" in capsys.readouterr().out


def test_cli_compare_sql_multi(capsys):
    from tests.conftest import ROOT

    assert main(["compare-sql", str(ROOT / "examples" / "demo_multi")]) == 0
    out = capsys.readouterr().out
    assert out.count("full outer join sandbox.demo_multi_scores_fresh_shadow") == 4
