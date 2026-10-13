"""Конструктор: пресеты, сборка страницы и логика формы (через node, если он установлен)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scoring_kit.cli import main
from scoring_kit.constructor import (END, START, PresetsError, build_constructor, check_presets, merge_presets,
                                     presets_from_pipelines)
from scoring_kit.render import render_dir
from scoring_kit.spec import load_pipeline

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(p for p in (ROOT / "examples").iterdir() if (p / "pipeline.yaml").exists())

PRESETS = {
    "gp_service": "gpsvc",
    "gitlab_new_file_url": "https://gitlab.example/team/scoring-pipelines/-/new/main?file_name=pipelines/{name}/pipeline.yaml",
    "domains": ["collection", "risk"],
    "images": [
        {"id": "lgbm", "kind": "model", "label": "LightGBM", "image": "reg/lgbm:1", "requirements": ["lightgbm==3.3.5"]},
        {"id": "cal", "kind": "job", "label": "Калибровка", "image": "reg/base:1"},
    ],
}


def _team(html: str) -> dict:
    block = html[html.index(START) + len(START):html.index(END)]
    return json.loads(block.strip().removeprefix("var TEAM = ").rstrip(";"))


def test_page_is_standalone_with_demo_presets():
    html = build_constructor()
    assert html.startswith("<!doctype html>")
    assert "<title>Конструктор скоринга</title>" in html
    assert _team(html)["images"], "демо-пресеты должны быть в шаблоне"


def test_presets_injected():
    team = _team(build_constructor(PRESETS))
    assert team["gp_service"] == "gpsvc"
    assert team["images"][1]["requirements"] == []
    assert team["domains"] == ["collection", "risk"]


@pytest.mark.parametrize("bad, msg", [
    ({"images": [{"id": "x", "kind": "gpu", "label": "x", "image": "x"}]}, "kind"),
    ({"images": [{"id": "custom", "kind": "model", "label": "x", "image": "x"}]}, "зарезервированное"),
    ({"images": [{"id": "x", "kind": "model", "label": "x"}]}, "image"),
    ({"registry": "x"}, "неизвестные"),
])
def test_presets_errors(bad, msg):
    with pytest.raises(PresetsError, match=msg):
        check_presets(bad)


def test_presets_from_pipelines_and_merge():
    found = presets_from_pipelines(EXAMPLES)
    kinds = {i["kind"] for i in found["images"]}
    assert kinds == {"model", "job", "spark"}
    merged = check_presets(merge_presets(PRESETS, found))
    assert merged["gp_service"] == "gpsvc"                      # файл главнее
    assert merged["images"][0]["id"] == "lgbm"
    assert len(merged["images"]) == len(PRESETS["images"]) + len(found["images"])


def test_cli_constructor(tmp_path):
    presets = tmp_path / "presets.yaml"
    presets.write_text(yaml.safe_dump(PRESETS, allow_unicode=True), encoding="utf-8")
    out = tmp_path / "c.html"
    assert main(["constructor", "--presets", str(presets), "--pipelines", *map(str, EXAMPLES), "-o", str(out)]) == 0
    assert _team(out.read_text(encoding="utf-8"))["gp_service"] == "gpsvc"


# ---------------------------------------------------------------- логика формы в node

HARNESS = r"""
const fs = require("fs"), vm = require("vm"), path = require("path");
const [html, work] = process.argv.slice(2);
const src = fs.readFileSync(html, "utf8");
const grab = id => { const a = src.indexOf('<script id="' + id + '">'); return src.slice(src.indexOf(">", a) + 1, src.indexOf("</script>", a)); };
const ctx = {}; vm.createContext(ctx);
vm.runInContext(grab("team-presets") + "\n" + grab("logic") + "\nthis.API={defaultState,buildYaml,validate,fromConfig,parseColumns};", ctx);
const A = ctx.API, report = {};
function emit(name, st) {
  const dir = path.join(work, "out", name); fs.mkdirSync(dir, {recursive: true});
  fs.writeFileSync(path.join(dir, "pipeline.yaml"), A.buildYaml(st));
  report[name] = A.validate(st).map(p => p.text);
}
let s = A.defaultState();
Object.assign(s, {name: "t_single", owner: "u"}); Object.assign(s.source, {query: "select 1", key: "id"});
Object.assign(s.model, {mrid: "t/m/1", packed: false, features: "a, b, c", cats: ["c"]}); s.sink.table = "s.t_single";
emit("t_single", s);
s = A.defaultState();
Object.assign(s, {name: "t_packed", owner: "u"}); Object.assign(s.source, {query: "select 1", key: "id"});
Object.assign(s.model, {mrid: "t/m_packed/1", checkRange: false}); s.sink.table = "s.t_packed";
emit("t_packed", s);
s = A.defaultState(); s.recipe = "dlh"; Object.assign(s, {name: "t_dlh", owner: "u"});
Object.assign(s.source, {mode: "table", table: "s.src", key: "id"}); Object.assign(s.model, {mrid: "t/b/1", features: "a b"});
s.sink.table = "s.t_dlh"; s.sink.mode = "append"; emit("t_dlh", s);
s = A.defaultState(); s.recipe = "multi"; Object.assign(s, {name: "t_multi", owner: "u"});
Object.assign(s.source, {query: "select 1", key: "id"});
s.multi.models = [{name: "m1", mrid: "t/m1/1", score: "s1", features: "a, c", cats: ["c"], keep: {}},
                  {name: "m2", mrid: "t/m2/1", score: "s2", features: "b, c", cats: ["c"], keep: {}}];
s.sink.table = "s.t_multi"; emit("t_multi", s);
s.name = "t_multi_packed"; s.sink.table = "s.t_multi_packed";
s.multi.models = [{name: "m1", mrid: "t/m1/2", score: "s1", packed: true, features: "", cats: [], keep: {}},
                  {name: "m2", mrid: "t/m2/2", score: "s2", packed: true, features: "", cats: [], keep: {}},
                  {name: "m3", mrid: "t/m3/1", score: "s3", packed: false, features: "b, c", cats: ["c"], keep: {}}];
emit("t_multi_packed", s);
s = A.defaultState(); s.recipe = "calib"; Object.assign(s, {name: "t_calib", owner: "u"});
Object.assign(s.calib, {devQuery: "select 1", applyQuery: "select 2", applyKey: "id", score: "p", target: "y", output: "p_cal",
  presets: [{name: "", segs: [{field: "seg", values: "1\n2, 3", numeric: true}], duration: 2, gap: 1, rawOffsets: null}]});
s.sink.table = "s.t_calib"; s.report.table = "s.t_calib_report"; emit("t_calib", s);
// как в ноутбуке: два поля сегментов, 4 когорты с пропуском 1, когорта из даты, пустой факт = 0
s.name = "t_calib_window"; s.sink.table = "s.t_calib_window"; s.report.table = "s.t_calib_window_report";
const dbl = {field: "double", values: "0\n1\n2", numeric: true};
Object.assign(s.calib, {presets: [{name: "", segs: [{field: "product_cd", values: "CLN\nREF\nMTG, MTF, MTB, MTP", numeric: false}, dbl],
                                   duration: 4, gap: 1, rawOffsets: null}],
  cohortMode: "date", cohortDate: "due_dt", cohortTrunc: "month", targetFill: true});
emit("t_calib_window", s);
// три пресета из ноутбука: у продуктов разная длина окна, результат один
s.name = "t_calib_presets"; s.sink.table = "s.t_calib_presets"; s.report.table = "s.t_calib_presets_report";
s.calib.uncal = "drop";
s.calib.presets = [["ccr", "CCR", 2], ["vkr", "VKR", 3], ["other", "CLN\nREF\nMTG, MTF, MTB, MTP", 4]].map(x => (
  {name: x[0], segs: [{field: "product_cd", values: x[1], numeric: false}, dbl], duration: x[2], gap: 1, rawOffsets: null}));
emit("t_calib_presets", s);
for (const f of fs.readdirSync(path.join(work, "in"))) {
  emit("import_" + f.replace(".json", ""), A.fromConfig(JSON.parse(fs.readFileSync(path.join(work, "in", f), "utf8"))));
}
// колонки, вставленные списком, и типы с размером
const pasted = {
  ddl: "create table s.x (\n  customer_rk bigint not null,\n  segment_cd character varying(5) default 'A',\n  score numeric(18, 6), -- скор\n  upd timestamp without time zone\n) distributed by (customer_rk);",
  lines: "id bigint,\nreport_dt date,\nsegment_cd varchar(5)",
  tabs: "column_name\tdata_type\nparty_rk\tbigint\nproduct_cd\tcharacter varying",
  names: "client_id, report_dt, product_cd",
  junk: "select 1 from 2",
};
const parsed = {};
for (const k of Object.keys(pasted)) {
  const r = A.parseColumns(pasted[k]);
  parsed[k] = {columns: r.columns.map(c => [c.name, c.type, c.guessed]), skipped: r.skipped.length};
}
s = A.defaultState();
Object.assign(s, {name: "t_types", owner: "u"}); Object.assign(s.source, {query: "select 1", key: "id"});
Object.assign(s.model, {mrid: "t/m/1"}); s.sink.table = "s.t_types";
s.sink.columns = A.parseColumns(pasted.lines).columns.concat([{name: "amount", type: "numeric(18,4)"}]);
emit("t_types", s);
s.sink.columns.push({name: "broken", type: "varchar(5"});
report.bad_type = A.validate(s).map(p => p.text);
fs.writeFileSync(path.join(work, "parsed.json"), JSON.stringify({parsed, bad_type: report.bad_type}));
delete report.bad_type;
console.log(JSON.stringify(report));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="нужен node")
def test_form_logic_builds_valid_and_lossless_configs(tmp_path):
    (tmp_path / "in").mkdir()
    for ex in EXAMPLES:
        raw = yaml.safe_load((ex / "pipeline.yaml").read_text(encoding="utf-8"))
        (tmp_path / "in" / f"{ex.name}.json").write_text(json.dumps(raw, default=str), encoding="utf-8")
    page = tmp_path / "constructor.html"
    page.write_text(build_constructor(), encoding="utf-8")
    script = tmp_path / "harness.js"
    script.write_text(HARNESS, encoding="utf-8")
    res = subprocess.run(["node", str(script), str(page), str(tmp_path)], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    report = json.loads(res.stdout)
    assert all(not problems for problems in report.values()), report

    for ex in EXAMPLES:
        out = tmp_path / "out" / f"import_{ex.name}"
        if (ex / "predictor.py").exists():
            shutil.copy(ex / "predictor.py", out / "predictor.py")
        assert load_pipeline(out).model_dump() == load_pipeline(ex).model_dump(), ex.name

    for name in ("t_single", "t_packed", "t_dlh", "t_multi", "t_multi_packed", "t_calib"):
        p = load_pipeline(tmp_path / "out" / name)
        assert p.dag_id == name
        render_dir(tmp_path / "out" / name)            # из конфига собирается DAG
    single, packed = load_pipeline(tmp_path / "out" / "t_single"), load_pipeline(tmp_path / "out" / "t_packed")
    assert not single.model.packed and single.model.features == ["a", "b", "c"]
    assert packed.model.packed and packed.model.features == [] and packed.model.score_range is None
    assert load_pipeline(tmp_path / "out" / "t_dlh").model.packed          # новый процесс на DLH — упакованная модель
    assert [m.packed for m in load_pipeline(tmp_path / "out" / "t_multi_packed").models] == [True, True, False]
    assert load_pipeline(tmp_path / "out" / "t_dlh").sinks[0].mode == "replace"

    # типы с размером доходят до конфига как есть
    assert load_pipeline(tmp_path / "out" / "t_types").sinks[0].columns == {
        "id": "bigint", "report_dt": "date", "segment_cd": "varchar(5)", "amount": "numeric(18,4)", "score": "numeric"}
    extra = json.loads((tmp_path / "parsed.json").read_text(encoding="utf-8"))
    parsed = extra["parsed"]
    assert parsed["ddl"]["columns"] == [
        ["customer_rk", "bigint", False], ["segment_cd", "varchar(5)", False], ["score", "numeric(18,6)", False],
        ["upd", "timestamp without time zone", False]]
    assert parsed["lines"]["columns"] == [["id", "bigint", False], ["report_dt", "date", False], ["segment_cd", "varchar(5)", False]]
    assert parsed["tabs"] == {"columns": [["party_rk", "bigint", False], ["product_cd", "varchar", False]], "skipped": 0}
    assert parsed["names"]["columns"] == [["client_id", "bigint", True], ["report_dt", "date", True], ["product_cd", "varchar", True]]
    assert parsed["junk"] == {"columns": [], "skipped": 1}
    assert any("varchar(5" in text for text in extra["bad_type"])
    assert load_pipeline(tmp_path / "out" / "t_calib").calibrate.segments == {"seg": [1, [2, 3]]}
    assert load_pipeline(tmp_path / "out" / "t_calib").calibrate.train_offsets == [-3, -2]
    window = load_pipeline(tmp_path / "out" / "t_calib_window").calibrate
    assert window.segments == {"product_cd": ["CLN", "REF", ["MTG", "MTF", "MTB", "MTP"]], "double": [0, 1, 2]}
    assert window.train_offsets == [-5, -4, -3, -2]
    assert window.cohort_from.column == "due_dt" and window.cohort_from.truncate == "month"
    assert window.target_fill == 0
    render_dir(tmp_path / "out" / "t_calib_window")
    presets = load_pipeline(tmp_path / "out" / "t_calib_presets").calibrate
    assert presets.segments is None and presets.uncalibrated == "drop"
    assert [(p["name"], p["train_offsets"]) for p in presets.resolved_presets] == [
        ("ccr", [-3, -2]), ("vkr", [-4, -3, -2]), ("other", [-5, -4, -3, -2])]
    assert presets.resolved_presets[2]["segments"] == {
        "product_cd": ["CLN", "REF", ["MTG", "MTF", "MTB", "MTP"]], "double": [0, 1, 2]}
    render_dir(tmp_path / "out" / "t_calib_presets")


@pytest.mark.skipif(shutil.which("node") is None, reason="нужен node")
@pytest.mark.parametrize("script_id", ["team-presets", "logic", "ui"])
def test_page_scripts_parse(tmp_path, script_id):
    html = build_constructor(PRESETS)
    a = html.index(f'<script id="{script_id}">')
    code = html[html.index(">", a) + 1:html.index("</script>", a)]
    path = tmp_path / f"{script_id}.js"
    path.write_text(code, encoding="utf-8")
    res = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr
