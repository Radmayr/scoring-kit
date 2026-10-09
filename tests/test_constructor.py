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
vm.runInContext(grab("team-presets") + "\n" + grab("logic") + "\nthis.API={defaultState,buildYaml,validate,fromConfig};", ctx);
const A = ctx.API, report = {};
function emit(name, st) {
  const dir = path.join(work, "out", name); fs.mkdirSync(dir, {recursive: true});
  fs.writeFileSync(path.join(dir, "pipeline.yaml"), A.buildYaml(st));
  report[name] = A.validate(st).map(p => p.text);
}
let s = A.defaultState();
Object.assign(s, {name: "t_single", owner: "u"}); Object.assign(s.source, {query: "select 1", key: "id"});
Object.assign(s.model, {mrid: "t/m/1", features: "a, b, c", cats: ["c"]}); s.sink.table = "s.t_single";
emit("t_single", s);
s = A.defaultState(); s.recipe = "dlh"; Object.assign(s, {name: "t_dlh", owner: "u"});
Object.assign(s.source, {mode: "table", table: "s.src", key: "id"}); Object.assign(s.model, {mrid: "t/b/1", features: "a b"});
s.sink.table = "s.t_dlh"; s.sink.mode = "append"; emit("t_dlh", s);
s = A.defaultState(); s.recipe = "multi"; Object.assign(s, {name: "t_multi", owner: "u"});
Object.assign(s.source, {query: "select 1", key: "id"});
s.multi.models = [{name: "m1", mrid: "t/m1/1", score: "s1", features: "a, c", cats: ["c"], keep: {}},
                  {name: "m2", mrid: "t/m2/1", score: "s2", features: "b, c", cats: ["c"], keep: {}}];
s.sink.table = "s.t_multi"; emit("t_multi", s);
s = A.defaultState(); s.recipe = "calib"; Object.assign(s, {name: "t_calib", owner: "u"});
Object.assign(s.calib, {devQuery: "select 1", applyQuery: "select 2", applyKey: "id", score: "p", target: "y", output: "p_cal",
  segField: "seg", segValues: "1\n2, 3", segNumeric: true});
s.sink.table = "s.t_calib"; s.report.table = "s.t_calib_report"; emit("t_calib", s);
for (const f of fs.readdirSync(path.join(work, "in"))) {
  emit("import_" + f.replace(".json", ""), A.fromConfig(JSON.parse(fs.readFileSync(path.join(work, "in", f), "utf8"))));
}
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

    for name in ("t_single", "t_dlh", "t_multi", "t_calib"):
        p = load_pipeline(tmp_path / "out" / name)
        assert p.dag_id == name
    assert load_pipeline(tmp_path / "out" / "t_dlh").sinks[0].mode == "replace"
    assert load_pipeline(tmp_path / "out" / "t_calib").calibrate.segments == {"seg": [1, [2, 3]]}
