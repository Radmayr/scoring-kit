// Имитация GitLab для проверки клиента конструктора: отправка процесса, проверки, слияние, выкладка.
// Запуск: node gitlab_flow.js <constructor.html>  ->  JSON с результатами сценариев.
const fs = require("fs"), vm = require("vm");
const src = fs.readFileSync(process.argv[2], "utf8");
const grab = id => { const a = src.indexOf('<script id="' + id + '">'); return src.slice(src.indexOf(">", a) + 1, src.indexOf("</script>", a)); };
const ctx = {};
vm.createContext(ctx);
vm.runInContext(grab("team-presets") + "\n" + grab("logic") +
  "\nthis.API={gitlabClient,trackStatus,gitlabConfig,defaultState,buildYaml};this.TEAM=TEAM;", ctx);
const A = ctx.API;
ctx.TEAM.gitlab = {url: "https://gitlab.example/", project: "team/scoring-pipelines", branch: "main"};
const cfg = A.gitlabConfig();

// Сервер: один проект, файлы основной ветки, merge request'ы и пайплайны с заданиями.
function fakeGitlab(options) {
  options = options || {};
  const state = {files: Object.assign({"pipelines/old_one/pipeline.yaml": "dag_id: old_one\n"}, options.files),
    mrs: {}, pipelines: {}, jobs: {}, calls: [], nextId: 100, approvers: options.approvers || ["boss"]};
  const user = options.user || "i.ivanov", prefix = "/api/v4/projects/" + encodeURIComponent("team/scoring-pipelines");
  const reply = (status, body) => Promise.resolve({ok: status < 300, status, text: () => Promise.resolve(typeof body === "string" ? body : JSON.stringify(body))});
  function mainPipeline(sha) {
    const id = state.nextId++, jobs = [
      {id: state.nextId++, name: "check-pipelines", status: "running", web_url: "job/check"},
      {id: state.nextId++, name: "deploy-test", status: "created", web_url: "job/test"},
      {id: state.nextId++, name: "deploy-prod", status: "created", web_url: "job/prod"}];
    state.pipelines[sha] = {id, status: "running", web_url: "pipeline/" + id, sha};
    state.jobs[id] = jobs;
    return state.pipelines[sha];
  }
  const server = {
    state,
    fetch(url, opt) {
      const u = new URL(url), path = u.pathname, body = opt.body ? JSON.parse(opt.body) : null;
      state.calls.push(opt.method + " " + decodeURIComponent(path.replace(prefix, "")));
      if (options.offline) return Promise.reject(new Error("network"));
      if (opt.headers["PRIVATE-TOKEN"] !== "good") return reply(401, {message: "401 Unauthorized"});
      if (path === "/api/v4/user") return reply(200, {username: user});
      if (!path.startsWith(prefix)) return reply(404, {message: "404 Project Not Found"});
      const rest = decodeURIComponent(path.slice(prefix.length));
      let m;
      if (rest === "/repository/tree") return reply(200, [...new Set(Object.keys(state.files).map(f => f.split("/")[1]))].map(name => ({name, type: "tree"})).concat([{name: "README.md", type: "blob"}]));
      if ((m = rest.match(/^\/repository\/files\/(.+)\/raw$/))) return m[1] in state.files ? reply(200, state.files[m[1]]) : reply(404, {message: "404 File Not Found"});
      if (rest === "/repository/commits" && opt.method === "POST") {
        const action = body.actions[0], exists = action.file_path in state.files;
        if ((action.action === "create") === exists) return reply(400, {message: "A file with this name " + (exists ? "already exists" : "doesn't exist")});
        state.branches = state.branches || {};
        state.branches[body.branch] = {file: action.file_path, content: action.content, from: body.start_branch};
        return reply(201, {id: "sha-" + body.branch});
      }
      if (rest === "/merge_requests" && opt.method === "POST") {
        const iid = Object.keys(state.mrs).length + 1;
        state.mrs[iid] = {iid, web_url: "mr/" + iid, state: "opened", source_branch: body.source_branch, title: body.title,
          description: body.description, head_pipeline: null, merge_when_pipeline_succeeds: false, sha: "sha-" + body.source_branch};
        return reply(201, state.mrs[iid]);
      }
      if ((m = rest.match(/^\/merge_requests\/(\d+)$/))) return reply(200, state.mrs[m[1]]);
      if ((m = rest.match(/^\/merge_requests\/(\d+)\/merge$/))) {
        const mr = state.mrs[m[1]];
        if (options.cannotMerge) return reply(403, {message: "403 Forbidden"});
        if (!mr.head_pipeline) return reply(405, {message: "405 Method Not Allowed"});
        mr.merge_when_pipeline_succeeds = true;
        if (mr.head_pipeline.status === "success") server.merge(m[1]);
        return reply(200, mr);
      }
      if (rest === "/pipelines") { const p = state.pipelines[u.searchParams.get("sha")]; return reply(200, p ? [p] : []); }
      if ((m = rest.match(/^\/pipelines\/(\d+)\/jobs$/))) return reply(200, state.jobs[m[1]]);
      if ((m = rest.match(/^\/jobs\/(\d+)\/play$/))) {
        const job = Object.values(state.jobs).flat().find(j => j.id === Number(m[1]));
        if (job.status !== "manual") return reply(400, {message: "Unplayable Job"});
        // в настоящем CI это делает задание deploy-prod: scoring check-approver
        job.status = job.name === "deploy-prod" && !state.approvers.includes(user) ? "failed" : "success";
        return reply(200, job);
      }
      return reply(404, {message: "404 Not Found: " + rest});
    },
    // --- события на стороне GitLab
    startChecks(iid) { state.mrs[iid].head_pipeline = {id: state.nextId++, status: "running", web_url: "pipeline/mr" + iid}; },
    finishChecks(iid, ok) {
      const mr = state.mrs[iid];
      mr.head_pipeline.status = ok ? "success" : "failed";
      if (ok && mr.merge_when_pipeline_succeeds) server.merge(iid);
    },
    merge(iid) {
      const mr = state.mrs[iid], branch = state.branches[mr.source_branch];
      state.files[branch.file] = branch.content;
      mr.state = "merged"; mr.merge_commit_sha = "merge-" + iid;
    },
    startMain(iid) { return mainPipeline(state.mrs[iid].merge_commit_sha); },
    finishMain(iid, ok, autoTest) {
      const p = state.pipelines[state.mrs[iid].merge_commit_sha], jobs = state.jobs[p.id];
      jobs[0].status = ok ? "success" : "failed";
      jobs[1].status = !ok ? "skipped" : autoTest === false ? "manual" : "success";
      jobs[2].status = ok ? "manual" : "skipped";
      p.status = ok ? "success" : "failed";
    },
  };
  return server;
}

const say = s => ({phase: s.phase, tone: s.tone, text: s.text, action: s.action && s.action.label});
async function submit(server, name, target, token, folder) {
  const client = A.gitlabClient(cfg, token || "good", server.fetch);
  const st = A.defaultState();
  Object.assign(st, {name, owner: "u", stage: target === "prod" ? "prod" : "test"});
  Object.assign(st.source, {query: "select 1", key: "id"}); st.model.mrid = "t/m/1"; st.sink.table = "s." + name;
  const mr = await client.submit({name, folder, yaml: A.buildYaml(st), stamp: "20261013120000", title: "x: " + name, description: "d"});
  return {client, tracking: {iid: mr.iid, url: mr.url, name, target, dag: name + (target === "test" ? "_shadow" : "")}, mr, yaml: A.buildYaml(st)};
}

(async () => {
  const out = {};

  // 1. Новый процесс на тест: проверки -> автослияние -> выкладка на тестовый инстанс
  {
    const g = fakeGitlab(), steps = [];
    const {client, tracking, mr, yaml} = await submit(g, "new_one", "test");
    const step = async () => steps.push(say(await A.trackStatus(client, cfg, tracking)));
    await step();                       // проверок ещё нет
    g.startChecks(tracking.iid); await step();      // идут проверки, включено автослияние
    await step();
    g.finishChecks(tracking.iid, true); await step();   // слито, пайплайна main ещё нет
    g.startMain(tracking.iid); await step();
    g.finishMain(tracking.iid, true); await step();
    out.test_flow = {mr, steps, committed: g.state.files["pipelines/new_one/pipeline.yaml"] === yaml,
      shadow: /^shadow: true/m.test(yaml), branch: Object.keys(g.state.branches)[0], action: g.state.branches[mr.branch].from,
      merges: g.state.calls.filter(c => c.endsWith("/merge")).length};
  }

  // 2. Сразу на прод: после слияния ждёт публикации; её запускает участник с правом
  {
    const g = fakeGitlab({user: "boss"}), steps = [];
    const {client, tracking, yaml} = await submit(g, "prod_one", "prod");
    const step = async () => { const s = await A.trackStatus(client, cfg, tracking); steps.push(say(s)); return s; };
    g.startChecks(tracking.iid); await step(); g.finishChecks(tracking.iid, true); await step();
    g.startMain(tracking.iid); g.finishMain(tracking.iid, true);
    const waiting = await step();
    await client.play(waiting.action.jobId);
    await step();
    out.prod_flow = {steps, shadow: /^shadow: true/m.test(yaml)};
  }

  // 3. На прод нажал участник без права: задание падает на проверке approvers.txt
  {
    const g = fakeGitlab({user: "i.ivanov"});
    const {client, tracking} = await submit(g, "prod_two", "prod");
    g.startChecks(tracking.iid); await A.trackStatus(client, cfg, tracking); g.finishChecks(tracking.iid, true);
    g.startMain(tracking.iid); g.finishMain(tracking.iid, true);
    const waiting = await A.trackStatus(client, cfg, tracking);
    await client.play(waiting.action.jobId);
    out.prod_denied = say(await A.trackStatus(client, cfg, tracking));
  }

  // 4. Ошибка в конфиге: проверки красные, ничего не слито
  {
    const g = fakeGitlab();
    const {client, tracking} = await submit(g, "bad_one", "test");
    g.startChecks(tracking.iid); await A.trackStatus(client, cfg, tracking); g.finishChecks(tracking.iid, false);
    out.checks_failed = {status: say(await A.trackStatus(client, cfg, tracking)), merged: "pipelines/bad_one/pipeline.yaml" in g.state.files};
  }

  // 5. Нет права сливать: конструктор просит участника с правами
  {
    const g = fakeGitlab({cannotMerge: true});
    const {client, tracking} = await submit(g, "need_merge", "test");
    g.startChecks(tracking.iid);
    out.cannot_merge = say(await A.trackStatus(client, cfg, tracking));
    g.finishChecks(tracking.iid, true); g.merge(tracking.iid); g.startMain(tracking.iid); g.finishMain(tracking.iid, true);
    out.cannot_merge_after = say(await A.trackStatus(client, cfg, tracking));
  }

  // 6. Правка существующего процесса -> update; тот же конфиг -> отказ; неверный токен; нет сети
  {
    const g = fakeGitlab();
    // процесс открыт из папки old_one, а DAG в форме назван иначе: правка идёт в ту же папку
    const {mr} = await submit(g, "old_one_v2", "test", "good", "old_one");
    out.update_existing = {existed: mr.existed, file: mr.file, folders: [...new Set(Object.keys(g.state.branches).map(b => g.state.branches[b].file))]};
    // новый процесс с именем, которое уже занято: отказ, чужой процесс не затирается
    out.name_taken = await submit(fakeGitlab(), "old_one", "test").then(() => "sent", e => e.message);
    const same = fakeGitlab();
    const st = A.defaultState(); Object.assign(st, {name: "old_one", owner: "u"});
    Object.assign(st.source, {query: "select 1", key: "id"}); st.model.mrid = "t/m/1"; st.sink.table = "s.old_one";
    same.state.files["pipelines/old_one/pipeline.yaml"] = A.buildYaml(st);
    out.same_config = await submit(same, "old_one", "test", "good", "old_one").then(() => "sent", e => e.message);
    out.bad_token = await submit(fakeGitlab(), "x1", "test", "wrong").then(() => "sent", e => e.message);
    out.offline = await submit(fakeGitlab({offline: true}), "x2", "test").then(() => "sent", e => e.message);
    out.list = await A.gitlabClient(cfg, "good", fakeGitlab({files: {"pipelines/b/pipeline.yaml": "x", "pipelines/a/pipeline.yaml": "y"}}).fetch).listPipelines();
  }

  // 7. Выкладка на тест ещё ручная (старый CI): конструктор предлагает кнопку
  {
    const g = fakeGitlab();
    const {client, tracking} = await submit(g, "manual_test", "test");
    g.startChecks(tracking.iid); await A.trackStatus(client, cfg, tracking); g.finishChecks(tracking.iid, true);
    g.startMain(tracking.iid); g.finishMain(tracking.iid, true, false);
    const waiting = await A.trackStatus(client, cfg, tracking);
    await client.play(waiting.action.jobId);
    out.manual_test = [say(waiting), say(await A.trackStatus(client, cfg, tracking))];
  }
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1); });
