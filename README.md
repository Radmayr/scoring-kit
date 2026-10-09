# scoring-kit

Фабрика Airflow DAG'ов для батч-скоринга. Процесс описывается в `pipeline.yaml` (что читать, какие
модели применить, куда писать, когда), фабрика проверяет описание и генерирует самодостаточный
`.py` DAG на операторах платформы. Airflow знать не нужно. Замена UI-конструктора пайплайнов.

Архитектура и решения — [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); схема блоками и план внедрения — [docs/scheme.html](docs/scheme.html) (открыть в браузере).

## Сценарии

| Рецепт | Когда | Пример |
|---|---|---|
| `single_model` | одна модель → один скор | [demo_scoring](examples/demo_scoring/pipeline.yaml) (свой предиктор), [demo_scoring_dlh](examples/demo_scoring_dlh/pipeline.yaml) (Spark в DLH) |
| `multi_model` | несколько моделей → несколько скоров → одна витрина | [demo_multi](examples/demo_multi/pipeline.yaml) |
| `fit_apply` | обучение на лету: калибровка на dev, применение к apply | [demo_calibration](examples/demo_calibration/pipeline.yaml) |

Два движка (`engine`):

| | `gp` (по умолчанию) | `dlh` |
|---|---|---|
| Данные | Greenplum | таблицы DLH (Iceberg) |
| Исполнение | `BatchInferenceOperator` (ML Core job, pandas по батчам) | `DlhBatchInferenceOperator` (Spark) |
| Рецепты | все | `single_model` |
| Модель в реестре | как есть (joblib/pickle) | бандл `scoring bundle` (см. ниже) |

```
gp:   wait_source → read_source (+проверки) → inference → write_<table>: stg → load → harmonize → actualize
dlh:  wait_source → prepare_source (если query) → check_source → inference (Spark) → check_result
```

Чем отличается от старого инструмента:

| | Старый инструмент | scoring-kit |
|---|---|---|
| Где живёт код | текстовые поля UI | git: ревью, история, откат |
| Типы фичей | восстанавливаются из csv в каждом батче, `make_lgbm_ready` в каждом predict | контракт в yaml, приведение одинаковое для всех батчей |
| Проверки | самописные шаги, не всегда блокируют запись | встроены: пустая выборка, фичи, ключ, колонки приёмника, скор без пропусков и в диапазоне |
| replace | drop/create: таблица пустая и без грантов, пока идёт запись | stg + `truncate`/`insert` в одной транзакции |
| Несколько моделей | свой predictor на каждый процесс | `models:` в yaml, без кода |
| Калибровка | через чужую модель в mrid, склейка dev/apply флагом | рецепт `fit_apply`, отчёт по ячейкам в историю |
| Алерты | нет | `alerts:` → сообщение в мессенджер при падении |
| Отладка | кнопка в UI | `scoring debug` локально на том же коде |

## Установка

```bash
pip install "scoring-kit @ git+https://<gitlab>/<group>/scoring-kit.git@v0.5.0"
```

Всегда закрепляйте версию (тег `@v0.5.0`): фреймворк меняется, и незакреплённая установка сломает
чужие пайплайны при выходе новой версии.

## Конструктор: процесс без программирования

`scoring_kit/web/constructor.html` — одна страница без сервера. Коллега выбирает сценарий, заполняет
пять шагов формы (данные, модель, куда писать, когда), справа сразу видит схему процесса, замечания
по-русски и готовый `pipeline.yaml` с кнопкой «Скопировать». Время задаётся по Москве, в конфиг
уходит cron в UTC. Существующий процесс можно открыть: вставить его `pipeline.yaml`, поправить и
скопировать обратно. Всё, чего форма не показывает (свой `predictor.py`, теги, таймауты, точные
ресурсы подов), сохраняется без изменений.

Собрать страницу под команду:

```bash
scoring constructor --presets presets.yaml --pipelines pipelines/*/ -o constructor.html
```

`presets.yaml` — образы для выпадающих списков, сервис Greenplum, направления и ссылка на создание
файла в GitLab (формат — в `scoring_kit/constructor.py`). `--pipelines` добавляет в списки образы
работающих процессов. Готовый файл открывается в браузере или выкладывается на GitLab Pages.
Логика формы покрыта тестом: каждый пример, открытый в конструкторе, сохраняется идентично.

## Процесс

### 1. Конфиг

```bash
scoring new pipelines/my_model                          # одна модель, встроенный предиктор
scoring new pipelines/my_multi --recipe multi_model     # несколько моделей в одну витрину
scoring new pipelines/my_calib --recipe fit_apply       # калибровка / обучение на лету
scoring new pipelines/my_dlh --recipe dlh               # одна модель на DLH
scoring new pipelines/my_model --custom-predictor       # + свой predictor.py
```

Заготовка заполнена рабочими значениями-примерами; поля, которые надо поменять, помечены `TODO`.

**Подсказки в редакторе.** В VS Code поставьте расширение YAML (Red Hat) и один раз сгенерируйте
схему в корне репозитория процессов: `scoring schema -o pipeline.schema.json`. Дальше при наборе
`pipeline.yaml` редактор подсказывает поля и допустимые значения, показывает описание при наведении
и подчёркивает опечатки.

Общая шапка одинакова для всех рецептов:

```yaml
recipe: single_model                # single_model | multi_model | fit_apply
engine: gp                          # gp | dlh (только single_model)
dag_id: my_model
owner: i.ivanov
domain: collection                  # группировка в каталоге и тегах Airflow
schedule: "30 4 * * *"              # UTC: 04:30 UTC = 07:30 МСК
alerts:
  recipients: ["#канал-команды"]    # сообщение при падении любого таска
wait_for:
  tables: [schema.features_fresh]   # сенсор актуальности
```

### 2. Модель: встроенный предиктор или свой

Для бустингов код не нужен:

```yaml
model:
  mrid: tenant/model/1.0.0
  image: <образ>
  requirements: [numpy==1.26.0, pandas==2.1.1, lightgbm==3.3.5]
  output_kind: proba                # predict_proba()[:, 1]; predict — для регрессии
  features: [f1, f2, segment]       # в порядке обучения
  cat_features: [segment]
  num_dtype: float32                # если при обучении приводили к float32 (make_lgbm_ready)
```

Нестандартная логика — свой `predictor.py` (без `output_kind`):

```python
import joblib

from scoring_kit import BasePredictor  # в DAG заменится на airflow_provider_inference


class Predictor(BasePredictor):
    def setup(self, model_paths: list[str]):
        self.model = joblib.load(model_paths[0])      # пути в порядке model.mrid

    def predict(self, df):
        # фичи уже приведены к типам контракта; порядок колонок модели — self.features
        df["score"] = self.model.predict_proba(df[self.features])[:, 1]
        return df
```

Правила для `predictor.py` (в под передаётся только исходник класса): на верхнем уровне — только
`import` и класс; константы — атрибутами класса, функции — методами; `predict` получает батч и
возвращает DataFrame той же длины с колонкой скора.

### 3. Несколько моделей

```yaml
recipe: multi_model
model_defaults:                     # общее; модель может переопределить
  image: <образ>
  output_kind: proba
  cat_features: [segment]
models:
  - {name: d30, mrid: t/d30/1.0.0, score_column: score_30, features: [...]}
  - {name: d90, mrid: t/d90/1.0.0, score_column: score_90, features: [...]}
```

Модели с одинаковым окружением (`image`, `requirements`, `flavor`, `batch_size`) считаются одним
job'ом: данные читаются один раз. Разные окружения — параллельные job'ы и слияние по
`source.checks.unique_key`. Падение любой модели валит процесс: витрина не пишется частично.

### 4. Калибровка (fit_apply)

```yaml
recipe: fit_apply
dev:   {query: "select ... from dev_calib_in"}
apply: {query: "select ... from apply_calib_in", key: [account_rk, due_dt]}
job:   {image: <образ>, requirements: [pandas==2.1.4, scikit-learn==1.3.2]}
calibrate:
  method: isotonic                  # isotonic | sigmoid
  score: pre_score
  target: pre_target
  output: pre_score_calib
  segments: {product_cd: [CCR, CUR, [MTG, MTF]]}   # список в списке = группа
  cohort_column: cohort
  train_offsets: [-3, -2]           # когорта на позиции i учится на i-3 и i-2
  uncalibrated: null                # null | copy_score
sinks:
  - {table: ..., columns: {..., pre_score_calib: numeric}}
  - {table: ..._report_hist, data: report, mode: append, add_scored_at: true, columns: {...}}
```

Отчёт (`data: report`): по каждой ячейке сегмент × когорта — объёмы, train-когорты, Brier до и
после калибровки, статус. Под запускается без модели (`BatchInferenceOperator` с `mrid=[]`).

### 5. DLH (engine: dlh)

`DlhBatchInferenceOperator` вызывает `model.predict(df)` в Spark-образе с другими версиями
библиотек (lightgbm 4.x, numpy 2, pandas 3). Поэтому в реестр кладётся **бандл**:

```bash
scoring bundle pipelines/my_model_dlh --model model.pkl -o bundle.pkl
```

Бандл хранит модель в переносимом виде (текст модели LightGBM / байты CatBoost), сам приводит типы
фичей по контракту из yaml и восстанавливается без scoring-kit в образе. Загрузите `bundle.pkl` в
Model Registry новой версией и укажите её mrid в `model.mrid`. Проверено: скоры бандла в
окружении lightgbm 4.7 / numpy 2 / pandas 3 совпадают с исходной моделью на lightgbm 3.3.5 бит в бит.

Во время работы оператор берёт фичи из бандла, но `features`/`cat_features` в yaml всё равно нужны:
из них собирается бандл (его можно пересобрать из исходной модели и yaml), по ним `check_source`
проверяет колонки источника до запуска Spark, и они видны в каталоге. **Изменили фичи в yaml —
пересоберите бандл и загрузите новую версию**, иначе проверка и модель разойдутся.

Источник — `source.table` (таблица DLH) или `source.query` + `dlh.staging_table` (запрос
материализуется в таблицу: оператор читает только таблицу целиком). Приёмник — один, `replace`
(`createOrReplace`); все колонки, кроме скора, переносятся из источника как есть, оператор
добавляет `processed_dttm`.

### 6. Локальная отладка

```bash
scoring debug pipelines/my_model --data sample.csv --model model.pkl
scoring debug pipelines/pd_model --data sample.csv --model d1=m1.pkl --model d2=m2.pkl ...
scoring debug pipelines/calib --data dev=dev.csv --data apply=apply.csv
scoring debug pipelines/my_model_dlh --data sample.csv --model bundle.pkl   # или исходная модель
```

Исполняется ровно тот код, который уйдёт в Airflow, по тем же правилам, что платформа: класс
предиктора в том виде, в котором его получает оператор, батчи `read_csv(chunksize)`, входы
монтируются как `executor_config`. В `debug_out/` — DAG, файлы каждого шага и то, что запишется.

### 7. Тень, сверка, переключение

Публикация — одной командой из корня репозитория процессов:

```bash
scoring publish --env test            # собрать набор, показать список, mlc --check, спросить «да»
scoring publish --env test --dry-run  # только собрать папку publish/ и показать список
```

Почему не руками: `mlc airflow publish` заменяет всё, что было опубликовано через mlc, содержимым
папки (DAG'и старого инструмента в `piper/` не затрагиваются), а тестовый и боевой инстансы пишут
в один Greenplum. `scoring publish` собирает правильный полный набор сам:

| | test | prod |
|---|---|---|
| процессы | теневые версии тех, что в `shadow.txt` | боевые версии всех, кроме тех, что в `shadow.txt` |
| ручные DAG'и | `dags_manual/test/` (или `dags_manual/*.py`) | `dags_manual/prod/` |
| черновики (`draft: true`) | не публикуются | не публикуются |

Инстансы и проект — в `environments.yaml` в корне репозитория процессов. Пустой набор не
публикуется никогда.

Целевые таблицы (в том числе теневые) создаются DAG'ом при первом запуске
(`create table if not exists` по `sinks[].columns`). После прогонов старого и теневого DAG'а:

```bash
scoring compare-sql pipelines/my_model      # SQL сверки по каждой колонке скора
```

Критерий приёмки: `only_in_prod = only_in_shadow = 0`, `n_diff_over_tol = 0`, `n_null_mismatch = 0`.

Переключение: поставить старый DAG на паузу → убрать процесс из `shadow.txt` →
`scoring publish --env prod` → вручную запустить и проверить → старый держать на паузе неделю
(откат = снять паузу).

Правка процесса: поменяли `pipeline.yaml` → `scoring validate` → `scoring publish --env test`.
Пересобирать и публиковать отдельные файлы не нужно.

## Справочник pipeline.yaml

| Поле | По умолчанию | Смысл |
|---|---|---|
| `recipe`, `engine` | `single_model`, `gp` | сценарий и движок |
| `dag_id`, `owner` | — | обязательные |
| `domain`, `tags`, `description` | — | каталог, теги Airflow, документация DAG'а |
| `schedule`, `timezone` | `null`, `UTC` | cron и его часовой пояс |
| `draft` | `false` | черновик: проверяется, но не публикуется |
| `alerts.recipients`, `.on_retry`, `.message` | —, `false`, `""` | уведомление при падении (`TiMeNotifier`) |
| `gp_service`, `gp_mode` | `vrcl`, `dal` | подключение к Greenplum |
| `time_limit`, `retries`, `retry_delay_minutes` | `7d`, `1`, `10` | для всех тасков; сенсор не перезапускается |
| `wait_for.tables`, `.timeout_seconds` | — , `83000` | сенсор актуальности |
| `source.query` / `source.table` | — | выборка (table — для `engine: dlh`) |
| `source.checks.min_rows`, `.unique_key` | `1`, `[]` | падение до скоринга |
| `model` / `models` / `model_defaults` | — | модель (single) / модели (multi) / общее для моделей |
| `model.mrid`, `.image`, `.requirements` | — | как в BatchInferenceOperator |
| `model.output_kind` | — | `proba` / `predict` — встроенный предиктор; без него — `predictor.py` |
| `model.features`, `.cat_features` | — | фичи **в порядке обучения**, категориальные из них |
| `model.num_dtype`, `.cat_as` | `float64`, `category` | `float32` как в `make_lgbm_ready`; `str` для CatBoost |
| `model.score_column`, `.score_range` | `score`, `[0, 1]` | `score_range: null` — не проверять диапазон |
| `dlh.image`, `.staging_table`, `.max_executors` | — | для `engine: dlh` |
| `dev`, `apply`, `job`, `calibrate` | — | для `recipe: fit_apply` |
| `sinks[].table`, `.columns` | — | `schema.table` и колонки с типами в нужном порядке |
| `sinks[].mode` | `replace` | `replace` (truncate+insert) \| `append` |
| `sinks[].data` | `result` | `report` — отчёт калибровки (`fit_apply`) |
| `sinks[].add_scored_at`, `.add_model_version` | `false` | служебные колонки `scored_at`, `model_mrid` |
| `sinks[].harmonize`, `.actualize` | `true` | гранты / актуализация после загрузки (`gp`) |
| `*.flavor` | `16cpu-256ram` | ресурсы пода |

## Команды

| | |
|---|---|
| `scoring constructor [--presets F] [--pipelines DIR...] [-o F]` | страница-конструктор pipeline.yaml для команды |
| `scoring new DIR [--recipe R] [--custom-predictor]` | заготовка процесса: single_model, multi_model, fit_apply, dlh |
| `scoring validate DIR...` | проверка конфига и predictor.py (для CI) |
| `scoring publish --env test\|prod [--dry-run] [--yes]` | собрать полный набор для инстанса и опубликовать |
| `scoring schema [-o pipeline.schema.json]` | схема для подсказок в редакторе |
| `scoring render DIR... [-o build/dags] [--shadow]` | генерация DAG-файлов |
| `scoring debug DIR --data F --model M [--limit N]` | локальный прогон |
| `scoring bundle DIR --model M -o bundle.pkl` | упаковка модели для `engine: dlh` |
| `scoring catalog DIR... [-o CATALOG.md]` | каталог: процессы, владельцы, модели, таблицы, зависимости |
| `scoring ddl DIR [--shadow]` | `create table` приёмников |
| `scoring compare-sql DIR [--key a,b] [--tol 1e-9]` | SQL сверки прода с тенью |

## Что известно о платформе (по исходникам провайдеров с инстанса)

- **BatchInferenceOperator** читает вход только `pd.read_csv(f, chunksize=batch_size)`; типы
  колонок определяются в каждом батче заново — для этого контракт `features`/`cat_features`.
  Класс предиктора передаётся исходником (вырезается `BasePredictor`, класс переименовывается в
  `Predictor`). Job живёт не дольше 2 часов, `requirements` ставятся при каждом запуске.
  `mrid=[]` допустим — так работает `fit_apply`.
- **DlhBatchInferenceOperator** (провайдер помечен как «в разработке»): одна модель; читает
  `source_table` целиком, пишет `createOrReplace` в `target_table`; у модели вызывает `predict`,
  список фичей берёт из `model.features`; без явного типа выход пишется как float32 (фабрика
  задаёт `DoubleType`); `max_wait` по умолчанию 1 час (фабрика — 3 часа). Сервисной учётке
  `dp_conn_dlh` нужен доступ ко всем колонкам источника.
- **DataframeToGreenplumOperator** в `mode="dal"` игнорирует `columns_types` — типы приводит фабрика.
- **GreenplumExecuteOperator** выполняет `begin; truncate; insert; commit;` одним вызовом
  (проверено теневым прогоном).
- **Алерты**: `airflow_provider_time.notifications.TiMeNotifier(message, recipients)`.
- **Публикация**: `mlc` синхронизирует свою часть инстанса целиком; старый инструмент — в `piper/`.
- Airflow 2.10.5, Python 3.11, pandas 2.1.4 в подах; Greenplum 6.27 (PostgreSQL 9.4).

Разведчик окружения — [tools/introspect_dag.py](tools/introspect_dag.py) (только чтение).

## Командная работа в GitLab

| Репозиторий | Что лежит | Кто меняет |
|---|---|---|
| `scoring-kit` (этот) | фреймворк | 1–2 человека, через merge request |
| `scoring-pipelines` | `pipelines/<имя>/pipeline.yaml` (+ `predictor.py`), `CATALOG.md` | все аналитики |

Пайплайны ставят фреймворк по закреплённой версии (`requirements.txt` в `scoring-pipelines`):
изменение фреймворка никогда не ломает боевые скоринги неожиданно.

**Перенос из GitHub в GitLab:** `git remote add gitlab <url> && git push gitlab main --tags`.

**Доступ к установке** (что разрешено политикой): SSH-ключ
(`git+ssh://git@<gitlab>/<group>/scoring-kit.git@v0.5.0`), deploy token с правом
`read_repository`, `CI_JOB_TOKEN` в CI или wheel во внутреннем pip-индексе
(`pip wheel . --no-deps`, затем `pip install scoring-kit==0.4.0`).

**Выпуск версии:** merge request → CI (тесты на версиях из подов и свежих) → поднять `version` в
`pyproject.toml` и `scoring_kit/__init__.py` → `git tag vX.Y.Z && git push --tags` → в
`scoring-pipelines` поднять тег отдельным merge request'ом.

## Разработка

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

Тесты исполняют сгенерированные DAG'и на заглушках Airflow ([scoring_kit/stubs.py](scoring_kit/stubs.py))
и сверяют результат с дословным кодом текущих процессов бит в бит: одна модель, четыре модели,
калибровка, DLH-бандл.
