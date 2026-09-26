# scoring-kit

Генератор Airflow DAG'ов для батч-скоринга: `pipeline.yaml` + `predictor.py` превращаются в
самодостаточный `.py` DAG на операторах платформы (`airflow_provider_greenplum`,
`airflow_provider_inference`). Замена UI-конструктора пайплайнов.

```
wait_source ─► read_source ─► inference ─► write_<table>: stg ─► load ─► harmonize ─► actualize
 (сенсор)      (Greenplum +     (Batch-         (df → *_stg)  (truncate+insert
               проверки)      Inference)                     в транзакции)
```

Чем отличается от старого инструмента:

| | Старый инструмент | scoring-kit |
|---|---|---|
| Где живёт код | текстовые поля UI | git: ревью, история, откат |
| Типы фичей | восстанавливаются из csv в каждом батче, `make_lgbm_ready` в каждом predict | контракт в yaml, приведение одинаковое для всех батчей |
| Проверки | самописные шаги (`test_df`) | встроены: пустая выборка, фичи, уникальность ключа, скор без NaN и в [0, 1] |
| replace | drop/create: таблица пустая и без грантов, пока идёт запись | stg + `truncate`/`insert` в одной транзакции |
| Шагов/подов | 6 | 3 тяжёлых + SQL |
| Отладка | кнопка в UI | `scoring debug` локально на том же коде |

## Установка

```bash
pip install "scoring-kit @ git+https://<gitlab>/<group>/scoring-kit.git"
```

## Процесс

### 1. Заготовка

```bash
scoring new pipelines/my_model      # pipeline.yaml + predictor.py
```

### 2. predictor.py

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

Правила (в под передаётся только исходник класса):
- на верхнем уровне файла — только `import` и класс; константы делайте атрибутами класса, функции — методами;
- импорты уровня модуля генератор сам копирует в каждый метод;
- `predict` получает батч (`batch_size` строк) и возвращает DataFrame той же длины с колонкой скора;
- имена `features`, `cat_features`, `num_dtype`, `cat_as`, `score_column`, `score_range` заняты
  контрактом (доступны как `self.features` и т.д.).

### 3. Локальная отладка

Выборку выгрузите из Greenplum в csv (например, `select ... limit 100000`), модель — из
Model Registry.

```bash
scoring debug pipelines/my_model --data sample.csv --model model.pkl
```

Прогоняет `read_source` → `inference` (по батчам, как оператор) → подготовку записи. Исполняется
ровно тот код, который уйдёт в Airflow. В `debug_out/` — сгенерированный DAG, файлы каждого шага,
`to_write.csv` и SQL загрузки.

### 4. Теневой прогон

Новый DAG пишет в `<table>_shadow`, старый продолжает работать.

```sql
-- один раз: теневая таблица с той же структурой, что и боевая
create table usr_coll.my_scores_fresh_shadow (like usr_coll.my_scores_fresh);
```

Для новой таблицы, которой ещё нет, DDL печатает `scoring ddl pipelines/my_model [--shadow]`.

```bash
scoring render pipelines/my_model --shadow -o build/dags   # -> build/dags/my_model_shadow.py
```

Публикация: `mlc` выкладывает папку, а `--prefixes` ограничивает набор файлов. Вывод `scoring render`
(`build/dags/<dag_id>.py`) ложится в ту структуру, которую ждёт инстанс (`dags/<файл>.py`):

```bash
# сначала посмотреть, что уйдёт (ничего не публикует)
mlc airflow publish coll-models-test -p <проект> -i build --prefixes dags/my_model_shadow.py --check
# затем то же без --check
mlc airflow publish coll-models-test -p <проект> -i build --prefixes dags/my_model_shadow.py
```

После прогона обоих DAG'ов сверка:

```bash
scoring compare-sql pipelines/my_model      # печатает SQL; ключ — source.checks.unique_key
```

Критерий приёмки: `only_in_prod = only_in_shadow = 0`, `n_diff_over_tol = 0`.

### 5. Переключение

1. `scoring render pipelines/my_model -o build/dags` → опубликовать боевой DAG.
2. Выключить (pause) старый DAG, после 2–3 успешных прогонов удалить его и `_shadow`-DAG/таблицу.

## Справочник pipeline.yaml

Полный пример — [examples/demo_scoring/pipeline.yaml](examples/demo_scoring/pipeline.yaml).

| Поле | По умолчанию | Смысл |
|---|---|---|
| `dag_id`, `owner`, `gp_service` | — | обязательные |
| `schedule`, `timezone` | `null`, `UTC` | cron и его часовой пояс |
| `time_limit`, `retries`, `retry_delay_minutes` | `7d`, `1`, `10` | для всех тасков; сенсор не перезапускается |
| `wait_for.tables`, `.timeout_seconds` | — , `83000` | сенсор актуализации; без блока сенсора нет |
| `source.query` | — | SQL выборки |
| `source.checks.min_rows`, `.unique_key` | `1`, `[]` | падение до скоринга, если выборка пустая или ключ не уникален |
| `model.mrid`, `.image`, `.requirements` | — | как в BatchInferenceOperator |
| `model.features` | — | фичи **в порядке обучения** |
| `model.cat_features` | `[]` | подмножество features |
| `model.num_dtype` | `float64` | `float32`, если так обучали (`make_lgbm_ready`) |
| `model.cat_as` | `category` | `category` — LightGBM, `str` — CatBoost (пропуск → `"nan"`) |
| `model.score_column`, `.score_range` | `score`, `[0, 1]` | `score_range: null` — не проверять диапазон |
| `sinks[].table`, `.columns` | — | `schema.table` и колонки с типами Greenplum в нужном порядке |
| `sinks[].mode` | `replace` | `replace` (truncate+insert) \| `append` |
| `sinks[].add_scored_at`, `.add_model_version` | `false` | служебные колонки `scored_at`, `model_mrid` |
| `sinks[].harmonize`, `.actualize` | `true` | `tcs_harmonize_grants` / `ulabs_actualize` после загрузки |
| `*.flavor` | `16cpu-256ram` | ресурсы пода для read / inference / write |

Приёмников может быть несколько: например, `*_fresh` в режиме `replace` и `*_hist` в режиме
`append` с `add_scored_at`.

## Команды

| | |
|---|---|
| `scoring new DIR` | заготовка пайплайна |
| `scoring validate DIR...` | проверка конфига и predictor.py (для CI) |
| `scoring render DIR... [-o build/dags] [--shadow]` | генерация DAG-файлов |
| `scoring debug DIR --data F --model M [--limit N]` | локальный прогон |
| `scoring ddl DIR [--shadow]` | `create table` приёмников |
| `scoring compare-sql DIR [--key a,b] [--tol 1e-9]` | SQL сверки прода с тенью |

## Разведка платформы

[tools/introspect_dag.py](tools/introspect_dag.py) печатает в лог версии библиотек, исходники
`airflow_provider_*` и версию Greenplum. Только чтение. Опубликовать, запустить вручную,
прочитать логи тасков `introspect` и `gp_version`.

## Что известно о платформе (по исходникам провайдеров с инстанса)

- **BatchInferenceOperator** читает вход только `pd.read_csv(f, chunksize=batch_size)`, поэтому
  между тасками всегда csv, а типы колонок определяются в каждом батче заново. Именно для этого
  нужен контракт `features`/`cat_features`.
- Из класса предиктора оператор берёт `inspect.getsource`, вырезает подстроку `BasePredictor`,
  переименовывает класс в `Predictor` и кладёт в `predict.py` без имён уровня модуля. Поэтому
  импорты должны быть внутри методов (генератор делает это сам), а `scoring debug` запускает класс
  ровно в таком виде.
- Job инференса живёт не дольше **2 часов** (`time_limit="2h"` зашит в оператор), `requirements`
  ставятся через `pip install` при каждом запуске.
- `DataframeToGreenplumOperator` в `mode="dal"` игнорирует `columns_types`: типы колонок
  определяет `dal.put_df` по dtype датафрейма. Поэтому `scoring-kit` приводит типы сам, а даты
  отдаёт как `datetime.date`.
- `GreenplumExecuteOperator` в dal-режиме выполняет `dal.execute(query)`; поддержка нескольких
  операторов (`begin; ...; commit;`) в одном вызове проверяется первым теневым прогоном.
- Airflow 2.10.5, Python 3.11, pandas 2.1.4 в подах тасков; Greenplum 6.27 (PostgreSQL 9.4).

## Открытые вопросы

- Удаляет ли `mlc airflow publish` с инстанса DAG'и, которых нет в публикуемом наборе (с `--prefixes` и без).
- Работает ли `begin; truncate; insert; commit;` одним вызовом `dal.execute`.

## Разработка

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

Тесты исполняют сгенерированные DAG'и на заглушках Airflow ([scoring_kit/stubs.py](scoring_kit/stubs.py))
и сверяют скоры со старым кодом `make_lgbm_ready` бит в бит.
