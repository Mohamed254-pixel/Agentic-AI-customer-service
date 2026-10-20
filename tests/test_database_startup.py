"""Cloud-like startup: only tracked source and sample CSVs, no local database."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import duckdb
import pytest

from db import bootstrap
from scripts import build_db

ROOT = Path(__file__).resolve().parents[1]


def test_concurrent_sessions_build_only_once(tmp_path, monkeypatch):
    calls = []
    builder = bootstrap.build_database

    def counted(source, target):
        calls.append(target)
        return builder(source, target)

    monkeypatch.setattr(bootstrap, 'build_database', counted)
    database = tmp_path / 'processed/sample.duckdb'
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: bootstrap.ensure_sample_database(database), range(4)))
    assert len(calls) == 1
    assert results == [database] * 4
    with duckdb.connect(str(database), read_only=True) as con:
        assert len(con.execute('SHOW TABLES').fetchall()) == 6


def test_failed_build_keeps_existing_database(tmp_path, monkeypatch):
    database = tmp_path / 'sample.duckdb'
    with duckdb.connect(str(database)) as con:
        con.execute('CREATE TABLE original AS SELECT 42 AS value')
    connect = build_db.duckdb.connect

    def fail(*args, **kwargs):
        raise duckdb.IOException('Simulated startup failure')

    monkeypatch.setattr(build_db.duckdb, 'connect', fail)
    with pytest.raises(duckdb.IOException):
        build_db.build_database(ROOT / 'data/sample', database)
    with connect(str(database), read_only=True) as con:
        assert con.execute('SELECT value FROM original').fetchone() == (42,)
    assert list(tmp_path.iterdir()) == [database]


def test_fresh_cloud_copy_and_cli_from_another_directory(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    for name in ['agents', 'db', 'scripts', 'ui']:
        shutil.copytree(ROOT / name, project / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'data/sample', project / 'data/sample')
    shutil.copytree(ROOT / 'data/policies', project / 'data/policies')
    shutil.copy2(ROOT / 'main.py', project / 'main.py')
    database = project / 'data/processed/instacart_sample.duckdb'
    assert not database.exists()
    env = {**os.environ, 'DEMO_MODE': '1', 'OPENAI_API_KEY': '', 'PYTHONPATH': str(project)}
    code = '''
from pathlib import Path
from streamlit.testing.v1 import AppTest
from db.queries import DB
assert DB == Path.cwd() / 'data/processed/instacart_sample.duckdb'
assert not DB.exists()
app = AppTest.from_file(Path.cwd() / 'main.py').run(timeout=30)
assert not app.exception, app.exception
assert DB.is_file()
app.chat_input[0].set_value('What was in my latest order?').run(timeout=30)
assert not app.exception
assert '1405149' in app.chat_message[-1].markdown[0].value
first_build = DB.stat().st_mtime_ns
app.run(timeout=30)
assert DB.stat().st_mtime_ns == first_build
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=project, env=env,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    # Original command remains valid even outside the project working directory.
    cli = subprocess.run([sys.executable, str(project / 'scripts/build_db.py'), '--dataset', 'sample'],
                         cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert cli.returncode == 0, cli.stdout + cli.stderr
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute('SELECT COUNT(*) FROM orders').fetchone()[0] == 10000
        assert con.execute('SELECT COUNT(*) FROM products').fetchone()[0] == 12207
