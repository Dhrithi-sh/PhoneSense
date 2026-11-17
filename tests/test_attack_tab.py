"""T25: the attack page runs headless. Skipped unless streamlit is installed
(pip install -r requirements-ui.txt)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from phonesense import config  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "phonesense" / "ui" / "attack_app.py"


@pytest.fixture
def app(monkeypatch, tmp_path):
    # point the data paths at an empty folder so the page always uses toy reviews
    (tmp_path / "config.yaml").write_text(
        (APP.parents[2] / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("PHONESENSE_ROOT", str(tmp_path))
    config.get_config.cache_clear()
    yield AppTest.from_file(str(APP), default_timeout=120).run()
    monkeypatch.undo()
    config.get_config.cache_clear()


def test_page_renders_on_toy_data(app):
    assert not app.exception
    assert "Toy data" in app.warning[0].value
    assert len(app.metric) == 5


def test_controls_update_the_result(app):
    app.slider[0].set_value(0).run()
    assert not app.exception
    assert all(m.delta == "no change" for m in app.metric)
    app.radio[0].set_value("ballot_stuff")
    app.slider[0].set_value(50).run()
    assert not app.exception
    assert any(m.delta.startswith("rose") for m in app.metric)
