"""Streamlit arayüzü duman testi (headless):  python tests/test_app.py"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_demo_mode():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4
    # Statik kaynak gizleme kapalıyken ve farklı takvim ölçüsüyle yeniden çalıştır
    at.toggle[0].set_value(False).run()
    assert not at.exception, at.exception
    for sb in at.selectbox:
        if sb.label in ("Calendar metric", "Takvim ölçüsü"):
            sb.set_value("frp").run()
            break
    assert not at.exception, at.exception


if __name__ == "__main__":
    test_app_demo_mode()
    print("OK test_app_demo_mode")
