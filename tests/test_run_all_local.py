import sys

from scripts.run_all_local import main


def test_run_all_local_invokes_modules(monkeypatch):
    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        return 0

    monkeypatch.setattr("scripts.run_all_local.run", fake_run)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    main([])
    assert len(calls) == 8
    assert calls[0][:3] == ["/usr/bin/python3", "-m", "scripts.prepare_dataset"]
