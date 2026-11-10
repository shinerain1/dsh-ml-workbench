from dsh_ml_workbench.doctor import diagnose


def test_doctor_closes_database_before_rename_and_temporary_cleanup(tmp_path):
    root = tmp_path / "installation"
    report = diagnose(root)
    storage = next(check for check in report["checks"] if check["name"] == "storage")
    assert storage["ok"], storage
    assert list(root.iterdir()) == []
