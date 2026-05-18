"""
Unit tests for authentication setup behavior.
"""

import json

from backend.auth import AuthManager


def test_auth_manager_does_not_create_default_admin(tmp_path, monkeypatch):
    """A fresh deployment must not expose a known admin/admin account."""
    monkeypatch.delenv("PROMPT2NOTES_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("PROMPT2NOTES_ADMIN_PASSWORD", raising=False)

    users_file = tmp_path / "users.json"
    manager = AuthManager(users_file=users_file)

    assert users_file.exists()
    assert json.loads(users_file.read_text()) == {}
    assert manager.authenticate("admin", "admin")[0] is False


def test_auth_manager_can_seed_admin_from_env(tmp_path, monkeypatch):
    """Deployments can still create an initial account with explicit secrets."""
    monkeypatch.setenv("PROMPT2NOTES_ADMIN_USERNAME", "owner")
    monkeypatch.setenv("PROMPT2NOTES_ADMIN_PASSWORD", "not-the-default")
    monkeypatch.setenv("PROMPT2NOTES_ADMIN_EMAIL", "owner@example.com")

    manager = AuthManager(users_file=tmp_path / "users.json")

    assert manager.authenticate("owner", "not-the-default") == (True, None)
    assert manager.authenticate("admin", "admin")[0] is False
