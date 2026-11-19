"""Validation of config/users.yaml."""


def test_unknown_scope_key_in_users_file_is_reported_at_startup(tmp_path):
    import pytest

    from agent.config import load_users

    bad = tmp_path / "users.yaml"
    bad.write_text("users:\n  eve:\n    scope:\n      region: [EU]\n")
    with pytest.raises(ValueError, match="user 'eve' has unsupported scope key"):
        load_users(bad)


def test_only_ceo_can_have_unrestricted_access(tmp_path):
    import pytest

    from agent.config import load_users

    path = tmp_path / "users.yaml"
    path.write_text("users:\n  manager:\n    scope: {}\n")
    with pytest.raises(ValueError, match="only admin"):
        load_users(path)


def test_managers_require_a_brand_list(tmp_path):
    import pytest

    from agent.config import load_users

    for value in ("[]", "Nike", "[null]"):
        path = tmp_path / "users.yaml"
        path.write_text(f"users:\n  manager:\n    scope:\n      brand: {value}\n")
        with pytest.raises(ValueError, match="non-empty list"):
            load_users(path)
    for key in ("department", "category"):
        path.write_text(f"users:\n  manager:\n    scope:\n      {key}: [Women]\n")
        with pytest.raises(ValueError, match="unsupported scope"):
            load_users(path)


def test_configured_managers_have_brand_scopes_and_ceo_is_unrestricted():
    from agent.config import load_users

    users = load_users()
    assert not users["admin"].scope
    assert all(set(user.scope) == {"brand"} and user.scope["brand"] for name, user in users.items() if name != "admin")


def test_demo_uses_dataset_brands_directly():
    from agent.config import load_users
    from agent.core.prompts import build_system_prompt

    users = load_users()
    assert users["alice"].scope == {"brand": ("Calvin Klein Jeans",)}
    assert users["bob"].scope == {"brand": ("Allegra K",)}
    assert users["carol"].describe_scope() == "brand in ['Allegra K', 'Calvin Klein Jeans']"
    prompt = build_system_prompt(users["alice"], [])
    assert "Calvin Klein Jeans" in prompt and "Demo brand labels" not in prompt
