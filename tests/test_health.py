from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings


def test_health_returns_ok() -> None:
    """Protects the liveness probe; rejects any non-200 or a body other than {"status": "ok"}."""
    client = TestClient(create_app(make_settings()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_settings_split_comma_separated_cors_origins() -> None:
    """Protects CORS_ORIGINS parsing; rejects treating the comma-separated string as one origin."""
    settings = make_settings(cors_origins="http://localhost:3000, https://wisdom.example.com")

    assert settings.cors_origins == ["http://localhost:3000", "https://wisdom.example.com"]
