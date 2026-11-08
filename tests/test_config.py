from recsys.config import Settings


def test_defaults_match_local_stack():
    settings = Settings()
    assert settings.kafka_bootstrap == "localhost:19092"
    assert settings.topic_user_events == "user-events"
    assert settings.s3_bucket == "recsys-lake"


def test_from_env_overrides_and_casts_types():
    env = {
        "KAFKA_BOOTSTRAP": "redpanda:9092",
        "SINK_BATCH_SIZE": "250",
        "SINK_FLUSH_INTERVAL_S": "2.5",
    }
    settings = Settings.from_env(env)
    assert settings.kafka_bootstrap == "redpanda:9092"
    assert settings.sink_batch_size == 250
    assert isinstance(settings.sink_batch_size, int)
    assert settings.sink_flush_interval_s == 2.5


def test_from_env_ignores_unrelated_variables():
    settings = Settings.from_env({"PATH": "/usr/bin", "HOME": "/root"})
    assert settings == Settings()


def test_dotenv_is_read_and_blank_values_and_comments_are_skipped(tmp_path):
    from recsys.config import read_dotenv

    file = tmp_path / ".env"
    file.write_text(
        '# comment\nLLM_API_KEY="secret"\n\nREDIS_URL=redis://x:1/0\nLLM_MODEL=\nexport A=b\n'
    )
    assert read_dotenv(file) == {"LLM_API_KEY": "secret", "REDIS_URL": "redis://x:1/0", "A": "b"}
    assert read_dotenv(tmp_path / "missing.env") == {}


def test_from_env_uses_the_dotenv_file_but_real_variables_win(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLM_API_KEY=from-file\nLLM_MODEL=file-model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "from-shell")
    settings = Settings.from_env()
    assert settings.llm_api_key == "from-file" and settings.llm_model == "from-shell"


def test_dotenv_saved_with_a_byte_order_mark_still_works(tmp_path):
    from recsys.config import read_dotenv

    file = tmp_path / ".env"
    file.write_bytes("\ufeffLLM_API_KEY=abc\n".encode())
    assert read_dotenv(file) == {"LLM_API_KEY": "abc"}
