import logging

from src.utils.notifier import FeishuNotifier


def test_feishu_send_failure_records_error_without_warning(caplog):
    notifier = FeishuNotifier(webhook_url="dummy", enabled=True)

    with caplog.at_level(logging.WARNING):
        sent = notifier.send_training_start(
            config_name="cfg",
            stages=[{"name": "stage", "epochs": 1}],
            total_epochs=1,
        )

    assert sent is False
    assert notifier.last_error == "unknown url type: 'dummy'"
    assert "Failed to send Feishu notification" not in caplog.text
