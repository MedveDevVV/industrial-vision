
from src.config import load_config
from src.logger import setup_logger

from src.app.arguments import parse_arguments
from src.app.application import Application


def main():
    config = load_config()
    logger = setup_logger(config["app"]["log_level"])
    args = parse_arguments()

    # Настройки запуска.
    video_config = config["video"]
    inference_config = config["inference"]

    # Аргументы командной строки имеют приоритет над YAML.
    source = (
        args.source
        if args.source is not None
        else video_config["source"]
    )

    update_interval = (
        args.update_interval
        if args.update_interval is not None
        else inference_config["update_interval_ms"]
    )

    logger.info(f"Запуск приложения: {config['app']['name']}")
    logger.info(f"Источник видео: {source}")

    try:
        application = Application(
            config=config,
            source=source,
            update_interval=update_interval,
            logger=logger
        )

        application.run()

    except Exception:
        logger.exception("Ошибка выполнения приложения")


if __name__ == "__main__":
    main()
