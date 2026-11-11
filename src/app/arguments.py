
import argparse


def parse_source(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Industrial Vision — система технического зрения"
    )

    parser.add_argument(
        "--source",
        type=parse_source,
        default=None,
        help="Источник видео: номер камеры или путь к видеофайлу"
    )

    parser.add_argument(
        "--update-interval",
        type=float,
        default=None,
        help="Интервал между запусками YOLO в миллисекундах"
    )

    args = parser.parse_args()

    if args.update_interval is not None and args.update_interval <= 0:
        parser.error("--update-interval должен быть больше нуля")

    return args
