"""CLI: python -m src.longterm [--report]"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-term holdings desk")
    parser.add_argument("--report", action="store_true",
                        help="write the HTML holdings report and exit")
    args = parser.parse_args()
    if args.report:
        from .report import write_report
        html_path, _ = write_report()
        print(f"report written: {html_path}")
        return
    from .board import menu
    menu()


if __name__ == "__main__":
    main()
