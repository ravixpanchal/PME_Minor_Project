"""Group 1 pipeline entrypoint: assembly -> preprocess -> extraction."""
import argparse

from src.utils.config_loader import config
from src.utils.logger import setup_logger

log = setup_logger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Report Foundation Pipeline")
    parser.add_argument("--phase", choices=["assembly", "preprocess", "extract", "all"],
                        default="all")
    parser.add_argument("--dummy", action="store_true",
                        help="Smoke test without touching real data")
    parser.add_argument("--limit", type=int, default=None,
                        help="preprocess only first N pages (dry-run mode)")
    parser.add_argument("--source", choices=["vlm", "clean", "raw"], default="vlm")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.phase == "all":
        parser.error("--limit is preprocessing-only; use --phase preprocess or extraction --ndocs")

    config.bootstrap_dirs()
    log.info("PME pipeline — phase=%s", args.phase)
    log.info("project root: %s", config.root_dir)

    if args.dummy:
        log.info("dummy run — nothing executed")
        return

    if args.phase in ("assembly", "all"):
        from src.assembly.run_assembly import main as run_assembly
        if run_assembly() != 0:
            log.error("assembly failed — aborting")
            raise SystemExit(1)

    if args.phase in ("preprocess", "all"):
        from src.preprocessing.run_preprocessing import main as run_preprocess
        if run_preprocess(limit=args.limit) != 0:
            raise SystemExit(1)
    if args.phase in ("extract", "all"):
        from src.extraction.run_extraction import main as run_extraction
        extraction_args = ["--source", args.source]
        if args.overwrite:
            extraction_args.append("--overwrite")
        raise SystemExit(run_extraction(extraction_args))


if __name__ == "__main__":
    main()