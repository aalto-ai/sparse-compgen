import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for root, dirnames, filenames in os.walk(args.directory):
        if len(dirnames) > 1 and all([d.startswith("version") for d in dirnames]):
            all_dirnames = sorted(dirnames, key=lambda d: int(d.split("_")[-1]))
            all_dirnames = [os.path.join(root, d) for d in all_dirnames]

            drop_directories = all_dirnames[:-1]
            print(f"In path {root}")
            for drop_dir in drop_directories:
                print(f"Dropping {drop_dir}")
            print(f"Keeping {all_dirnames[-1]}")

            if not args.dry_run:
                for drop_dir in drop_directories:
                    shutil.rmtree(drop_dir, ignore_errors=True)


if __name__ == "__main__":
    main()