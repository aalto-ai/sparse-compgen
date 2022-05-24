import argparse
import os
import torch
import fnmatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pattern", default="*.pt")
    args = parser.parse_args()

    for root, dirnames, filenames in os.walk(args.directory):
        for filename in fnmatch.filter(filenames, args.pattern):
            fullpath = os.path.join(root, filename)

            print(f"Opening {fullpath}")
            try:
                model = torch.load(fullpath, map_location=torch.device("cpu"))[
                    "state_dict"
                ]
            except KeyError:
                print(f"Ignoring {fullpath} since it is not a pl-checkpoint")

            print(f"Writing {fullpath}")
            if not args.dry_run:
                torch.save(model, fullpath)

    print("Done")


if __name__ == "__main__":
    main()
