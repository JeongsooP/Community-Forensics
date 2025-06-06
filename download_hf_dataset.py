import datasets as ds
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="The repo id of the dataset to download.")
    parser.add_argument("--token", type=str, default=None, help="The token to use for authentication.")
    parser.add_argument("--cache_dir", type=str, default="~/.cache", help="The directory to cache the dataset.")
    parser.add_argument("--num_proc", type=int, default=1, help="The number of processes to use for downloading the dataset.")

    return parser.parse_args()

def main():
    # Downloads the dataset. This part is separated out for compatibility with DDP pipelines.
    args = parse_args()
    print(f"Downloading dataset: {args.repo_id}...")
    ds.load_dataset(
        args.repo_id,
        cache_dir=os.path.expanduser(args.cache_dir),
        split="train",
        token=args.token,
        num_proc=args.num_proc,
    )
    print(f"Dataset `{args.repo_id}` downloaded successfully.")

if __name__ == "__main__":
    main()