import argparse
import collections
import json
import os
import re
import html
import datetime
from tqdm import tqdm
import numpy as np


def clean_text(text):
    """Clean text by removing HTML and excessive whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", str(text))         # remove HTML tags
    text = html.unescape(text)                       # decode HTML entities
    text = re.sub(r"\s+", " ", text)                 # collapse whitespace
    return text.strip()


def check_path(path):
    os.makedirs(path, exist_ok=True)


def write_json_file(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def write_remap_index(index_map, file_path):
    with open(file_path, "w") as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")

def get_timestamp_start(year, month):
    """Return epoch seconds for start of a month."""
    return int(datetime.datetime(year, month, 1).timestamp())


def convert_ms_to_sec(ts_ms):
    """Amazon23 sort_timestamp is in milliseconds. Convert to seconds."""
    try:
        return int(ts_ms // 1000)
    except:
        return 0

def load_reviews_amazon23(reviews_file):

    if not os.path.exists(reviews_file):
        print(f"[ERROR] Reviews file not found: {reviews_file}")
        return []

    reviews = []
    print(f"Loading Amazon23 reviews from: {reviews_file}")

    with open(reviews_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading reviews"):
            try:
                r = json.loads(line.strip())

                # Amazon 23 official: use parent_asin as product ID
                item_id = r.get("parent_asin")
                if item_id is None:
                    print("wrong!")
                if item_id is None:
                    continue

                if "user_id" not in r:
                    continue

                ts = convert_ms_to_sec(r.get("timestamp", 0))

                review_obj = {
                    "user_id": r["user_id"],
                    "asin": item_id,  # <-- THIS IS THE FIX
                    "rating": float(r.get("rating", 0)),
                    "review_title": clean_text(r.get("title", "")),
                    "review_text": clean_text(r.get("text", "")),
                    "timestamp": ts,
                    "verified": bool(r.get("verified_purchase", False)),
                    "helpful_votes": int(r.get("helpful_votes", 0)),
                    "images": r.get("images", [])
                }

                reviews.append(review_obj)

            except Exception:
                continue

    print(f"[INFO] Loaded {len(reviews)} Amazon23 reviews")
    return reviews

def load_metadata_amazon23(metadata_file):

    if not os.path.exists(metadata_file):
        print(f"[ERROR] Metadata file not found: {metadata_file}")
        return {}, {}

    print(f"Loading Amazon23 metadata from: {metadata_file}")

    asin2meta = {}
    asin2title = {}

    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading metadata"):
            try:
                m = json.loads(line.strip())

                # Amazon 23 metadata may not have "asin", only "parent_asin"
                asin = m.get("parent_asin", None)

                if asin is None:
                    # skip items with no valid id
                    print("wrong!")
                    continue

                title = clean_text(m.get("title", ""))

                asin2meta[asin] = {
                    "title": title,
                    "main_category": m.get("main_category", ""),
                    "features": m.get("features", []),
                    "description": m.get("description", []),
                    "price": m.get("price", None),
                    "images": m.get("images", []),
                    "categories": m.get("categories", []),
                    "details": m.get("details", {}),
                    "store": m.get("store", ""),
                    "brand": m.get("store", ""),
                    "parent_asin": m.get("parent_asin", "")
                }

                if len(title) > 0:
                    asin2title[asin] = title

            except Exception:
                continue

    print(f"[INFO] Loaded {len(asin2meta)} metadata entries, {len(asin2title)} titles")
    return asin2meta, asin2title

def k_core_filter_amazon23(reviews, asin2title, K=5, start_timestamp=None, end_timestamp=None):
    """
    K-core filtering for Amazon23 reviews.
    Uses:
        - user_id
        - asin
        - timestamp (converted from sort_timestamp)
    And behaves exactly like your original json2csv-style k-core:
        - time filtering inside the loop
        - remove items without titles
        - iterative k-core until stable
    """

    print("[INFO] Begin k-core filtering for Amazon23")

    remove_users = set()
    remove_items = set()

    # Items without valid title removed first
    for r in reviews:
        if r["asin"] not in asin2title:
            remove_items.add(r["asin"])

    while True:
        user_counts = {}
        item_counts = {}
        new_reviews = []
        changed = False
        kept_total = 0

        for r in tqdm(reviews, desc="K-core iteration"):

            # time filtering inside loop (same as your original script)
            if start_timestamp and end_timestamp:
                ts = r["timestamp"]
                if ts < start_timestamp or ts > end_timestamp:
                    continue

            user = r["user_id"]
            item = r["asin"]

            if user in remove_users:
                continue
            if item in remove_items:
                continue

            # accumulate counts
            user_counts[user] = user_counts.get(user, 0) + 1
            item_counts[item] = item_counts.get(item, 0) + 1

            new_reviews.append(r)
            kept_total += 1

        # Identify users/items to remove
        for u, c in user_counts.items():
            if c < K:
                remove_users.add(u)
                changed = True

        for i, c in item_counts.items():
            if c < K:
                remove_items.add(i)
                changed = True

        print(
            f"[k-core] Users={len(user_counts)}, Items={len(item_counts)}, "
            f"Reviews={kept_total}, Density={kept_total / (max(1, len(user_counts) * len(item_counts))):.6f}"
        )

        # Stop if stable
        if not changed:
            break

        reviews = new_reviews

    print("[INFO] K-core filtering finished")
    return new_reviews, user_counts, item_counts

def convert_interactions_amazon23(reviews):
    """
    Convert interactions to amazon18-style:
    - user2index
    - item2index
    - user2items[user_index] = [item_index...]
    - interactions list of (user_id, asin, rating, timestamp)
    """

    user_reviews = collections.defaultdict(list)

    for r in reviews:
        user_reviews[r["user_id"]].append(r)

    # Sort each user's reviews by time
    for u in user_reviews:
        user_reviews[u].sort(key=lambda x: x["timestamp"])

    user2index, item2index = {}, {}
    user2items = collections.defaultdict(list)
    interactions = []

    for user in user_reviews:
        if user not in user2index:
            user2index[user] = len(user2index)

        for r in user_reviews[user]:
            item = r["asin"]
            if item not in item2index:
                item2index[item] = len(item2index)

            uid = user2index[user]
            iid = item2index[item]

            user2items[uid].append(iid)
            interactions.append((user, item, r["rating"], r["timestamp"]))

    print(
        f"[Mapping] Users={len(user2index)}, Items={len(item2index)}, "
        f"Interactions={len(interactions)}"
    )
    return user2items, user2index, item2index, interactions


def build_interaction_list_amazon23(reviews, user2index, item2index, asin2title):
    """
    Build interaction sequences (history_len=10).
    Same logic as your original json2csv version but using:
        - user_id
        - rating
        - timestamp
        - title from metadata
    Output entry:
        [user_id_original,
         history_asins,
         target_asin,
         history_item_ids,
         target_item_id,
         history_titles,
         target_title,
         history_ratings,
         target_rating,
         history_timestamps,
         target_timestamp]
    """

    # Group by user
    interact = {}

    for r in tqdm(reviews, desc="Collecting interactions"):
        u = r["user_id"]
        i = r["asin"]

        if u not in interact:
            interact[u] = {
                "items": [],
                "ratings": [],
                "timestamps": [],
                "item_ids": [],
                "titles": [],
            }

        interact[u]["items"].append(i)
        interact[u]["ratings"].append(r["rating"])
        interact[u]["timestamps"].append(r["timestamp"])
        interact[u]["item_ids"].append(item2index[i])
        interact[u]["titles"].append(asin2title.get(i, ""))

    # Build full list
    interaction_list = []

    for u in tqdm(interact.keys(), desc="Building sequence windows"):
        items = interact[u]["items"]
        ratings = interact[u]["ratings"]
        timestamps = interact[u]["timestamps"]
        item_ids = interact[u]["item_ids"]
        titles = interact[u]["titles"]

        # sort by timestamp
        all_data = list(zip(items, ratings, timestamps, item_ids, titles))
        all_data.sort(key=lambda x: x[2])
        items, ratings, timestamps, item_ids, titles = zip(*all_data)

        items = list(items)
        ratings = list(ratings)
        timestamps = list(timestamps)
        item_ids = list(item_ids)
        titles = list(titles)

        for i in range(1, len(items)):
            st = max(i - 10, 0)

            interaction_list.append([
                u,                       # original user_id
                items[st:i],             # history asins
                items[i],                # target asin
                item_ids[st:i],          # history item_ids
                item_ids[i],             # target item_id
                titles[st:i],            # history titles
                titles[i],               # target title
                ratings[st:i],           # history ratings
                ratings[i],              # target rating
                timestamps[st:i],        # history timestamps
                timestamps[i]            # target timestamp
            ])

    # sort global interactions by target timestamp
    interaction_list.sort(key=lambda x: int(x[-1]))
    print(f"[Sequence] Total sequences: {len(interaction_list)}")

    return interaction_list


def write_atomic_files(args, interaction_list, user2index):
    print("[INFO] Writing atomic train/valid/test files")

    out_dir = os.path.join(args.output_path, args.dataset)
    check_path(out_dir)

    total = len(interaction_list)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    train_data = interaction_list[:train_end]
    valid_data = interaction_list[train_end:valid_end]
    test_data = interaction_list[valid_end:]

    print(f"[Split] train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")

    def write_file(path, data):
        with open(path, "w") as f:
            f.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
            for it in data:
                user_original = it[0]
                uid = user2index[user_original]

                hist = [str(x) for x in it[3]]
                target = str(it[4])

                hist = hist[-50:]  # cap history length = 50
                f.write(f"{uid}\t{' '.join(hist)}\t{target}\n")

    write_file(os.path.join(out_dir, f"{args.dataset}.train.inter"), train_data)
    write_file(os.path.join(out_dir, f"{args.dataset}.valid.inter"), valid_data)
    write_file(os.path.join(out_dir, f"{args.dataset}.test.inter"), test_data)

    return train_data, valid_data, test_data


def build_item_features_amazon23(asin2meta, item2index):
    """
    Convert Amazon23 metadata to amazon18-style:
        - title
        - description (list → str)
        - features
        - categories
        - brand/store
        - details (flattened)
        - images (hi_res list)
    """

    item2feature = {}

    for asin, idx in item2index.items():
        m = asin2meta.get(asin, {})

        title = clean_text(m.get("title", ""))

        # description: list → single string
        desc = m.get("description", [])
        if isinstance(desc, list):
            desc = " ".join([clean_text(d) for d in desc])
        else:
            desc = clean_text(desc)

        # features: list
        feats = m.get("features", [])
        feats = " ".join([clean_text(f) for f in feats]) if isinstance(feats, list) else clean_text(str(feats))

        # categories: list of strings → join
        cats = m.get("categories", [])
        if isinstance(cats, list):
            cats = ", ".join([clean_text(c) for c in cats])
        else:
            cats = clean_text(str(cats))

        # brand/store
        brand = clean_text(m.get("store", ""))
        details = m.get("details", {})

        # images: extract hi_res if exists
        images = m.get("images", [])
        image_urls = []
        for img in images:
            if isinstance(img, dict):
                if "hi_res" in img:
                    image_urls.append(img["hi_res"])
                elif "large" in img:
                    image_urls.append(img["large"])
                elif "thumb" in img:
                    image_urls.append(img["thumb"])

        item2feature[idx] = {
            "title": title,
            "description": desc,
            "features": feats,
            "categories": cats,
            "brand": brand,
            "details": details,
            "images": image_urls,
        }

    print(f"[Features] Built item features for {len(item2feature)} items")
    return item2feature


def build_review_data_amazon23(reviews, user2index, item2index):
    """
    Build review_data:
        key = (uid, iid, timestamp)
        value = {review, title}
    """

    review_data = {}

    for r in tqdm(reviews, desc="Building review_data"):
        user = r["user_id"]
        asin = r["asin"]

        if user not in user2index or asin not in item2index:
            continue

        uid = user2index[user]
        iid = item2index[asin]
        ts = r["timestamp"]

        key = str((uid, iid, ts))

        review_data[key] = {
            "review": clean_text(r.get("review_text", "")),
            "summary": clean_text(r.get("review_title", "")),
            "helpful_votes": r.get("helpful_votes", 0),
            "verified": r.get("verified", False),
        }

    print(f"[ReviewData] Stored {len(review_data)} review entries")
    return review_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="All_Beauty",
                        help="Dataset name (folder name)")

    parser.add_argument("--user_k", type=int, default=5,
                        help="K-core threshold for users and items")

    parser.add_argument("--st_year", type=int, default=1996)
    parser.add_argument("--st_month", type=int, default=1)
    parser.add_argument("--ed_year", type=int, default=2025)
    parser.add_argument("--ed_month", type=int, default=1)

    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to meta_...jsonl file")

    parser.add_argument("--reviews_file", type=str, required=True,
                        help="Path to ...jsonl review file")

    parser.add_argument("--output_path", type=str, default="./data",
                        help="Output folder")

    return parser.parse_args()


def main():
    args = parse_args()

    print("===================================================")
    print(" Amazon 2023 Dataset Processing Pipeline Starting ")
    print("===================================================")
    print(f"Dataset         : {args.dataset}")
    print(f"Metadata file   : {args.metadata_file}")
    print(f"Reviews file    : {args.reviews_file}")
    print(f"K-core threshold: {args.user_k}")
    print("---------------------------------------------------")

    # Load metadata and reviews
    asin2meta, asin2title = load_metadata_amazon23(args.metadata_file)
    reviews = load_reviews_amazon23(args.reviews_file)

    if len(reviews) == 0:
        print("[ERROR] No reviews loaded. Abort.")
        return

    # Time range filtering is done inside k-core (json2csv style)
    start_ts = get_timestamp_start(args.st_year, args.st_month)
    end_ts = get_timestamp_start(args.ed_year, args.ed_month)

    print(f"[INFO] Time range filtering inside k-core: {args.st_year}-{args.st_month} → {args.ed_year}-{args.ed_month}")


    print("Total reviews loaded:", len(reviews))
    print("Total metadata items:", len(asin2title))

    unique_items = len(set(r["asin"] for r in reviews))
    print("Unique items in reviews:", unique_items)

    items_in_meta = sum(1 for r in reviews if r["asin"] in asin2title)
    print("Reviews with metadata title:", items_in_meta)

    print("Min timestamp:", min(r["timestamp"] for r in reviews))
    print("Max timestamp:", max(r["timestamp"] for r in reviews))

    # K-core filtering
    reviews_filtered, user_counts, item_counts = k_core_filter_amazon23(
        reviews, asin2title, K=args.user_k,
        start_timestamp=start_ts,
        end_timestamp=end_ts
    )

    print(f"[K-core Result] Users={len(user_counts)}, Items={len(item_counts)}, Reviews={len(reviews_filtered)}")

    # Convert to index mappings
    user2items, user2index, item2index, interactions = convert_interactions_amazon23(
        reviews_filtered
    )

    # Build interaction_list for sequences
    interaction_list = build_interaction_list_amazon23(
        reviews_filtered, user2index, item2index, asin2title
    )

    # Write train/valid/test atomic files
    train_data, valid_data, test_data = write_atomic_files(
        args, interaction_list, user2index
    )

    # Build item features
    item2feature = build_item_features_amazon23(asin2meta, item2index)

    # Build review features
    review_data = build_review_data_amazon23(
        reviews_filtered, user2index, item2index
    )

    # Output folder
    out_dir = os.path.join(args.output_path, args.dataset)
    check_path(out_dir)

    # Save JSON outputs
    write_json_file(user2items, os.path.join(out_dir, f"{args.dataset}.inter.json"))
    write_json_file(item2feature, os.path.join(out_dir, f"{args.dataset}.item.json"))
    write_json_file(review_data, os.path.join(out_dir, f"{args.dataset}.review.json"))

    # Save mapping files
    write_remap_index(user2index, os.path.join(out_dir, f"{args.dataset}.user2id"))
    write_remap_index(item2index, os.path.join(out_dir, f"{args.dataset}.item2id"))

    print("===================================================")
    print(" Amazon 2023 Dataset Processing Completed Successfully")
    print("===================================================")


if __name__ == "__main__":
    main()
