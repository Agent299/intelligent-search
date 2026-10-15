"""Synthetic users and search behavior, standing in for production analytics.

Produces what the offline pipeline, ranking and discovery phases need:
- user_profile rows (brand/category affinities, search + purchase history)
- search events in the analytic-log schema (doc p.166-167), written as Parquet to the S3 data lake
  at s3://datalake/search_events/dt=YYYY-MM-DD/part-0.parquet (and data/lake/ locally)
- co-purchases: after a purchase, sessions sometimes buy a complementary item (pizza -> soda,
  brownie mix -> ice cream toppings), which Phase 7 mines into 'pairs well with'

Queries are derived from the products a user intends to buy (product_type, brand, attributes,
search tags), with a Zipf-like popularity skew so a small head of queries dominates traffic (the
80/20 split the cache tiers rely on).

    uv run python -m pipelines.synthetic.generate --users 5000 --days 14 [--publish-recent 2000]
"""

import argparse
import asyncio
import io
import json
import random
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import polars as pl
from psycopg.types.json import Jsonb

from common.config import get_settings
from common.db import connect
from common.normalize import normalize_query
from common.topics import QUERY_LOGS

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata", "Jaipur"]
DEVICES = ["mobile_app_android", "mobile_app_ios", "web"]
LOCAL_LAKE = Path("data/lake/search_events")

# (trigger keywords on the purchased item) -> (keywords for the complementary item)
COMPLEMENTS = [
    (("pizza", "chips", "popcorn", "snack", "pretzel"), ("soda", "soft drink", "cola", "sparkling", "juice")),
    (("brownie", "cake mix", "baking mix", "cookie"), ("ice cream", "whipped", "chocolate sauce", "sprinkles")),
    (("coffee",), ("creamer", "mug", "syrup", "filter")),
    (("tea",), ("honey", "kettle", "infuser", "mug")),
    (("pasta", "spaghetti"), ("pasta sauce", "parmesan", "olive oil")),
    (("rice",), ("lentil", "dal", "spice", "ghee")),
    (("shoe", "sneaker", "boot"), ("sock", "insole", "shoe care")),
    (("dress",), ("earring", "necklace", "sandal", "handbag")),
    (("jacket", "fleece", "coat"), ("beanie", "glove", "scarf", "thermal")),
    (("laptop", "notebook computer"), ("mouse", "laptop sleeve", "usb hub", "keyboard")),
    (("phone", "smartphone"), ("phone case", "screen protector", "charger")),
    (("protein", "energy bar"), ("shaker", "electrolyte", "nut butter")),
    (("cookware", "frying pan", "skillet"), ("spatula", "utensil", "oven mitt")),
]
QUERY_MODIFIERS = ["cheap", "best", "organic", "premium", "small", "large", "gift"]


def load_products() -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT product_id, name, brand, category_path, product_type, search_tags, attributes,
                              price, rating, review_count FROM product_catalog WHERE in_stock""")
        rows = cur.fetchall()
    for p in rows:
        p["product_type"] = p["product_type"] or p["category_path"].split(" > ")[-1].lower()
        p["text"] = f"{p['name']} {p['product_type']} {p['category_path']}".lower()
        p["top"] = p["category_path"].split(" > ")[0]
    return rows


def build_complement_index(products: list[dict]) -> dict[str, list[int]]:
    """For each trigger group, product indices that qualify as its complement."""
    index: dict[str, list[int]] = {}
    for triggers, targets in COMPLEMENTS:
        index["|".join(triggers)] = [i for i, p in enumerate(products) if any(t in p["text"] for t in targets)]
    return index


def complement_for(p: dict, cindex: dict[str, list[int]], rng: random.Random) -> int | None:
    for triggers, _ in COMPLEMENTS:
        if any(t in p["text"] for t in triggers):
            pool = cindex["|".join(triggers)]
            return rng.choice(pool) if pool else None
    return None


def make_query(p: dict, rng: random.Random) -> str:
    """Head queries (type, brand+type) dominate; long-tail variants use attributes and tags."""
    r = rng.random()
    attrs = [str(v) for v in (p["attributes"] or {}).values() if v and len(str(v)) < 25]
    if r < 0.45:
        q = p["product_type"]
    elif r < 0.65 and p["brand"]:
        q = f"{p['brand']} {p['product_type']}"
    elif r < 0.80 and attrs:
        q = f"{rng.choice(attrs)} {p['product_type']}"
    elif r < 0.90 and p["search_tags"]:
        q = rng.choice(p["search_tags"])
    else:
        q = f"{rng.choice(QUERY_MODIFIERS)} {p['product_type']}"
    if rng.random() < 0.04 and len(q) > 5:  # occasional typo for spell-correction to handle
        i = rng.randrange(1, len(q) - 1)
        q = q[:i] + q[i + 1] + q[i] + q[i + 2:]
    return q


def generate(n_users: int, days: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    products = load_products()
    if not products:
        raise SystemExit("product_catalog is empty - run the ingestion pipeline first")
    by_top: dict[str, list[int]] = defaultdict(list)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(products):
        by_top[p["top"]].append(i)
        by_cat[p["category_path"]].append(i)
    # Popularity skew: review volume ^ 0.6 plus noise.
    weight = [((p["review_count"] or 0) + 1) ** 0.6 * rng.uniform(0.5, 1.5) for p in products]
    tops = sorted(by_top)
    brands = Counter(p["brand"] for p in products if p["brand"])
    top_brands = [b for b, _ in brands.most_common(400)]
    cindex = build_complement_index(products)
    brand_of = {p["product_id"]: p["brand"] for p in products}

    now = datetime.now(UTC).replace(microsecond=0)
    users, events = [], []
    for u in range(n_users):
        uid = f"u{u:05d}"
        cat_aff = {t: rng.random() ** 2 for t in tops}
        fav_brands = rng.sample(top_brands, k=min(3, len(top_brands)))
        device, city = rng.choice(DEVICES), rng.choice(CITIES)
        searches, purchases = [], []
        for _ in range(max(1, int(rng.expovariate(1 / 6)))):  # sessions per user, mean ~6
            sid = uuid.uuid4().hex[:12]
            t = now - timedelta(days=rng.uniform(0, days), hours=rng.gauss(0, 3))
            top = rng.choices(tops, weights=[cat_aff[x] for x in tops])[0]
            pool = by_top[top]
            for _ in range(rng.randint(1, 4)):  # searches per session
                target_i = rng.choices(pool, weights=[weight[i] for i in pool])[0]
                if rng.random() < 0.3:  # brand loyalty: swap to a favourite brand in the same category
                    same = [i for i in by_cat[products[target_i]["category_path"]]
                            if products[i]["brand"] in fav_brands]
                    target_i = rng.choice(same) if same else target_i
                t += timedelta(seconds=rng.randint(20, 240))
                purchased_i = _search_event(events, products, by_cat, target_i, uid, sid, t, device, city, rng)
                searches.append(make_query(products[target_i], rng))
                if purchased_i is None:
                    continue
                purchases.append(products[purchased_i]["product_id"])
                comp = complement_for(products[purchased_i], cindex, rng)
                if comp is not None and rng.random() < 0.45:
                    t += timedelta(seconds=rng.randint(20, 180))
                    bought = _search_event(events, products, by_cat, comp, uid, sid, t, device, city, rng,
                                           purchase_p=0.8)
                    if bought is not None:
                        purchases.append(products[bought]["product_id"])
        brand_aff = Counter(brand_of[pid] for pid in purchases if brand_of.get(pid))
        total = sum(brand_aff.values()) or 1
        users.append({
            "user_id": uid,
            "search_history": searches[-20:],
            "purchase_history": purchases[-50:],
            "preferences": {
                "brand_affinities": {b: round(c / total, 3) for b, c in brand_aff.most_common(5)}
                or {b: 0.2 for b in fav_brands},
                "category_affinities": {k: round(v, 3) for k, v in cat_aff.items()},
                "device": device, "geolocation": city,
            },
        })
    return users, events


def _search_event(events, products, by_cat, target_i, uid, sid, t, device, city, rng, purchase_p=0.5):
    """One search: target product shown among same-category items; click/cart/purchase funnel.
    Returns the purchased product index, if any."""
    target = products[target_i]
    query = make_query(target, rng)
    siblings = [i for i in by_cat[target["category_path"]] if i != target_i]
    shown = rng.sample(siblings, k=min(19, len(siblings)))
    shown.insert(min(len(shown), int(rng.expovariate(1 / 3))), target_i)
    clicked = cart = None
    purchased: list[str] = []
    bought_i = None
    if rng.random() < 0.65:
        click_i = target_i if rng.random() < 0.7 else rng.choice(shown)
        clicked = products[click_i]["product_id"]
        if rng.random() < 0.45:
            cart = clicked
            if rng.random() < purchase_p:
                purchased, bought_i = [clicked], click_i
    events.append({
        "event_id": uuid.uuid4().hex,
        "event_timestamp": t,
        "user_id": uid,
        "session_id": sid,
        "raw_query": query,
        "normalized_query": normalize_query(query),
        "filters_applied": json.dumps({}),
        "search_type": "hybrid",
        "displayed_product_ids": [products[i]["product_id"] for i in shown],
        "clicked_product_id": clicked,
        "added_to_cart_product_id": cart,
        "purchased_product_ids": purchased,
        "device_type": device,
        "geolocation": city,
    })
    return bought_i


def write_lake(events: list[dict]) -> list[str]:
    s = get_settings()
    s3 = boto3.client("s3", endpoint_url=s.s3_endpoint, aws_access_key_id=s.s3_access_key,
                      aws_secret_access_key=s.s3_secret_key, region_name="us-east-1")
    df = pl.DataFrame(events).with_columns(pl.col("event_timestamp").dt.date().alias("dt"))
    keys = []
    for (dt,), part in df.partition_by("dt", as_dict=True).items():
        part = part.drop("dt").sort("event_timestamp")
        local = LOCAL_LAKE / f"dt={dt}"
        local.mkdir(parents=True, exist_ok=True)
        part.write_parquet(local / "part-0.parquet")
        buf = io.BytesIO()
        part.write_parquet(buf)
        key = f"search_events/dt={dt}/part-0.parquet"
        s3.put_object(Bucket="datalake", Key=key, Body=buf.getvalue())
        keys.append(key)
    return keys


def write_profiles(users: list[dict]) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO user_profile (user_id, search_history, purchase_history, preferences)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (user_id) DO UPDATE SET search_history = EXCLUDED.search_history,
                   purchase_history = EXCLUDED.purchase_history, preferences = EXCLUDED.preferences,
                   updated_at = now()""",
            [(u["user_id"], u["search_history"], u["purchase_history"], Jsonb(u["preferences"])) for u in users],
        )


async def publish_recent(events: list[dict], n: int) -> None:
    from aiokafka import AIOKafkaProducer

    producer = AIOKafkaProducer(bootstrap_servers=get_settings().kafka_bootstrap)
    await producer.start()
    try:
        for e in sorted(events, key=lambda e: e["event_timestamp"])[-n:]:
            payload = {**e, "event_timestamp": e["event_timestamp"].isoformat()}
            await producer.send(QUERY_LOGS, json.dumps(payload).encode(), key=e["session_id"].encode())
        await producer.flush()
    finally:
        await producer.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--publish-recent", type=int, default=0, help="also stream the N latest events to query_logs")
    args = ap.parse_args()

    users, events = generate(args.users, args.days, args.seed)
    keys = write_lake(events)
    write_profiles(users)
    if args.publish_recent:
        asyncio.run(publish_recent(events, args.publish_recent))
    purchases = sum(1 for e in events if e["purchased_product_ids"])
    clicks = sum(1 for e in events if e["clicked_product_id"])
    uniq = Counter(e["normalized_query"] for e in events)
    head = sum(c for _, c in uniq.most_common(max(1, len(uniq) // 5)))
    print(f"users={len(users)} events={len(events)} clicks={clicks} purchases={purchases} "
          f"unique_queries={len(uniq)} top20%_share={head / len(events):.0%} lake_partitions={len(keys)}")


if __name__ == "__main__":
    main()
