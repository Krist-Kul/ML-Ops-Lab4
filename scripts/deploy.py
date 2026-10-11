"""Lab 3 Task 2 — put a registered version behind a managed endpoint.

    make deploy                         # build, push, register a serving version, deploy
    make deploy SOURCE_VERSION=production
    make deploy INSTANCE=n1-standard-4
    make smoke                          # then call it with three known payloads
    make teardown                       # THE ENDPOINT IS BILLED BY THE HOUR. Do this.

Why a second registered version exists. A Vertex version carries the container that serves
it, and Lab 2's version 1 names the *training* image — it was registered before a serving
image existed, and Vertex refuses a version with no container at all. Deploying version 1
would start `python -m src.train` behind an HTTP endpoint. So this registers a new version
whose artifact_uri is the same object (the same model bytes, not a retrain) and whose
container is the serving image with Vertex's routes declared on it.

The lineage copied onto the new version is the source version's, unchanged: those eight
fields describe how the model bytes were produced, and they did not change because we put
a different web server in front of them. The serving image is recorded separately, in the
container spec and in reports/lab3-deploy.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_remote import resolve_image  # noqa: E402

from cloudlayer.factory import get_adapter  # noqa: E402
from src import config  # noqa: E402

RECORD = Path("reports/lab3-deploy.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", help="digest-pinned serving image already pushed")
    src.add_argument("--local-tag", help="local serving image to push first")
    p.add_argument("--name", default=None, help="registry name; defaults to MODEL_REGISTRY_NAME")
    p.add_argument("--source-version", default="production",
                   help="the registered version whose MODEL BYTES get served; "
                        "a version id or an alias")
    p.add_argument("--serving-version", default=None,
                   help="deploy this existing serving version instead of registering one")
    p.add_argument("--endpoint", default=None, help="endpoint display name")
    p.add_argument("--instance", default="n1-standard-2")
    p.add_argument("--min-replicas", type=int, default=1)
    p.add_argument("--max-replicas", type=int, default=1)
    p.add_argument("--traffic", type=int, default=100,
                   help="share of traffic this deployment takes; Task 4 uses less")
    p.add_argument("--spot", action="store_true",
                   help="discounted replicas. Evictions make a load test measure the "
                        "platform rather than the service, so not for Task 3")
    p.add_argument("--promote", default="serving",
                   help="alias to move to the new serving version; empty string to skip")
    p.add_argument("--record", type=Path, default=RECORD)
    p.add_argument("--register-only", action="store_true",
                   help="register the serving version and move the alias, but start no "
                        "replica. Lab 4's CD: staging is a registry alias, not a warm endpoint")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = config.load()
    if cfg.provider == "local":
        print("CLOUD_PROVIDER=local has no managed endpoint. Set your provider in cloud.env.")
        return 2
    adapter = get_adapter(cfg)
    name = args.name or cfg.model_registry_name
    endpoint_name = args.endpoint or f"{name}-endpoint"

    source = adapter.get_model(name, args.source_version)
    print(f"source   {name} version {source['version']} "
          f"(aliases: {', '.join(source['aliases']) or '—'})")
    print(f"artifact {source['artifact_uri']}")

    if args.serving_version:
        serving_version, image = args.serving_version, None
        entry = adapter.get_model(name, serving_version)
        print(f"serving  reusing version {entry['version']}")
    else:
        if not (args.image or args.local_tag):
            raise SystemExit("pass --image or --local-tag: a deployment needs a serving image")
        image = resolve_image(adapter, args.image, args.local_tag)
        # MODEL_VERSION tells the container which model's bytes it is serving, so every
        # response body names it. Vertex reports the *serving* version separately in its
        # own predict response; both are recorded, because they answer different questions.
        spec = adapter.serving_spec(env={"MODEL_VERSION": source["version"]})
        serving_version = adapter.register_model(
            source["artifact_uri"], name, lineage=source["lineage"],
            serving_image=image, serving=spec)
        print(f"serving  registered version {serving_version} "
              f"(routes {spec['health_route']} / {spec['predict_route']})")
        if args.promote:
            aliases = adapter.promote_model(name, serving_version, args.promote)["aliases"]
            print(f"alias    {args.promote} -> version {serving_version} "
                  f"({', '.join(aliases)})")

    if args.register_only:
        record = {
            "serving_version": serving_version,
            "alias": args.promote,
            "source_version": source["version"],
            "source_artifact_uri": source["artifact_uri"],
            "serving_image": image,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": None,
            "lineage": source["lineage"],
        }
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2))
        print(f"no endpoint started (--register-only). wrote {args.record}")
        return 0

    started = time.time()
    print(f"deploy   {endpoint_name} on {args.instance} "
          f"({args.min_replicas}-{args.max_replicas} replicas, {args.traffic}% traffic) "
          "— several minutes, and the meter starts when the replica does")
    endpoint = adapter.deploy(
        f"{name}@{serving_version}", endpoint_name, args.instance,
        min_replicas=args.min_replicas, max_replicas=args.max_replicas,
        spot=args.spot, traffic=args.traffic, lab=3)
    elapsed = time.time() - started
    print(f"endpoint {endpoint}\ndeployed in {elapsed / 60:.1f} min")

    record = {
        "endpoint": endpoint,
        "endpoint_display_name": endpoint_name,
        "deployed_model_id": getattr(adapter, "_last_deployed_model_id", ""),
        "serving_version": serving_version,
        "source_version": source["version"],
        "source_artifact_uri": source["artifact_uri"],
        "serving_image": image,
        "instance": args.instance,
        "min_replicas": args.min_replicas,
        "max_replicas": args.max_replicas,
        "spot": args.spot,
        "traffic": args.traffic,
        "deploy_seconds": round(elapsed, 1),
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lineage": source["lineage"],
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2))
    print(f"wrote    {args.record}")
    print("\nThis endpoint is billed per replica-hour whether or not it is called.\n"
          "  make smoke        verify it answers\n"
          "  make teardown     delete it — do this the moment you are finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
