from __future__ import annotations

import argparse
import email
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def locked_names(path: Path) -> set[str]:
    return {
        normalize(line.split("==", 1)[0])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def wheel_identity(wheel: Path) -> tuple[str, str, list[str]]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_path))
        license_paths = [
            name
            for name in archive.namelist()
            if ".dist-info/licenses/" in name.casefold()
            or PurePosixPath(name).name.casefold().startswith(
                ("license", "copying", "notice")
            )
        ]
    return normalize(str(metadata["Name"])), str(metadata["Version"]), license_paths


def collect(
    wheels_dir: Path,
    output_dir: Path,
    lock_path: Path,
    review_path: Path,
) -> None:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    reviewed = {normalize(item["name"]): item for item in review["packages"]}
    expected = locked_names(lock_path)
    if set(reviewed) != expected:
        raise ValueError("Converter license review does not match the locked distributions.")
    wheels: dict[str, tuple[Path, str, list[str]]] = {}
    for wheel in wheels_dir.glob("*.whl"):
        name, version, license_paths = wheel_identity(wheel)
        if name in wheels:
            raise ValueError(f"Duplicate converter wheel for {name}.")
        wheels[name] = (wheel, version, license_paths)
    if set(wheels) != expected:
        missing = sorted(expected - set(wheels))
        extra = sorted(set(wheels) - expected)
        raise ValueError(f"Converter wheels do not match lock; missing={missing}, extra={extra}.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    fallbacks = review.get("packages_by_name", {})
    for name in sorted(expected):
        wheel, version, license_paths = wheels[name]
        package_dir = output_dir / name
        package_dir.mkdir()
        copied: list[str] = []
        with zipfile.ZipFile(wheel) as archive:
            for index, member in enumerate(license_paths, start=1):
                source_name = PurePosixPath(member).name
                target = package_dir / f"{index:03d}-{source_name}"
                target.write_bytes(archive.read(member))
                copied.append(target.relative_to(output_dir).as_posix())
        if not copied:
            fallback = fallbacks.get(name)
            if not fallback:
                raise ValueError(f"Wheel {wheel.name} has no license file or fallback.")
            source = review_path.parent / fallback["license_file"]
            if not source.is_file():
                raise ValueError(f"Reviewed fallback license is missing: {source}.")
            target = package_dir / source.name
            shutil.copy2(source, target)
            copied.append(target.relative_to(output_dir).as_posix())
        manifest.append(
            {
                "name": name,
                "version": version,
                "license": reviewed[name]["license"],
                "files": copied,
            }
        )
    (output_dir / "NOTICE.json").write_text(
        json.dumps({"schema": review["schema"], "packages": manifest}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    collect(args.wheels, args.output, args.lock, args.review)


if __name__ == "__main__":
    main()
