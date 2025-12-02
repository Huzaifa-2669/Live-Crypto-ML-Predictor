import os
import sys
from pathlib import Path

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print("Missing 'minio' package. Install with: pip install minio")
    sys.exit(1)


def get_env(key: str, default: str = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        print(f"Environment variable {key} is required.")
        sys.exit(1)
    return val


def load_env_file(env_path: Path):
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    # Load .env.minio if present
    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env.minio"
    load_env_file(env_file)

    endpoint = get_env("MINIO_ENDPOINT", "http://localhost:9000")
    access_key = get_env("MINIO_ACCESS_KEY", "admin")
    secret_key = get_env("MINIO_SECRET_KEY", "admin12345")
    bucket = get_env("MINIO_BUCKET", "crypto-data")

    # MinIO SDK expects host without scheme; use secure flag accordingly
    secure = endpoint.startswith("https://")
    host = endpoint.replace("http://", "").replace("https://", "")

    client = Minio(host, access_key=access_key, secret_key=secret_key, secure=secure)

    try:
        # Ensure bucket exists
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"Created bucket: {bucket}")
        else:
            print(f"Bucket exists: {bucket}")

        # Prepare sample file
        data_dir = root / "minio" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        sample_file = data_dir / "hello.txt"
        sample_file.write_text("Hello MinIO!\n")

        object_name = "samples/hello.txt"
        client.fput_object(bucket, object_name, str(sample_file))
        print(f"Uploaded {object_name} to bucket {bucket}")

        # List objects to confirm
        print("Objects in bucket:")
        for obj in client.list_objects(bucket, prefix="samples/", recursive=True):
            print(f"- {obj.object_name} ({obj.size} bytes)")

    except S3Error as e:
        print(f"S3Error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
