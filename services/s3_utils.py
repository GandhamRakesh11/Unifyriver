# services/s3_utils.py

import boto3
import os
from botocore.exceptions import NoCredentialsError

from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION     = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET      = os.getenv("S3_BUCKET_NAME", "point9ml")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def upload_to_s3(local_path: str, s3_key: str):
    try:
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"✅ Uploaded to S3: {s3_key}")
        return True
    except Exception as e:
        print(f"❌ S3 upload failed: {e}")
        return False
