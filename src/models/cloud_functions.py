from google.cloud import storage
from datetime import datetime
import torch
import io


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def uploadModelwithTimestamp(config):
    print("[INFO] Uploading best model...")
    now = datetime.now()
    timestamp = now.strftime("D%d%m%YT%H%M%S")
    best_model_gcp_path = config.BUCKET_PATH + timestamp + config.BEST_MODEL_PATH.split("/")[-1]
    upload_blob(config.BUCKET_NAME, config.BEST_MODEL_PATH, best_model_gcp_path)


def loadCheckpointFromGCP(config):
    BUCKET_NAME = config.BUCKET_NAME
    MODEL_FILE = config.BUCKET_BEST_MODEL

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(MODEL_FILE)
    checkpoint = torch.load(io.BytesIO(blob.download_as_string()))
    return checkpoint
