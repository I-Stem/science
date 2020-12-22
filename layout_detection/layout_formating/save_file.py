"""
Save the file into the S3 bucket.
"""
import boto3
import tempfile
import os

sesssion = boto3.Session()
client = boto3.client("s3")
location = client.get_bucket_location(Bucket=os.getenv("STORAGE_BUCKET"))["LocationConstraint"]
istembucket = os.getenv("STORAGE_BUCKET")

def bucket_saver(file, hash_value, document_name, extention):
    """
    save the data into S3 bucket.
    """
    key_value = f"files/{hash_value}/{document_name}.{extention}"
    if extention == 'zip':
        res = client.upload_file(file, os.getenv("STORAGE_BUCKET"), key_value)
    else:
        if extention != "docx":
            client.put_object(Body=file, Bucket=os.getenv("STORAGE_BUCKET"), Key=key_value)
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                file_path = f"{tmpdirname}/temp.docx"
                file.save(file_path)
                client.upload_file(file_path, Bucket=os.getenv("STORAGE_BUCKET"), Key=key_value)
    url = f"https://s3-{location}.amazonaws.com/{istembucket}/{key_value}"
    print("S3 OCR file URL: ",url)
    return url
