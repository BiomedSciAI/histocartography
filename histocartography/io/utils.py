"""Whole Slide Image IO module."""
import os
import logging
import sys
import boto3

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::IO::UTILS')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)


def get_s3(
    endpoint_url='http://data.digital-pathology.zc2.ibm.com:9000',
    aws_access_key_id=None,
    aws_secret_access_key=None
):

    if aws_access_key_id is None or aws_secret_access_key is None:
        try:
            aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
            aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        except KeyError:
            log.error(
                "No credentials for %s. Set Environment Variables",
                endpoint_url
            )

    s3 = boto3.resource(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    log.debug("S3 Connection established")

    return s3


def download_file_to_local(
    s3=None,
    bucket_name='test-data',
    s3file='test_wsi.svs',
    local_name='tmp.svs'
):

    if s3 is None:
        s3 = get_s3()
    try:
        with open(local_name, "wb") as file:
            filepath = os.path.abspath(file.name)
            dirname = os.path.dirname(file.name)
            s3.meta.client.download_fileobj(bucket_name, s3file, file)

    except Exception as error:
        log.error("%s could not be downloaded to %s", s3file, local_name)
        log.error(str(error))
        local_name = None

    return local_name


def save_local_file(local_file, s3=None, bucket_name=None, s3file=None):
    if s3 is None:
        s3 = get_s3()

    try:
        s3.meta.client.upload_file(
            Filename=local_file, Bucket=bucket_name, Key=s3file
        )
        log.debug(
            "{} saved to {}/{} as {}".format(
                local_file, s3, bucket_name, s3file
            )
        )
    except Exception as e:
        log.error(
            "Error saving {} to {}/{} as {}".format(
                local_file, s3, bucket_name, s3file
            )
        )
        log.error(str(e))


def download_s3_dataset(s3=None, bucket_name=None, pattern='', path='tmp/'):

    if s3 is None:
        s3 = get_s3()
    
    os.makedirs(path, exist_ok=True)
    if bucket_name is not None:
        bucket = s3.Bucket(bucket_name)

        file_generator = (
            s3_object for s3_object in bucket.objects.all()
            if s3_object.key.startswith(pattern)
        )

        for s3_object in file_generator:
            output_filename = os.path.join(path, s3_object.key)
            if not os.path.exists(output_filename):
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                download_file_to_local(
                    s3=s3,
                    bucket_name=bucket_name,
                    s3file=s3_object.key,
                    local_name=output_filename
                )
