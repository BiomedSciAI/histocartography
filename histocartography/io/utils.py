"""Whole Slide Image IO module."""
import os
import logging
import sys
import boto3
import math

from ftplib import FTP

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


def open_ftp_connection(ftp_host):

    ftp_connection = FTP(ftp_host)
    ftp_connection.connect()
    ftp_connection.login()
    return ftp_connection


def transfer_file_from_ftp_to_s3(s3_connection,
                                 ftp_connection,
                                 bucket_name,
                                 ftp_file_path,
                                 s3_file_path):
    ftp_file_size = ftp_connection.size(ftp_file_path)

    try:
            s3_file = s3_connection.head_object(Bucket = bucket_name,
                                  Key = s3_file_path)
            if s3_file['ContentLength'] == ftp_file_size:
                    print('File Already Exists in S3 bucket')
                    return
    except Exception as e:
            pass

    print('Transferring File from FTP to S3 in chunks...')
    #upload file in chunks
    multipart_upload = s3_connection.create_multipart_upload(Bucket = bucket_name,
                                                             Key = s3_file_path)
    s3_up = s3_upload(
            s3_connection,
            multipart_upload,
            bucket_name,
            ftp_file_path,
            s3_file_path,
            ftp_file_size
    )
    ftp_connection.retrbinary(f'RETR {ftp_file_path}',
                              s3_up.cback)
    s3_up.finalize()
    part_info = {
            'Parts': s3_up.parts
    }
    s3_connection.complete_multipart_upload(
    Bucket = bucket_name,
    Key = s3_file_path,
    UploadId = multipart_upload['UploadId'],
    MultipartUpload = part_info
    )
    print('All chunks Transferred to S3 bucket! File Transfer successful!')

class s3_upload():
    def __init__(self,
                 s3_connection,
                 multipart_upload,
                 bucket_name,
                 ftp_file_path,
                 s3_file_path,
                 file_size):
        self.s3_connection = s3_connection
        self.multipart_upload = multipart_upload
        self.bucket_name = bucket_name
        self.ftp_file_path = ftp_file_path
        self.s3_file_path = s3_file_path
        self.part_number = 0
        self.parts = []
        self.buffer = bytes()
        self.file_size = file_size
        self.progress = 0

    def cback(self, chunk):
        self.buffer += chunk
	# make sure s3 uploaded chunks are bigger than 5MB
        if(len(self.buffer) > 5242880):
            self.part_number += 1
            part = self.s3_connection.upload_part(
                    Bucket = self.bucket_name,
                    Key = self.s3_file_path,
                    PartNumber = self.part_number,
                    UploadId = self.multipart_upload['UploadId'],
                    Body = self.buffer,
            )
            part_output = {
                    'PartNumber': self.part_number,
                    'ETag': part['ETag']
            }
            self.parts.append(part_output)
            self.progress += len(self.buffer)
            pct_done = self.progress / self.file_size * 100
            print('{:.2f}% Transfering chunk: {}'.format(
                pct_done, self.part_number), end='\r')
            self.buffer = bytes()
        #print('Chunk {} Transferred Successfully!'.format(self.part_number))
    
    def finalize(self):
        self.part_number += 1
        part = self.s3_connection.upload_part(
                Bucket = self.bucket_name,
                Key = self.s3_file_path,
                PartNumber = self.part_number,
                UploadId = self.multipart_upload['UploadId'],
                Body = self.buffer,
        )
        part_output = {
                'PartNumber': self.part_number,
                'ETag': part['ETag']
        }
        self.parts.append(part_output)
        self.buffer = bytes()


def get_s3(
    endpoint_url='',
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
