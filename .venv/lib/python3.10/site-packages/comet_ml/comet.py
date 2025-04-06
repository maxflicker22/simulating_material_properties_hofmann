# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

"""
Author: Gideon Mendels

This module contains the main components of comet client side

"""
import abc
import logging
import os
import random
import shutil
import string
import tempfile
import threading
import time
from http import HTTPStatus
from os.path import basename, splitext
from queue import Empty, Queue
from typing import NamedTuple, Type

import requests
from requests import RequestException

from . import offline_utils
from ._reporting import ON_EXIT_DIDNT_FINISH_UPLOAD_SDK, ON_RECONNECTION_EVENT
from ._typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    List,
    OnMessagesBatchSentCallback,
    OnMessageSentCallback,
    Optional,
)
from .batch_utils import MessageBatch, MessageBatchItem, ParametersBatch
from .config import (
    ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
    ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_INTERVAL_SECONDS,
    ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_MAX_BATCH_SIZE,
    DEFAULT_FILE_UPLOAD_READ_TIMEOUT,
    DEFAULT_STREAMER_MSG_TIMEOUT,
    DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL,
    MESSAGE_BATCH_METRIC_INTERVAL_SECONDS,
    MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE,
    MESSAGE_BATCH_PARAMETERS_INTERVAL_SECONDS,
    MESSAGE_BATCH_STDOUT_INTERVAL_SECONDS,
    MESSAGE_BATCH_STDOUT_MAX_BATCH_SIZE,
    MESSAGE_BATCH_USE_COMPRESSION_DEFAULT,
    OFFLINE_EXPERIMENT_MESSAGES_JSON_FILE_NAME,
    S3_MULTIPART_EXPIRES_IN,
    S3_MULTIPART_SIZE_THRESHOLD_DEFAULT,
    Config,
)
from .connection import RestApiClient, RestServerConnection
from .connection.connection_monitor import ConnectionStatus, ServerConnectionMonitor
from .constants import RESUME_STRATEGY_CREATE
from .convert_utils import convert_dict_to_string
from .exceptions import CometRestApiException
from .file_upload_manager import FileUploadManager, FileUploadManagerMonitor
from .logging_extensions.filters import (
    GENERAL_THROTTLING_MSG_EXTRA,
    METRICS_THROTTLING_MSG_EXTRA,
)
from .logging_messages import (
    ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_REST_ERROR,
    ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_UNKNOWN_ERROR,
    CLOUD_DETAILS_MSG_SENDING_ERROR,
    EXPERIMENT_MESSAGE_QUEUE_FLUSH_PROMPT,
    FAILED_TO_FLUSH_ARTIFACT_REMOTE_ASSETS_BATCH,
    FAILED_TO_FLUSH_METRICS_BATCH,
    FAILED_TO_FLUSH_PARAMETERS_BATCH,
    FAILED_TO_FLUSH_STDOUT_BATCH,
    FAILED_TO_REGISTER_MODEL,
    FAILED_TO_SEND_CLOUD_DETAILS_MESSAGE_ERROR,
    FAILED_TO_SEND_FILE_NAME_MESSAGE_ERROR,
    FAILED_TO_SEND_GIT_METADATA_MESSAGE_ERROR,
    FAILED_TO_SEND_GPU_STATIC_MESSAGE_ERROR,
    FAILED_TO_SEND_HTML_MESSAGE_ERROR,
    FAILED_TO_SEND_HTML_OVERRIDE_MESSAGE_ERROR,
    FAILED_TO_SEND_INSTALLED_PACKAGES_MESSAGE_ERROR,
    FAILED_TO_SEND_LOG_DEPENDENCY_MESSAGE_ERROR,
    FAILED_TO_SEND_LOG_OTHER_MESSAGE_ERROR,
    FAILED_TO_SEND_LOG_REMOTE_MODEL_MESSAGE_ERROR,
    FAILED_TO_SEND_MODEL_GRAPH_MESSAGE_ERROR,
    FAILED_TO_SEND_OS_PACKAGES_MESSAGE_ERROR,
    FAILED_TO_SEND_SYSTEM_DETAILS_MESSAGE_ERROR,
    FAILED_TO_SEND_SYSTEM_INFO_MESSAGE_ERROR,
    FAILED_TTO_ADD_MESSAGE_TO_THE_PARAMETERS_BATCH,
    FALLBACK_STREAMER_ARCHIVE_UPLOAD_MESSAGE_KEEP_ENABLED,
    FALLBACK_STREAMER_FAILED_NO_CONNECTION_NO_OFFLINE,
    FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE,
    FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE_ERROR,
    FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_STREAMER_WARNING,
    FALLBACK_STREAMER_FAILED_TO_WRITE_OFFLINE_DATA_ERROR,
    FALLBACK_STREAMER_ONLINE_FAILED_ARCHIVE_UPLOAD_MESSAGE,
    FILE_UPLOADS_PROMPT,
    FILENAME_DETAILS_MSG_SENDING_ERROR,
    GIT_METADATA_MSG_SENDING_ERROR,
    GPU_STATIC_INFO_MSG_SENDING_ERROR,
    HTML_MSG_SENDING_ERROR,
    HTML_OVERRIDE_MSG_SENDING_ERROR,
    INSTALLED_PACKAGES_MSG_SENDING_ERROR,
    LOG_DEPENDENCY_MESSAGE_SENDING_ERROR,
    LOG_OTHER_MSG_SENDING_ERROR,
    MESSAGES_THROTTLED_BY_BACKEND_ERROR,
    METRICS_BATCH_MSG_SENDING_ERROR,
    MODEL_GRAPH_MSG_SENDING_ERROR,
    OFFLINE_SENDER_FAILED_TO_WRITE_ALL_DATA,
    OFFLINE_SENDER_REMAINING_DATA_ITEMS_TO_WRITE,
    OFFLINE_SENDER_WAIT_FOR_FINISH_PROMPT,
    OS_PACKAGE_MSG_SENDING_ERROR,
    PARAMETERS_BATCH_MSG_SENDING_ERROR,
    REGISTER_FAILED_DUE_TO_UPLOADS_FAILED,
    REMOTE_MODEL_MESSAGE_SENDING_ERROR,
    STANDARD_OUTPUT_SENDING_ERROR,
    STREAMER_CLOSED_PUT_MESSAGE_FAILED,
    STREAMER_FAILED_TO_PROCESS_ALL_MESSAGES,
    STREAMER_FAILED_TO_REMOVE_TMP_FILE_COPY_WARNING,
    STREAMER_FLUSH_DIDNT_COMPLETE_SUCCESSFULLY_INFO,
    STREAMER_PROGRESS_MESSAGE_INFO,
    STREAMER_UNKNOWN_MESSAGE_TYPE_EXCEPTION,
    STREAMER_WAIT_FOR_FINISH_FAILED,
    SYSTEM_DETAILS_MSG_SENDING_ERROR,
    SYSTEM_INFO_MESSAGE_SENDING_ERROR,
    UNEXPECTED_OFFLINE_STREAMER_ERROR,
    UNEXPECTED_STREAMING_ERROR,
)
from .messages import (
    BaseMessage,
    CloseMessage,
    CloudDetailsMessage,
    FileNameMessage,
    GitMetadataMessage,
    GpuStaticInfoMessage,
    HtmlMessage,
    HtmlOverrideMessage,
    InstalledPackagesMessage,
    Log3DCloudMessage,
    LogDependencyMessage,
    LogOtherMessage,
    MetricMessage,
    ModelGraphMessage,
    OsPackagesMessage,
    ParameterMessage,
    RegisterModelMessage,
    RemoteAssetMessage,
    RemoteModelMessage,
    StandardOutputMessage,
    SystemDetailsMessage,
    SystemInfoMessage,
    UploadFileMessage,
    UploadInMemoryMessage,
)
from .messages_utils import convert_upload_in_memory_to_file_message
from .offline_utils import get_offline_data_dir_path
from .replay_manager import MessageStatus, ReplayManager
from .s3.multipart_upload.multipart_upload_options import MultipartUploadOptions
from .s3.multipart_upload.upload_error import S3UploadError
from .upload_callback.callback import UploadCallback
from .upload_options import (
    AssetItemUploadOptions,
    FileLikeUploadOptions,
    FileUploadOptions,
    RemoteAssetsUploadOptions,
    ThumbnailUploadOptions,
)
from .utils import compact_json_dump, local_timestamp, log_once_at_level, wait_for_done

LOGGER = logging.getLogger(__name__)


class RestApiSendResult(NamedTuple):
    success: bool
    connection_error: bool


class BaseStreamer(threading.Thread):
    __metaclass__ = abc.ABCMeta

    def __init__(self, initial_offset: int, queue_timeout: float) -> None:
        threading.Thread.__init__(self)

        self._counter = initial_offset
        self.messages = Queue()
        self.queue_timeout = queue_timeout
        self.closed = False
        self.__lock_closed__ = threading.RLock()

        LOGGER.debug("%r instantiated with duration %s", self, self.queue_timeout)

    def put_message_in_q(self, message: BaseMessage):
        """
        Puts a message in the queue
        :param message: Some kind of payload, type agnostic
        """
        with self.__lock_closed__:
            if message is not None:
                self._counter += 1
                if message.message_id is None:
                    # update message_id only if it was not already set
                    message.message_id = self._counter

                if not self.closed:
                    self.messages.put(message)
                else:
                    LOGGER.debug(STREAMER_CLOSED_PUT_MESSAGE_FAILED)
                    LOGGER.debug("Ignored message (streamer closed): %s", message)

    def close(self) -> None:
        """
        Puts a None in the queue which leads to closing it.
        """
        with self.__lock_closed__:
            if self.closed is True:
                LOGGER.debug("Streamer tried to be closed more than once: %r", self)
                return

            # Send a message to close
            self.put_message_in_q(CloseMessage())

            self.closed = True

            LOGGER.debug("Streamer %r, closed: %r", self, self.closed)

    def _before_run(self) -> None:
        pass

    def run(self) -> None:
        """
        Continuously pulls messages from the queue and process them.
        """
        self._before_run()

        while True:
            out = self._loop()

            # Exit the infinite loop
            if out is not None and isinstance(out, CloseMessage):
                break

        self._after_run()

        LOGGER.debug(
            "%r has finished RUN loop, closed: %r", self.__class__, self.closed
        )

    @abc.abstractmethod
    def wait_for_finish(self, **kwargs) -> bool:
        pass

    @abc.abstractmethod
    def has_connection_to_server(self) -> bool:
        pass

    @abc.abstractmethod
    def flush(self) -> bool:
        pass

    @abc.abstractmethod
    def _loop(self) -> BaseMessage:
        pass

    @abc.abstractmethod
    def _report_experiment_error(self, message: str, has_crashed: bool = False) -> None:
        pass

    def _after_run(self):
        pass

    def getn(self, n: int) -> Optional[List[BaseMessage]]:
        """
        Pops n messages from the queue.
        Args:
            n: Number of messages to pull from queue

        Returns: n messages

        """
        try:
            msg = self.messages.get(
                timeout=self.queue_timeout
            )  # block until at least 1
        except Empty:
            LOGGER.debug("No message in queue, timeout")
            return None

        if isinstance(msg, CloseMessage):
            return [msg]

        result = [msg]
        try:
            while len(result) < n:
                another_msg = self.messages.get_nowait()
                result.append(another_msg)
        except Empty:
            LOGGER.debug("No more messages in queue, while getting more than 1 message")
        return result


class Streamer(BaseStreamer):
    """
    This class extends threading.Thread and provides a simple concurrent queue
    and an async service that pulls data from the queue and sends it to the server.
    """

    def __init__(
        self,
        beat_duration: float,
        connection: RestServerConnection,
        initial_offset: int,
        experiment_key: str,
        api_key: str,
        run_id: str,
        project_id: str,
        rest_api_client: RestApiClient,
        worker_cpu_ratio: int,
        worker_count: Optional[int],
        verify_tls: bool,
        msg_waiting_timeout: float = DEFAULT_STREAMER_MSG_TIMEOUT,
        file_upload_waiting_timeout: float = ADDITIONAL_STREAMER_UPLOAD_TIMEOUT,
        file_upload_read_timeout: float = DEFAULT_FILE_UPLOAD_READ_TIMEOUT,
        wait_for_finish_sleep_interval: float = DEFAULT_WAIT_FOR_FINISH_SLEEP_INTERVAL,
        parameters_batch_base_interval: float = MESSAGE_BATCH_PARAMETERS_INTERVAL_SECONDS,
        message_batch_compress: bool = MESSAGE_BATCH_USE_COMPRESSION_DEFAULT,
        message_batch_metric_interval: float = MESSAGE_BATCH_METRIC_INTERVAL_SECONDS,
        message_batch_metric_max_size: int = MESSAGE_BATCH_METRIC_MAX_BATCH_SIZE,
        message_batch_stdout_interval: float = MESSAGE_BATCH_STDOUT_INTERVAL_SECONDS,
        message_batch_stdout_max_size: int = MESSAGE_BATCH_STDOUT_MAX_BATCH_SIZE,
        s3_multipart_threshold: int = S3_MULTIPART_SIZE_THRESHOLD_DEFAULT,
        s3_multipart_expires_in: int = S3_MULTIPART_EXPIRES_IN,
        s3_multipart_upload_enabled: bool = False,
        artifact_remote_assets_batch_enabled: bool = False,
        artifact_remote_assets_batch_metric_interval: float = ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_INTERVAL_SECONDS,
        artifact_remote_assets_batch_metric_max_size: int = ARTIFACT_REMOTE_ASSETS_BATCH_METRIC_MAX_BATCH_SIZE,
        use_raw_throttling_messages: bool = True,
    ) -> None:
        super().__init__(
            initial_offset=initial_offset, queue_timeout=beat_duration / 1000.0
        )

        self.daemon = True
        self.name = "Streamer(experiment=%r)" % experiment_key

        self._connection = connection
        self._rest_api_client = rest_api_client

        self._stop_message_queue_processing = False

        self._msg_waiting_timeout = msg_waiting_timeout
        self._wait_for_finish_sleep_interval = wait_for_finish_sleep_interval
        self._file_upload_waiting_timeout = file_upload_waiting_timeout
        self._file_upload_read_timeout = file_upload_read_timeout

        self._file_upload_manager = FileUploadManager(
            worker_cpu_ratio=worker_cpu_ratio,
            worker_count=worker_count,
            s3_upload_options=MultipartUploadOptions(
                file_size_threshold=s3_multipart_threshold,
                upload_expires_in=s3_multipart_expires_in,
                direct_s3_upload_enabled=s3_multipart_upload_enabled,
            ),
        )
        self._file_uploads_to_clean = list()

        self.experiment_key = experiment_key
        self.api_key = api_key
        self.run_id = run_id
        self.project_id = project_id

        self.on_message_sent_callback = None
        self.on_messages_batch_sent_callback = None

        self._verify_tls = verify_tls

        self._parameters_batch = ParametersBatch(parameters_batch_base_interval)

        self._message_batch_compress = message_batch_compress
        self._message_batch_metrics = MessageBatch(
            base_interval=message_batch_metric_interval,
            max_size=message_batch_metric_max_size,
        )

        self._message_batch_stdout = MessageBatch(
            base_interval=message_batch_stdout_interval,
            max_size=message_batch_stdout_max_size,
        )

        self._artifact_remote_assets_batch_enabled = (
            artifact_remote_assets_batch_enabled
        )
        self._artifact_remote_assets_batch = MessageBatch(
            base_interval=artifact_remote_assets_batch_metric_interval,
            max_size=artifact_remote_assets_batch_metric_max_size,
        )
        self._has_artifact_remote_assets_batch_send_failed = False

        self._message_handlers: Optional[Dict[Type[BaseMessage], Callable]] = None

        self.use_raw_throttling_messages = use_raw_throttling_messages

        LOGGER.debug(
            "Streamer created with REST API url: %r, clientlib url: %r",
            self._rest_api_client.base_url,
            self._connection.server_address,
        )
        LOGGER.debug(
            "---------------------------------------------------------------------"
        )
        LOGGER.debug(
            "Parameters batch base interval: %s seconds", parameters_batch_base_interval
        )
        LOGGER.debug(
            "Metrics max batch size: %d, metrics batch interval: %s seconds",
            message_batch_metric_max_size,
            message_batch_metric_interval,
        )
        LOGGER.debug(
            "Stdout batch max size: %d, batch interval: %s seconds",
            message_batch_stdout_max_size,
            message_batch_stdout_interval,
        )
        LOGGER.debug(
            "Artifact remote assets batching enabled: %s, max batch size: %d, batch interval: %s seconds",
            artifact_remote_assets_batch_enabled,
            artifact_remote_assets_batch_metric_max_size,
            artifact_remote_assets_batch_metric_interval,
        )
        LOGGER.debug("Batches compression enabled: %s", message_batch_compress)
        LOGGER.debug(
            "---------------------------------------------------------------------"
        )

    def register_message_sent_callback(
        self, message_sent_callback: OnMessageSentCallback
    ):
        self.on_message_sent_callback = message_sent_callback

    def register_messages_batch_sent_callback(
        self, messages_batch_sent_callback: OnMessagesBatchSentCallback
    ):
        self.on_messages_batch_sent_callback = messages_batch_sent_callback

    def _register_message_handlers(self):
        self._message_handlers = {
            UploadFileMessage: self._process_upload_message,
            UploadInMemoryMessage: self._process_upload_in_memory_message,
            RemoteAssetMessage: self._process_upload_remote_asset_message,
            MetricMessage: self._process_metric_message,
            ParameterMessage: self._process_parameter_message,
            OsPackagesMessage: self._process_os_package_message,
            ModelGraphMessage: self._process_model_graph_message,
            SystemDetailsMessage: self._process_system_details_message,
            CloudDetailsMessage: self._process_cloud_details_message,
            LogOtherMessage: self._process_log_other_message,
            FileNameMessage: self._process_file_name_message,
            HtmlMessage: self._process_html_message,
            HtmlOverrideMessage: self._process_html_override_message,
            InstalledPackagesMessage: self._process_installed_packages_message,
            GpuStaticInfoMessage: self._process_gpu_static_info_message,
            GitMetadataMessage: self._process_git_metadata_message,
            SystemInfoMessage: self._process_system_info_message,
            LogDependencyMessage: self._process_log_dependency_message,
            StandardOutputMessage: self._process_standard_output_message,
            RegisterModelMessage: self._process_register_model_message,
            RemoteModelMessage: self._process_remote_model_message,
            Log3DCloudMessage: self._process_log_3d_cloud_message,
        }

    def _loop(self):
        """
        A single loop of running
        """
        try:
            # If we should stop processing the queue, abort early
            if self._stop_message_queue_processing is True:
                LOGGER.debug("Streamer: force close event loop")
                return CloseMessage()

            if self._message_handlers is None:
                self._register_message_handlers()

            messages = self.getn(1)

            if messages is not None:
                for message in messages:
                    if isinstance(message, CloseMessage):
                        LOGGER.debug(
                            "Streamer: closing event loop - CloseMessage found in the messages\n"
                        )
                        # force batches to flush
                        self._send_batches(flush=True)

                        return message

                    handler = self._message_handlers.get(type(message))
                    if handler is None:
                        raise ValueError(
                            STREAMER_UNKNOWN_MESSAGE_TYPE_EXCEPTION, message
                        )
                    handler(message)

            # send collected batch data
            self._send_batches(flush=False)
        except Exception:
            LOGGER.debug(UNEXPECTED_STREAMING_ERROR, exc_info=True)
            # report experiment error
            self._report_experiment_error(UNEXPECTED_STREAMING_ERROR)

    def _on_upload_failed_callback(
        self, message_id: int, message_callback: Optional[Callable] = None
    ):
        def _callback(response):
            if (
                isinstance(response, ConnectionError)
                or isinstance(response, requests.ConnectionError)
                or isinstance(response, S3UploadError)
            ):
                self._on_messages_sent(
                    [message_id],
                    success=False,
                    connection_error=True,
                    failure_reason=str(response),
                )

        if message_callback is None:
            callback = lambda response: _callback(response)  # noqa: E731
        else:
            callback = lambda response: (  # noqa: E731
                _callback(response),
                message_callback(response),
            )

        return callback

    def _on_upload_success_callback(
        self, message_id: int, message_callback: Optional[Callable] = None
    ):
        if message_callback is None:
            callback = lambda response: self._on_messages_sent(  # noqa: E731
                [message_id]
            )
        else:
            callback = lambda response: (  # noqa: E731
                self._on_messages_sent([message_id]),
                message_callback(response),
            )

        return callback

    def _process_upload_message(self, message: UploadFileMessage) -> None:
        # Compute the url from the upload type
        url = self._connection.get_upload_url(message.upload_type)

        self._file_upload_manager.upload_file_thread(
            options=FileUploadOptions(
                additional_params=message.additional_params,
                api_key=self.api_key,
                clean=False,  # do not clean immediately after upload - this would be handled later
                experiment_id=self.experiment_key,
                file_path=message.file_path,
                metadata=message.metadata,
                project_id=self.project_id,
                timeout=self._file_upload_read_timeout,
                verify_tls=self._verify_tls,
                upload_endpoint=url,
                on_asset_upload=self._on_upload_success_callback(
                    message_id=message.message_id,
                    message_callback=message._on_asset_upload,
                ),
                on_failed_asset_upload=self._on_upload_failed_callback(
                    message_id=message.message_id,
                    message_callback=message._on_failed_asset_upload,
                ),
                estimated_size=message._size,
                upload_type=message.upload_type,
                base_url=self._connection.server_address,
            ),
            critical=message._critical,
        )
        if message.clean is True:
            self._file_uploads_to_clean.append(message.file_path)

    def _process_upload_in_memory_message(self, message: UploadInMemoryMessage) -> None:
        # Compute the url from the upload type
        url = self._connection.get_upload_url(message.upload_type)

        self._file_upload_manager.upload_file_like_thread(
            options=FileLikeUploadOptions(
                additional_params=message.additional_params,
                api_key=self.api_key,
                experiment_id=self.experiment_key,
                file_like=message.file_like,
                metadata=message.metadata,
                project_id=self.project_id,
                timeout=self._file_upload_read_timeout,
                verify_tls=self._verify_tls,
                upload_endpoint=url,
                on_asset_upload=self._on_upload_success_callback(
                    message_id=message.message_id,
                    message_callback=message._on_asset_upload,
                ),
                on_failed_asset_upload=self._on_upload_failed_callback(
                    message_id=message.message_id,
                    message_callback=message._on_failed_asset_upload,
                ),
                estimated_size=message._size,
                upload_type=message.upload_type,
                base_url=self._connection.server_address,
            ),
            critical=message._critical,
        )
        LOGGER.debug("Processing in-memory uploading message done")

    def _process_upload_remote_asset_message(self, message: RemoteAssetMessage) -> None:
        if self._artifact_remote_assets_batch_enabled and message.is_artifact_asset():
            # add message to the batch
            self._artifact_remote_assets_batch.append(message)
        else:
            self._upload_remote_asset_message(message)

    def _upload_remote_asset_message(self, message: RemoteAssetMessage) -> None:
        # Compute the url from the upload type
        url = self._connection.get_upload_url(message.upload_type)

        self._file_upload_manager.upload_remote_asset_thread(
            options=RemoteAssetsUploadOptions(
                additional_params=message.additional_params,
                api_key=self.api_key,
                experiment_id=self.experiment_key,
                metadata=message.metadata,
                project_id=self.project_id,
                remote_uri=message.remote_uri,
                timeout=self._file_upload_read_timeout,
                verify_tls=self._verify_tls,
                upload_endpoint=url,
                on_asset_upload=self._on_upload_success_callback(
                    message_id=message.message_id,
                    message_callback=message._on_asset_upload,
                ),
                on_failed_asset_upload=self._on_upload_failed_callback(
                    message_id=message.message_id,
                    message_callback=message._on_failed_asset_upload,
                ),
                estimated_size=message._size,
            ),
            critical=message._critical,
        )
        LOGGER.debug(
            "Remote asset has been scheduled for upload, asset URI: %r",
            message.remote_uri,
        )

    def _send_artifact_remote_assets_batch(self, message_items: List[MessageBatchItem]):
        if len(message_items) == 0:
            LOGGER.debug("Empty artifact remote assets batch, send request ignored")
            return

        # check that we have only artifact remote assets in the batch
        is_artifact_assets = all(
            [
                isinstance(m.message, RemoteAssetMessage)
                and m.message.is_artifact_asset()
                for m in message_items
            ]
        )
        if not is_artifact_assets:
            LOGGER.debug("Only artifact remote assets should be in the batch")
            raise ValueError("Only artifact remote assets should be in the batch")

        send_batch_start = time.time()
        LOGGER.debug("Streamer: _send_artifact_remote_assets_batch started")

        messages_ids = [m.message.message_id for m in message_items]
        send_result = self._process_rest_api_send(
            sender=self._rest_api_client.send_artifact_remote_assets_batch,
            rest_fail_prompt=ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_REST_ERROR,
            general_fail_prompt=ARTIFACT_REMOTE_ASSETS_BATCH_SENDING_UNKNOWN_ERROR,
            messages_ids=messages_ids,
            batch_items=message_items,
            compress=self._message_batch_compress,
            experiment_key=self.experiment_key,
        )

        # notify callbacks
        for m in message_items:
            callbacks = m.message.get_message_callbacks()
            if send_result.success and callbacks.on_asset_upload is not None:
                callbacks.on_asset_upload()
            elif (
                not send_result.connection_error
                and callbacks.on_failed_asset_upload is not None
            ):
                # signal failure if not a connection error
                callbacks.on_failed_asset_upload()
                self._has_artifact_remote_assets_batch_send_failed = True

        elapsed = time.time() - send_batch_start
        LOGGER.debug(
            "Streamer: _send_artifact_remote_assets_batch completed, elapsed: %r",
            elapsed,
        )

    def _send_stdout_messages_batch(
        self, message_items: List[MessageBatchItem]
    ) -> None:
        send_batch_start = time.time()
        LOGGER.debug("Streamer: _send_stdout_messages_batch started")

        messages_ids = [m.message.message_id for m in message_items]
        self._process_rest_api_send(
            sender=self._rest_api_client.send_stdout_batch,
            rest_fail_prompt=STANDARD_OUTPUT_SENDING_ERROR,
            general_fail_prompt="Error sending stdout/stderr batch (online experiment)",
            messages_ids=messages_ids,
            batch_items=message_items,
            compress=self._message_batch_compress,
            experiment_key=self.experiment_key,
        )

        elapsed = time.time() - send_batch_start
        LOGGER.debug(
            "Streamer: _send_stdout_messages_batch completed, elapsed: %r", elapsed
        )

    def _send_metric_messages_batch(
        self, message_items: List[MessageBatchItem]
    ) -> None:
        send_batch_start = time.time()
        LOGGER.debug("Streamer: _send_metric_messages_batch started")

        messages_ids = [m.message.message_id for m in message_items]
        self._process_rest_api_send(
            sender=self._connection.log_metrics_batch,
            rest_fail_prompt=METRICS_BATCH_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending metrics batch (online experiment)",
            messages_ids=messages_ids,
            rest_error_log_record_extra=METRICS_THROTTLING_MSG_EXTRA,
            items=message_items,
            compress=self._message_batch_compress,
        )

        elapsed = time.time() - send_batch_start
        LOGGER.debug(
            "Streamer: _send_metric_messages_batch completed, elapsed: %r", elapsed
        )

    def _send_parameter_messages_batch(
        self, message_items: List[MessageBatchItem]
    ) -> None:
        send_batch_start = time.time()
        LOGGER.debug("Streamer: _send_parameter_messages_batch started")

        messages_ids = [m.message.message_id for m in message_items]
        self._process_rest_api_send(
            sender=self._connection.log_parameters_batch,
            rest_fail_prompt=PARAMETERS_BATCH_MSG_SENDING_ERROR,
            general_fail_prompt="Error sending parameters batch (online experiment)",
            messages_ids=messages_ids,
            items=message_items,
            compress=self._message_batch_compress,
        )

        elapsed = time.time() - send_batch_start
        LOGGER.debug(
            "Streamer: _send_parameter_messages_batch completed, elapsed: %r", elapsed
        )

    def _process_parameter_message(self, message: ParameterMessage) -> None:
        # add message to the parameters batch
        if not self._parameters_batch.append(message):
            LOGGER.warning(FAILED_TTO_ADD_MESSAGE_TO_THE_PARAMETERS_BATCH, message)
            # report experiment error
            self._report_experiment_error(
                FAILED_TTO_ADD_MESSAGE_TO_THE_PARAMETERS_BATCH % message
            )

    def _process_metric_message(self, message: MetricMessage) -> None:
        self._message_batch_metrics.append(message)

    def _process_rest_api_send(
        self,
        sender: Callable,
        rest_fail_prompt: str,
        general_fail_prompt: str,
        messages_ids: List[int],
        rest_error_log_record_extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RestApiSendResult:
        try:
            sender(**kwargs)
            # notify successful delivery
            self._on_messages_sent(messages_ids)
            return RestApiSendResult(success=True, connection_error=False)
        except (ConnectionError, requests.ConnectionError) as conn_err:
            # just log and do not report because there is no connection
            LOGGER.debug(general_fail_prompt, exc_info=True)
            self._on_messages_sent(
                messages_ids,
                success=False,
                connection_error=True,
                failure_reason=str(conn_err),
            )
            return RestApiSendResult(success=False, connection_error=True)
        except (CometRestApiException, RequestException) as exc:
            if exc.response is not None:
                if (
                    isinstance(exc, CometRestApiException)
                    and exc.response.status_code == HTTPStatus.TOO_MANY_REQUESTS
                ):
                    # create common throttling error message
                    if self.use_raw_throttling_messages:
                        msg = exc.safe_msg
                    else:
                        msg = MESSAGES_THROTTLED_BY_BACKEND_ERROR % exc.safe_msg
                else:
                    msg = rest_fail_prompt % (
                        exc.response.status_code,
                        exc.response.content,
                    )
            else:
                msg = rest_fail_prompt % (-1, str(exc))

            if exc.response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                # set extra into log record to allow filtering of records based on metadata
                if rest_error_log_record_extra is None:
                    rest_error_log_record_extra = GENERAL_THROTTLING_MSG_EXTRA
                LOGGER.error(msg, extra=rest_error_log_record_extra)
            else:
                LOGGER.error(msg, exc_info=True)

            # report experiment error
            self._report_experiment_error(msg)
        except Exception as exception:
            error_message = "%s. %s: %s" % (
                general_fail_prompt,
                exception.__class__.__name__,
                str(exception),
            )
            LOGGER.error(error_message, exc_info=True)
            # report experiment error
            self._report_experiment_error(error_message)

        return RestApiSendResult(success=False, connection_error=False)

    def _on_messages_sent(
        self,
        messages_ids: List[int],
        success: bool = True,
        connection_error: bool = False,
        failure_reason: Optional[str] = None,
    ):
        size = len(messages_ids)
        if size == 1 and self.on_message_sent_callback is not None:
            self.on_message_sent_callback(
                messages_ids[0], success, connection_error, failure_reason
            )
        elif size > 1 and self.on_messages_batch_sent_callback is not None:
            batch_start = time.time()
            LOGGER.debug("Streamer: batch - start on_messages_batch_sent_callback")

            self.on_messages_batch_sent_callback(
                messages_ids, success, connection_error, failure_reason
            )

            elapsed = time.time() - batch_start
            LOGGER.debug(
                "Streamer: batch - on_messages_batch_sent_callback completed, size: %d, elapsed: %r",
                len(messages_ids),
                elapsed,
            )

    def _process_os_package_message(self, message: OsPackagesMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_os_packages,
            rest_fail_prompt=OS_PACKAGE_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_OS_PACKAGES_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            os_packages=message.os_packages,
        )

    def _process_model_graph_message(self, message: ModelGraphMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_model_graph,
            rest_fail_prompt=MODEL_GRAPH_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_MODEL_GRAPH_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            graph_str=message.graph,
        )

    def _process_system_details_message(self, message: SystemDetailsMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_system_details,
            rest_fail_prompt=SYSTEM_DETAILS_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_SYSTEM_DETAILS_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            _os=message.os,
            command=message.command,
            env=message.env,
            experiment_key=self.experiment_key,
            hostname=message.hostname,
            ip=message.ip,
            machine=message.machine,
            os_release=message.os_release,
            os_type=message.os_type,
            pid=message.pid,
            processor=message.processor,
            python_exe=message.python_exe,
            python_version_verbose=message.python_version_verbose,
            python_version=message.python_version,
            user=message.user,
        )

    def _process_log_other_message(self, message: LogOtherMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_other,
            rest_fail_prompt=LOG_OTHER_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_LOG_OTHER_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            key=message.key,
            value=message.value,
        )

    def _process_cloud_details_message(self, message: CloudDetailsMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_cloud_details,
            rest_fail_prompt=CLOUD_DETAILS_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_CLOUD_DETAILS_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            provider=message.provider,
            cloud_metadata=message.cloud_metadata,
        )

    def _process_file_name_message(self, message: FileNameMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_filename,
            rest_fail_prompt=FILENAME_DETAILS_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_FILE_NAME_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            filename=message.file_name,
        )

    def _process_html_message(self, message: HtmlMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_html,
            rest_fail_prompt=HTML_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_HTML_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            html=message.html,
        )

    def _process_installed_packages_message(
        self, message: InstalledPackagesMessage
    ) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_installed_packages,
            rest_fail_prompt=INSTALLED_PACKAGES_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_INSTALLED_PACKAGES_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            installed_packages=message.installed_packages,
        )

    def _process_html_override_message(self, message: HtmlOverrideMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_html,
            rest_fail_prompt=HTML_OVERRIDE_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_HTML_OVERRIDE_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            html=message.htmlOverride,
            overwrite=True,
        )

    def _process_gpu_static_info_message(self, message: GpuStaticInfoMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_gpu_static_info,
            rest_fail_prompt=GPU_STATIC_INFO_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_GPU_STATIC_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            gpu_static_info=message.gpu_static_info,
        )

    def _process_git_metadata_message(self, message: GitMetadataMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.set_experiment_git_metadata,
            rest_fail_prompt=GIT_METADATA_MSG_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_GIT_METADATA_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            user=message.git_metadata.get("user"),
            root=message.git_metadata.get("root"),
            branch=message.git_metadata.get("branch"),
            parent=message.git_metadata.get("parent"),
            origin=message.git_metadata.get("origin"),
        )

    def _process_system_info_message(self, message: SystemInfoMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_system_info,
            rest_fail_prompt=SYSTEM_INFO_MESSAGE_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_SYSTEM_INFO_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            system_info=[message.system_info],
        )

    def _process_log_dependency_message(self, message: LogDependencyMessage) -> None:
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_dependency,
            rest_fail_prompt=LOG_DEPENDENCY_MESSAGE_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_LOG_DEPENDENCY_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            name=message.name,
            version=message.version,
            timestamp=message.local_timestamp,
        )

    def _process_log_3d_cloud_message(self, message: Log3DCloudMessage) -> None:
        asset_id = self._connection.create_asset(
            experiment_key=self.experiment_key,
            asset_name=message.name,
            asset_type=message.type,
            metadata=message.metadata,
            step=message.step,
        )

        thumbnail_upload_url = "%sasset/%s/thumbnail" % (
            self._connection.server_address,
            asset_id,
        )

        self._file_upload_manager.upload_thumbnail_thread(
            options=ThumbnailUploadOptions(
                api_key=self.api_key,
                experiment_id=self.experiment_key,
                project_id=self.project_id,
                timeout=self._file_upload_read_timeout,
                verify_tls=self._verify_tls,
                upload_endpoint=thumbnail_upload_url,
                estimated_size=0,
                thumbnail_path=message.thumbnail_path,
            ),
        )

        asset_item_url = self._connection.get_upload_url(message.upload_type)
        for item in message.items:
            self._file_upload_manager.upload_asset_item_thread(
                options=AssetItemUploadOptions(
                    api_key=self.api_key,
                    experiment_id=self.experiment_key,
                    project_id=self.project_id,
                    timeout=self._file_upload_read_timeout,
                    verify_tls=self._verify_tls,
                    upload_endpoint=asset_item_url,
                    estimated_size=0,
                    asset_id=asset_id,
                    asset_item=item,
                ),
            )

    def _process_standard_output_message(self, message: StandardOutputMessage) -> None:
        self._message_batch_stdout.append(message)

    def has_connection_to_server(self):
        return True

    def is_message_loop_active(self):
        with self.__lock_closed__:
            return not self.closed and not self._stop_message_queue_processing

    def _process_register_model_message(self, message: RegisterModelMessage) -> None:
        try:
            status = message.upload_status_observer_callback()
            if status == "IN_PROGRESS":
                if self.is_message_loop_active():
                    self.put_message_in_q(message)
                    return

                # message loop not active - force wait for upload complete
                LOGGER.debug(
                    "Message loop is not active! Force wait for model %r upload to complete.",
                    message.model_name,
                )
                while status == "IN_PROGRESS":
                    time.sleep(0.5)
                    status = message.upload_status_observer_callback()

            LOGGER.debug(
                "Model %r upload complete with status: %r", message.model_name, status
            )
            if status == "FAILED":
                LOGGER.error(REGISTER_FAILED_DUE_TO_UPLOADS_FAILED, message.model_name)
                return

            workspace = (
                message.workspace
                or self._rest_api_client.get_experiment_metadata(message.experiment_id)[
                    "workspaceName"
                ]
            )

            LOGGER.debug(
                "Trying to register model %r with registry name %r and version %r",
                message.model_name,
                message.registry_name,
                message.version,
            )
            self._rest_api_client.register_model_v2(
                message.experiment_id,
                message.model_name,
                message.version,
                workspace,
                message.registry_name,
                message.public,
                message.description,
                message.comment,
                message.tags,
                message.status,
                message.stages,
            )
            message.on_model_register()
            # notify message tracker
            self.on_message_sent_callback(message.message_id, True, False, None)
        except CometRestApiException as exception:
            error_message = "{} {}".format(FAILED_TO_REGISTER_MODEL, exception.safe_msg)
            LOGGER.error(error_message)
            self._report_experiment_error(error_message)
            message.on_failed_model_register()
        except (ConnectionError, requests.ConnectionError) as conn_err:
            LOGGER.debug("Failed to register model - connection failure", exc_info=True)
            self.on_message_sent_callback(
                message.message_id, False, True, str(conn_err)
            )
        except Exception as exception:
            error_message = "{} {}".format(FAILED_TO_REGISTER_MODEL, str(exception))
            LOGGER.error(error_message)
            self._report_experiment_error(error_message)
            message.on_failed_model_register()

    def _process_remote_model_message(self, message: RemoteModelMessage):
        self._process_rest_api_send(
            sender=self._rest_api_client.log_experiment_remote_model,
            rest_fail_prompt=REMOTE_MODEL_MESSAGE_SENDING_ERROR,
            general_fail_prompt=FAILED_TO_SEND_LOG_REMOTE_MODEL_MESSAGE_ERROR,
            messages_ids=[message.message_id],
            experiment_key=self.experiment_key,
            model_name=message.model_name,
            remote_assets=message.remote_assets,
            on_model_upload=message.on_model_upload,
            on_failed_model_upload=message.on_failed_model_upload,
        )

    def _flush_message_queue(self, show_all_prompts=True):
        mq_flush_start = time.time()
        LOGGER.debug("Streamer: start flushing messages queue")

        success = True
        if not self._is_msg_queue_empty():
            if show_all_prompts:
                log_once_at_level(
                    logging.INFO,
                    EXPERIMENT_MESSAGE_QUEUE_FLUSH_PROMPT,
                    self._msg_waiting_timeout,
                )

            wait_for_done(
                self._is_msg_queue_empty,
                self._msg_waiting_timeout,
                progress_callback=self._show_remaining_messages,
                sleep_time=self._wait_for_finish_sleep_interval,
            )
            success = self._is_msg_queue_empty()

        elapsed = time.time() - mq_flush_start
        LOGGER.debug(
            "Streamer: flushing messages queue completed, successful: %s, elapsed: %r",
            success,
            elapsed,
        )

        if not success:
            LOGGER.warning(STREAMER_FAILED_TO_PROCESS_ALL_MESSAGES)

        return success

    def _send_batches(self, flush=False) -> bool:
        batch_start_time = time.time()
        LOGGER.debug("Streamer: start send batches, flush: %s", flush)

        success = True
        if not self._parameters_batch.empty():
            if not self._parameters_batch.accept(
                self._send_parameter_messages_batch, unconditional=flush
            ):
                if flush:
                    # the batch was not sent while full batch flush is requested
                    success = False
                    LOGGER.error(FAILED_TO_FLUSH_PARAMETERS_BATCH)
                    self._report_experiment_error(FAILED_TO_FLUSH_PARAMETERS_BATCH)
            else:
                LOGGER.debug("Parameters batch has been sent")

        if not self._message_batch_metrics.empty():
            if not self._message_batch_metrics.accept(
                self._send_metric_messages_batch, unconditional=flush
            ):
                if flush:
                    success = False
                    LOGGER.error(FAILED_TO_FLUSH_METRICS_BATCH)
                    self._report_experiment_error(FAILED_TO_FLUSH_METRICS_BATCH)
            else:
                LOGGER.debug("Metrics batch has been sent")

        if not self._message_batch_stdout.empty():
            if not self._message_batch_stdout.accept(
                self._send_stdout_messages_batch, unconditional=flush
            ):
                if flush:
                    success = False
                    LOGGER.error(FAILED_TO_FLUSH_STDOUT_BATCH)
                    self._report_experiment_error(FAILED_TO_FLUSH_STDOUT_BATCH)
            else:
                LOGGER.debug("stdout/stderr batch has been sent")

        if self._artifact_remote_assets_batch_enabled:
            if not self._artifact_remote_assets_batch.empty():
                if not self._artifact_remote_assets_batch.accept(
                    self._send_artifact_remote_assets_batch, unconditional=flush
                ):
                    if flush:
                        success = False
                        LOGGER.error(FAILED_TO_FLUSH_ARTIFACT_REMOTE_ASSETS_BATCH)
                        self._report_experiment_error(
                            FAILED_TO_FLUSH_ARTIFACT_REMOTE_ASSETS_BATCH
                        )
                else:
                    LOGGER.debug("Artifact remote assets batch has been sent")

        elapsed = time.time() - batch_start_time
        LOGGER.debug(
            "Streamer: send batches completed, successful: %s, elapsed: %r, flush: %s",
            success,
            elapsed,
            flush,
        )

        return success

    def _flush_file_upload_manager(self, show_all_prompts=True):
        fu_start_time = time.time()
        LOGGER.debug("Streamer: start flushing file upload manager")

        success = True
        if not self._file_upload_manager.all_done():
            monitor = FileUploadManagerMonitor(self._file_upload_manager)
            if show_all_prompts:
                LOGGER.info(
                    FILE_UPLOADS_PROMPT,
                    self._file_upload_waiting_timeout,
                )

            wait_for_done(
                monitor.all_done,
                self._file_upload_waiting_timeout,
                progress_callback=monitor.log_remaining_uploads,
                sleep_time=self._wait_for_finish_sleep_interval,
            )

            success = self._file_upload_manager.all_done()

        elapsed = time.time() - fu_start_time
        LOGGER.debug(
            "Streamer: flushing file upload manager completed, successful: %s, elapsed: %r",
            success,
            elapsed,
        )

        return success

    def flush(self) -> bool:
        """Flushes all pending data but do not close any threads.
        This method can be invoked multiple times during experiment lifetime."""

        LOGGER.debug("Start flushing all pending data to Comet")

        message_queue_flushed = self._flush_message_queue(show_all_prompts=False)
        batches_flushed = self._send_batches(flush=True)
        uploads_flushed = self._flush_file_upload_manager(show_all_prompts=False)

        if not (message_queue_flushed and batches_flushed and uploads_flushed):
            LOGGER.info(
                STREAMER_FLUSH_DIDNT_COMPLETE_SUCCESSFULLY_INFO
                % (message_queue_flushed, batches_flushed, uploads_flushed)
            )

        return message_queue_flushed & batches_flushed & uploads_flushed

    def wait_for_finish(self, **kwargs):
        """Blocks the experiment from exiting until all data was sent to server
        OR the configured timeouts has expired."""
        # We need to wait for online streamer to be closed first.
        # The streamer closed in asynchronous manner to allow all pending messages to be logged before closing.
        wf_start_time = time.time()
        LOGGER.debug("Streamer: started wait_for_finish")

        wait_for_done(
            lambda: self.closed, timeout=self._msg_waiting_timeout, sleep_time=0.5
        )

        self._flush_message_queue()

        # stop message processing only after message queue flushed to give thread loop a chance to go through
        # all accumulated messages in _flush_message_queue() - loop process one message in a time
        self._stop_message_queue_processing = True

        self._send_batches(flush=True)

        self._file_upload_manager.close()
        self._flush_file_upload_manager()

        if not self._is_msg_queue_empty() or not self._file_upload_manager.all_done():
            remaining = self.messages.qsize()
            remaining_upload = self._file_upload_manager.remaining_uploads()
            error_message = STREAMER_WAIT_FOR_FINISH_FAILED % (
                remaining,
                remaining_upload,
                self.experiment_key,
            )
            LOGGER.error(error_message)
            # report experiment error
            self._report_experiment_error(error_message)

            self._connection.report(
                event_name=ON_EXIT_DIDNT_FINISH_UPLOAD_SDK, err_msg=error_message
            )

            return False

        self._file_upload_manager.join()

        elapsed = time.time() - wf_start_time
        LOGGER.debug(
            "Streamer: wait_for_finish completed successfully, elapsed: %r", elapsed
        )

        return True

    def _clean_file_uploads(self):
        for file_path in self._file_uploads_to_clean:
            LOGGER.debug("Removing temporary copy of the uploaded file: %r", file_path)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                LOGGER.warning(
                    STREAMER_FAILED_TO_REMOVE_TMP_FILE_COPY_WARNING,
                    file_path,
                    exc_info=True,
                )

    def _report_experiment_error(self, message: str, has_crashed: bool = False):
        try:
            self._rest_api_client.update_experiment_error_status(
                experiment_key=self.experiment_key,
                is_alive=True,
                error_value=message,
                has_crashed=has_crashed,
            )
        except (ConnectionError, RequestException):
            LOGGER.debug("Failed to report experiment error", exc_info=True)
        except Exception as ex:
            LOGGER.debug("Failed to report experiment error, %r", ex, exc_info=True)

    def _is_msg_queue_empty(self):
        finished = self.messages.empty()

        if finished is False:
            LOGGER.debug(
                "Streamer: Messages queue not empty, %d messages, streamer closed: %s",
                self.messages.qsize(),
                self.closed,
            )

        return finished

    def _show_remaining_messages(self) -> None:
        remaining = self.messages.qsize()
        LOGGER.info(STREAMER_PROGRESS_MESSAGE_INFO, remaining)

    def has_upload_failed(self) -> bool:
        return (
            self._file_upload_manager.has_failed()
            or self._has_artifact_remote_assets_batch_send_failed
        )

    @property
    def has_artifact_remote_assets_batch_send_failed(self) -> bool:
        return self._has_artifact_remote_assets_batch_send_failed


class OfflineStreamer(BaseStreamer):
    """
    This class extends threading.Thread and provides a simple concurrent queue
    and an async service that pulls data from the queue and writes it to the file.
    """

    def __init__(
        self,
        tmp_dir: AnyStr,
        wait_timeout: int,
        on_error_callback: Callable[[str], None] = None,
        fallback_mode: bool = False,
        experiment_key: Optional[str] = None,
    ) -> None:
        super().__init__(initial_offset=0, queue_timeout=1)

        self.daemon = True
        if experiment_key is not None:
            self.name = "OfflineStreamer(experiment=%r)" % experiment_key

        self.tmp_dir = tmp_dir
        self.wait_timeout = wait_timeout
        self.on_error = on_error_callback
        self.fallback_mode = fallback_mode

        self.abort_processing = False
        self.__write_lock__ = threading.RLock()

        self._message_handlers: Optional[Dict[Type[BaseMessage], Callable]] = None

        self.file = open(
            os.path.join(self.tmp_dir, OFFLINE_EXPERIMENT_MESSAGES_JSON_FILE_NAME), "w"
        )

    def _write(self, json_line_message: Dict[str, Any]) -> None:
        if self.abort_processing:
            # offline streamer was aborted
            return

        with self.__write_lock__:
            compact_json_dump(json_line_message, self.file)
            self.file.write("\n")
            self.file.flush()

    def _after_run(self):
        # Close the messages files once we are sure we won't write in it
        # anymore
        self.file.close()

    def _register_message_handlers(self):
        self._message_handlers = {
            UploadFileMessage: self._process_upload_message,
            UploadInMemoryMessage: self._process_upload_in_memory_message,
            RemoteAssetMessage: self._process_message,
            MetricMessage: self._process_message,
            ParameterMessage: self._process_message,
            OsPackagesMessage: self._process_message,
            ModelGraphMessage: self._process_message,
            SystemDetailsMessage: self._process_message,
            CloudDetailsMessage: self._process_message,
            FileNameMessage: self._process_message,
            HtmlMessage: self._process_message,
            LogOtherMessage: self._process_message,
            HtmlOverrideMessage: self._process_message,
            InstalledPackagesMessage: self._process_message,
            GpuStaticInfoMessage: self._process_message,
            GitMetadataMessage: self._process_message,
            SystemInfoMessage: self._process_message,
            StandardOutputMessage: self._process_message,
            LogDependencyMessage: self._process_message,
            RegisterModelMessage: self._process_message,
            RemoteModelMessage: self._process_message,
            Log3DCloudMessage: self._process_message,
        }

    def _loop(self):
        """
        A single loop of running
        """
        if self.abort_processing is True:
            # force stop the thread run()
            return CloseMessage()

        if self._message_handlers is None:
            self._register_message_handlers()

        try:
            messages = self.getn(1)

            if messages is not None:
                LOGGER.debug(
                    "OfflineStreamer: Got %d messages, %d still in queue",
                    len(messages),
                    self.messages.qsize(),
                )

                for message in messages:
                    if isinstance(message, CloseMessage):
                        return message
                    elif self.abort_processing is True:
                        # ignore all messages to empty the queue
                        continue

                    handler = self._message_handlers.get(type(message))

                    if handler is None:
                        raise ValueError(
                            STREAMER_UNKNOWN_MESSAGE_TYPE_EXCEPTION, message
                        )
                    else:
                        handler(message)
        except Exception as ex:
            LOGGER.warning(UNEXPECTED_OFFLINE_STREAMER_ERROR, ex, exc_info=True)
            self._report_experiment_error(UNEXPECTED_OFFLINE_STREAMER_ERROR % ex)

    def _report_experiment_error(self, message: str, has_crashed: bool = False):
        if self.on_error is not None:
            self.on_error(message)

    def _process_upload_message(self, message: UploadFileMessage) -> None:
        # Create the file on disk with the same extension if set
        ext = splitext(message.file_path)[1]

        if ext:
            suffix = ".%s" % ext
        else:
            suffix = ""

        tmpfile = tempfile.NamedTemporaryFile(
            dir=self.tmp_dir, suffix=suffix, delete=False
        )
        tmpfile.close()

        # do not remove original file in fallback mode, i.e., when offline streamer is just a backup for online streamer
        if self.fallback_mode:
            # fallback mode
            if message.clean:
                # just copy to keep it after message would be cleaned by online streamer
                shutil.copy(message.file_path, tmpfile.name)
        else:
            # Offline mode
            if message.clean:
                # Clean by moving the original file to the newly created file
                shutil.move(message.file_path, tmpfile.name)
            else:
                # copy to our tmp dir
                shutil.copy(message.file_path, tmpfile.name)

        msg_json = message.to_message_dict()
        # update file_path directly in JSON representation to avoid side effects
        msg_json["file_path"] = basename(tmpfile.name)
        data = {"type": UploadFileMessage.type, "payload": msg_json}
        self._write(data)

    def _process_upload_in_memory_message(self, message: UploadInMemoryMessage) -> None:
        # We need to convert the in-memory file to a file one
        new_message = convert_upload_in_memory_to_file_message(message, self.tmp_dir)
        self._process_upload_message(new_message)

    def _process_message(self, message: BaseMessage):
        msg_json = message.to_message_dict()

        data = {"type": message.type, "payload": msg_json}
        self._write(data)

    def abort_and_wait(self, timeout: float = 10, _join: bool = True) -> None:
        """Aborts streamer by forcing immediate drop in processing of all scheduled messages. The currently
        processed message can still be written to the disk. If there is a message currently in process of writing
        to the disk, the invoking thread will be blocked until message data is fully written. Otherwise, this method
        waits for offline streamer's thread to terminate and return."""
        if not self.is_alive():
            return

        self.abort_processing = True
        if self.__write_lock__.acquire(timeout=timeout):
            # make sure to wait for current write operation (if any in progress) to complete within given timeout
            self.__write_lock__.release()

        if _join is True:
            self.join(timeout)

    def has_connection_to_server(self):
        return False

    def flush(self):
        return True

    def wait_for_finish(self, timeout: float = 10, _join: bool = True) -> bool:
        """Blocks the experiment from exiting until all data is saved or timeout exceeded."""
        if not self.is_alive():
            # already finished
            return True

        log_once_at_level(
            logging.INFO,
            OFFLINE_SENDER_WAIT_FOR_FINISH_PROMPT,
            int(self.wait_timeout * 2),
        )

        # Wait maximum for 2 times of self.wait_timeout
        wait_for_done(check_function=self.messages.empty, timeout=self.wait_timeout)

        if not self.messages.empty():
            LOGGER.info(OFFLINE_SENDER_WAIT_FOR_FINISH_PROMPT, int(self.wait_timeout))

            def progress_callback():
                LOGGER.info(
                    OFFLINE_SENDER_REMAINING_DATA_ITEMS_TO_WRITE, self.messages.qsize()
                )

            if not self.messages.empty():
                wait_for_done(
                    check_function=self.messages.empty,
                    timeout=self.wait_timeout,
                    progress_callback=progress_callback,
                    sleep_time=5,
                )

        if not self.messages.empty():
            remaining = self.messages.qsize()
            LOGGER.error(OFFLINE_SENDER_FAILED_TO_WRITE_ALL_DATA, remaining)
            self._report_experiment_error(
                OFFLINE_SENDER_FAILED_TO_WRITE_ALL_DATA % remaining
            )

            self.abort_and_wait(timeout)
            return False

        # Also wait for the thread to finish to be sure that all messages are
        # written to the messages file
        if _join is True:
            self.join(timeout)

        LOGGER.debug("OfflineStreamer finished in time")
        return True


class FallbackStreamer(BaseStreamer):
    """
    This class implements online streamer with support of fallback to offline streamer in case connection issues was
    detected during streaming.
    """

    def __init__(
        self,
        online_streamer: Streamer,
        server_connection_monitor: ServerConnectionMonitor,
        rest_server_connection: RestServerConnection,
        initial_offset: int,
        terminate_timeout: int = 10,
        offline_directory: Optional[str] = None,
        enable_fallback_to_offline: bool = True,
        keep_offline_zip: bool = False,  # allows to retain offline zip even without connection interruption
    ) -> None:
        super().__init__(initial_offset=initial_offset, queue_timeout=0.5)

        self.daemon = True
        self.name = "FallbackStreamer(experiment=%r)" % online_streamer.experiment_key

        self.online_streamer = online_streamer
        self.online_streamer.register_message_sent_callback(self._track_message)
        self.online_streamer.register_messages_batch_sent_callback(
            self._track_messages_batch
        )

        self.server_connection_monitor = server_connection_monitor
        self.rest_server_connection = rest_server_connection
        self.terminate_timeout = terminate_timeout
        self.offline_directory = offline_directory

        self.start_time = None
        self.stop_time = None

        self.offline_archive_file: Optional[str] = None
        self.offline_streamer = None
        self.temp_dir = None

        self.offline_zip_uploader: Optional[UploadCallback] = None

        self.enabled_fallback_to_offline = enable_fallback_to_offline
        self.keep_offline_zip = keep_offline_zip

        self.replay_manager = ReplayManager()

        self.queue_empty = threading.Condition()

        if self.enabled_fallback_to_offline:
            try:
                self._create_offline_streamer()
                self.offline_streamer_disabled_or_failed = False
            except Exception:
                LOGGER.warning(
                    FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_STREAMER_WARNING,
                    exc_info=True,
                )
                self.offline_streamer_disabled_or_failed = True
        else:
            self.offline_streamer_disabled_or_failed = True
            LOGGER.debug(
                "Skip creation of fallback offline streamer - disabled by configuration"
            )

    def parameters_update_interval_callback(self, parameters_update_interval):
        self.online_streamer._parameters_batch.update_interval(
            parameters_update_interval
        )

    def _create_offline_streamer(self):
        self.temp_dir = tempfile.mkdtemp()
        self.offline_streamer = OfflineStreamer(
            tmp_dir=self.temp_dir,
            wait_timeout=60,
            fallback_mode=True,
            experiment_key=self.online_streamer.experiment_key,
        )
        self.offline_streamer.on_error = self._offline_streamer_on_error_callback
        LOGGER.debug(
            "Offline streamer created with temporary data dir: %r", self.temp_dir
        )

    def _offline_streamer_on_error_callback(self, message: str) -> None:
        LOGGER.debug("Offline streamer failed with error: %s", message)
        self.offline_streamer_disabled_or_failed = True

    def _before_run(self):
        self.start_time = local_timestamp()
        LOGGER.debug("FallbackStreamer._before_run(), start time: %r", self.start_time)

    def _after_run(self):
        self.stop_time = local_timestamp()
        LOGGER.debug("FallbackStreamer._after_run(), stop time: %r", self.stop_time)

    def _loop(self) -> Optional[CloseMessage]:
        status = self.server_connection_monitor.tick(
            connection_probe=self._check_server_connection,
        )
        if status == ConnectionStatus.connection_restored:
            self._on_connection_restored()

        # process buffered messages if any
        self._process_buffered_messages()

        # notify waiting thread that queue already processed
        with self.queue_empty:
            self.queue_empty.notify()

        with self.__lock_closed__:
            if self.closed is True:
                # one more try to process pending messages before closing streamers
                self._process_buffered_messages()
                self._close_streamers()
                # force stop the thread run()
                return CloseMessage()

        return None

    def _process_buffered_messages(self):
        messages = []
        try:
            while True:
                message = self.messages.get(timeout=0.1)
                self._counter += 1
                message.message_id = self._counter
                messages.append(message)
        except Empty:
            pass
        if len(messages) > 0:
            self._process_queue_messages(messages)

    def _close_streamers(self):
        self.online_streamer.close()
        if self.offline_streamer is not None:
            self.offline_streamer.close()

    def _process_queue_messages(self, messages: List[BaseMessage]) -> None:
        connected = self.has_connection_to_server()
        status = MessageStatus.registered if connected else MessageStatus.failed
        # register message with replay manager
        self.replay_manager.register_messages(messages, status=status)

        for message in messages:
            if connected:
                self.online_streamer.put_message_in_q(message)

            if (
                self.offline_streamer is not None
                and self.offline_streamer_disabled_or_failed is False
            ):
                self.offline_streamer.put_message_in_q(message)

    def _check_server_connection(self) -> (bool, Optional[str]):
        try:
            self.rest_server_connection.ping_backend()
            return True, None
        except Exception as ex:
            LOGGER.debug("Backend ping failed, reason: %r", ex)
            return False, str(ex)

    def _replay_message(self, message: BaseMessage) -> None:
        self.online_streamer.put_message_in_q(message)

    def _on_connection_restored(self):
        # send BI event if appropriate
        if self.server_connection_monitor.disconnect_reason is not None:
            err_msg_dict = {
                "disconnect_reason": self.server_connection_monitor.disconnect_reason,
                "disconnect_time": self.server_connection_monitor.disconnect_time,
            }
            self.online_streamer._connection.report(
                event_name=ON_RECONNECTION_EVENT,
                err_msg=convert_dict_to_string(err_msg_dict),
            )
        self.server_connection_monitor.reset()

        # replay all failed messages
        self.replay_manager.replay_failed_messages(self._replay_message)

    def _track_message(
        self,
        message_id: int,
        delivered: bool,
        connection_error: bool,
        failure_reason: Optional[str] = None,
    ):
        if connection_error:
            self.server_connection_monitor.connection_failed(failure_reason)

        status = MessageStatus.delivered if delivered else MessageStatus.failed
        self.replay_manager.update_message(message_id=message_id, status=status)

    def _track_messages_batch(
        self,
        message_ids: List[int],
        delivered: bool,
        connection_error: bool,
        failure_reason: Optional[str] = None,
    ):
        if connection_error:
            self.server_connection_monitor.connection_failed(failure_reason)

        status = MessageStatus.delivered if delivered else MessageStatus.failed
        self.replay_manager.update_messages_batch(
            message_ids=message_ids, status=status
        )

    def _report_experiment_error(self, message: str, has_crashed: bool = False) -> None:
        if self.has_connection_to_server():
            self.online_streamer._report_experiment_error(
                message, has_crashed=has_crashed
            )

    def put_message_in_q(self, message) -> None:
        with self.__lock_closed__:
            if self.closed:
                return

            LOGGER.debug("Put message in queue: %r", message)
            self.messages.put(message, block=False)

    def has_upload_failed(self) -> bool:
        return self.online_streamer.has_upload_failed()

    def start(self) -> None:
        super().start()
        self.online_streamer.start()
        if self.offline_streamer is not None:
            self.offline_streamer.start()

    def close(self) -> None:
        with self.__lock_closed__:
            if self.closed is True:
                LOGGER.debug("FallbackStreamer tried to be closed more than once")
                return

            # mark as closed to block any new messages
            self.closed = True

        LOGGER.debug("Streamer %r, closed: %r", self, self.closed)

    def has_connection_to_server(self) -> bool:
        return self.server_connection_monitor.has_server_connection

    def flush(self) -> bool:
        """Flushes all pending data in online streamer.
        This method can be invoked multiple times during experiment lifetime."""
        with self.queue_empty:
            self.queue_empty.wait()
        return self.online_streamer.flush()

    def set_offline_zip_uploader(self, upload_callback: UploadCallback) -> None:
        self.offline_zip_uploader = upload_callback

    def wait_for_finish(
        self,
        experiment_key: str,
        workspace: str,
        project_name: str,
        tags: List[Any],
        comet_config: Config,
        _join: bool = True,  # this is for testing purposes only
    ) -> bool:
        fb_wait_start = time.time()
        LOGGER.debug(
            "FallbackStreamer: wait_for_finish started, experiment ID: %s",
            self.online_streamer.experiment_key,
        )

        online_finished_successfully = False
        if self.has_connection_to_server():
            online_finished_successfully = self.online_streamer.wait_for_finish()

        # Also wait for the current thread to finish
        if _join is True:
            self.join(self.terminate_timeout)

        # Close replay manager after online and this streamer complete its run
        self.replay_manager.close()

        if online_finished_successfully:
            create_offline_zip = self.keep_offline_zip
        elif not self.offline_streamer_disabled_or_failed:
            create_offline_zip = True
        else:
            # failed both modes - clean temporary directory
            self._clean_offline_data()
            LOGGER.error(FALLBACK_STREAMER_FAILED_NO_CONNECTION_NO_OFFLINE)
            return False

        finished_successfully = online_finished_successfully
        archive_created = False
        if create_offline_zip:
            LOGGER.debug(
                "FallbackStreamer: trying to create offline ZIP, experiment ID: %s",
                self.online_streamer.experiment_key,
            )
            # try to create offline archive as fallback
            archive_created = self._create_offline_archive_and_clean(
                experiment_key=experiment_key,
                workspace=workspace,
                project_name=project_name,
                tags=tags,
                comet_config=comet_config,
                _join=_join,
            )
            LOGGER.debug(
                "FallbackStreamer: offline ZIP created: %r, "
                "online streamer finished successfully: %r, experiment ID: %s",
                archive_created,
                online_finished_successfully,
                self.online_streamer.experiment_key,
            )

            if not online_finished_successfully:
                # offline mode - return flag indicating whether archive was created
                if archive_created:
                    # Display the full command to upload the offline experiment if archive was created
                    LOGGER.warning(
                        FALLBACK_STREAMER_ONLINE_FAILED_ARCHIVE_UPLOAD_MESSAGE,
                        self.offline_archive_file,
                    )
                    self._try_to_upload_offline_zip_file(self.offline_archive_file)
                finished_successfully = archive_created
            elif archive_created:
                # online finished successfully and archive retained - prompt user about having it
                LOGGER.info(
                    FALLBACK_STREAMER_ARCHIVE_UPLOAD_MESSAGE_KEEP_ENABLED,
                    self.offline_archive_file,
                )
        else:
            self._abort_offline_and_clean_offline(_join)

        elapsed = time.time() - fb_wait_start
        LOGGER.debug(
            "FallbackStreamer: wait_for_finish completed, success: %r, elapsed: %r, "
            "offline ZIP enabled: %r, offline ZIP created: %r, experiment ID: %s",
            finished_successfully,
            elapsed,
            create_offline_zip,
            archive_created,
            self.online_streamer.experiment_key,
        )

        return finished_successfully

    def _try_to_upload_offline_zip_file(self, offline_zip_file: str):
        if self.offline_zip_uploader is not None:
            offline_upload_start = time.time()
            LOGGER.debug(
                "FallbackStreamer: trying to upload offline ZIP file: %r",
                offline_zip_file,
            )
            try:
                self.offline_zip_uploader(offline_zip_file)

                elapsed = time.time() - offline_upload_start
                LOGGER.debug(
                    "FallbackStreamer: offline ZIP uploaded successfully, elapsed: %r",
                    elapsed,
                )
            except Exception as ex:
                LOGGER.error("Failed to upload offline ZIP file: %r", ex, exc_info=True)

    def _create_offline_archive_and_clean(
        self,
        experiment_key: str,
        workspace: str,
        project_name: str,
        tags: List[Any],
        comet_config: Config,
        _join: bool = True,
    ) -> bool:
        try:
            self._create_offline_archive(
                experiment_key=experiment_key,
                workspace=workspace,
                project_name=project_name,
                tags=tags,
                comet_config=comet_config,
                _join=_join,
            )
            return True
        except Exception:
            LOGGER.error(
                FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE, exc_info=True
            )
            return False
        finally:
            # make sure to clean up
            self._clean_offline_data()

    def _abort_offline_and_clean_offline(self, _join):
        if self.offline_streamer is not None:
            LOGGER.debug("Aborting offline streamer before cleaning")
            self.offline_streamer.abort_and_wait(self.terminate_timeout, _join=_join)

        self._clean_offline_data()

    def _clean_offline_data(self) -> None:
        if self.temp_dir is not None:
            try:
                LOGGER.debug(
                    "Cleaning collected offline data from dir: %r", self.temp_dir
                )
                shutil.rmtree(self.temp_dir)
            except Exception:
                LOGGER.debug(
                    "Failed to remove collected offline data from temporary directory: %r",
                    self.temp_dir,
                    exc_info=True,
                )

        LOGGER.debug(
            "Cleaning collected temporary files being uploaded by online streamer"
        )
        self.online_streamer._clean_file_uploads()

    def _create_offline_archive(
        self,
        experiment_key: str,
        workspace: str,
        project_name: str,
        tags: List[Any],
        comet_config: Config,
        _join: bool,
    ) -> None:
        if self.offline_streamer is None or self.offline_streamer_disabled_or_failed:
            LOGGER.error(FALLBACK_STREAMER_FAILED_TO_CREATE_OFFLINE_ARCHIVE_ERROR)
            return

        finished_successfully = self.offline_streamer.wait_for_finish(_join=_join)
        if not finished_successfully:
            LOGGER.error(FALLBACK_STREAMER_FAILED_TO_WRITE_OFFLINE_DATA_ERROR)

        # create offline experiment archive
        #
        offline_utils.write_experiment_meta_file(
            tempdir=self.temp_dir,
            experiment_key=experiment_key,
            workspace=workspace,
            project_name=project_name,
            tags=tags,
            start_time=self.start_time,
            stop_time=self.stop_time,
            resume_strategy=RESUME_STRATEGY_CREATE,
        )

        # adjust offline directory path if appropriate
        self.offline_directory, _ = get_offline_data_dir_path(
            comet_config=comet_config,
            offline_directory=self.offline_directory,
            logger=LOGGER,
        )

        # create offline archive into offline directory
        random_string = "".join(random.choice(string.ascii_letters) for _ in range(6))
        offline_archive_file_name = "%s-%s.zip" % (experiment_key, random_string)
        (
            self.offline_archive_file,
            self.offline_directory,
        ) = offline_utils.create_experiment_archive(
            offline_directory=self.offline_directory,
            offline_archive_file_name=offline_archive_file_name,
            data_dir=self.temp_dir,
            logger=LOGGER,
        )

    @property
    def counter(self) -> int:
        return self.online_streamer._counter

    @counter.setter
    def counter(self, new_value: int) -> None:
        self._counter = new_value

    @property
    def msg_waiting_timeout(self) -> float:
        return self.online_streamer._msg_waiting_timeout

    @property
    def file_upload_waiting_timeout(self) -> float:
        return self.online_streamer._file_upload_waiting_timeout

    @property
    def file_upload_read_timeout(self) -> float:
        return self.online_streamer._file_upload_read_timeout

    @property
    def message_batch_compress(self) -> bool:
        return self.online_streamer._message_batch_compress

    @message_batch_compress.setter
    def message_batch_compress(self, compress: bool) -> None:
        self.online_streamer._message_batch_compress = compress
