# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2024 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************

import json
import os
import tempfile
from collections import namedtuple
from logging import getLogger
from urllib.parse import urlparse

import semantic_version

from ._typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from .api import APIExperiment
from .artifact_helpers.artifact_assets_downloader import (
    download_artifact_asset,
    download_cloud_storage_artifact_asset,
)
from .assets.data_writers import AssetDataWriterFromGCS, AssetDataWriterFromS3
from .cloud_storage_utils import (
    META_ERROR_MESSAGE,
    META_FILE_SIZE,
    META_SYNCED,
    META_VERSION_ID,
)
from .config import Config, get_check_tls_certificate
from .connection import RestApiClient
from .constants import ASSET_TYPE_DEFAULT
from .debug import debug_helpers
from .exceptions import (
    ArtifactAssetNotFound,
    ArtifactConflictingAssetLogicalPath,
    ArtifactDownloadException,
    LogAssetException,
)
from .file_downloader import FileDownloadManager, FileDownloadManagerMonitor
from .file_uploader import (
    FileUpload,
    FolderUpload,
    MemoryFileUpload,
    PreprocessedAsset,
    PreprocessedAssetFolder,
    PreprocessedFileAsset,
    PreprocessedRemoteAsset,
    PreprocessedSyncedRemoteAsset,
    dispatch_user_file_upload,
    preprocess_asset_file,
    preprocess_asset_folder,
    preprocess_asset_memory_file,
    preprocess_remote_asset,
)
from .gs_bucket_info import preprocess_remote_gs_assets
from .logging_messages import (
    ARTIFACT_ASSET_DOWNLOAD_FAILED,
    ARTIFACT_ASSET_TYPE_DEPRECATED_WARNING,
    ARTIFACT_DOWNLOAD_CANNOT_DOWNLOAD_ASSET_EXCEPTION,
    ARTIFACT_DOWNLOAD_CANNOT_GET_ASSETS_LIST_EXCEPTION,
    ARTIFACT_DOWNLOAD_FINISHED,
    ARTIFACT_DOWNLOAD_START_MESSAGE,
    ARTIFACT_FAILED_TO_PARSE_REMOTE_ASSET_URI_WARNING,
    ARTIFACT_OVERWRITE_INVALID_STRATEGY_EXCEPTION,
    ASSET_DOWNLOAD_FAILED_WITH_ERROR,
    FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET,
    SYNC_MODE_IS_NOT_SUPPORTED_FOR_STRING_REMOTE_ARTIFACT,
    UNSUPPORTED_URI_SYNCED_REMOTE_ASSET,
)
from .s3_bucket_info import preprocess_remote_s3_assets
from .summary import Summary
from .utils import ImmutableDict, generate_guid, wait_for_done
from .validation.metadata_validator import validate_metadata

LOGGER = getLogger(__name__)


def _validate_overwrite_strategy(user_overwrite_strategy: Union[str, bool]) -> str:

    if isinstance(user_overwrite_strategy, str):
        lower_user_overwrite_strategy = user_overwrite_strategy.lower()
    else:
        lower_user_overwrite_strategy = user_overwrite_strategy

    if (
        lower_user_overwrite_strategy is False
        or lower_user_overwrite_strategy == "fail"
    ):
        return "FAIL"

    elif lower_user_overwrite_strategy == "preserve":
        return "PRESERVE"

    elif (
        lower_user_overwrite_strategy is True
        or lower_user_overwrite_strategy == "overwrite"
    ):
        return "OVERWRITE"

    else:
        raise ValueError(
            ARTIFACT_OVERWRITE_INVALID_STRATEGY_EXCEPTION % user_overwrite_strategy
        )


class ArtifactAsset(object):
    """ArtifactAsset(remote, logical_path, size, link, metadata, asset_type, local_path_or_data):
    represent local and remote assets added to an Artifact object but not yet uploaded
    """

    __slots__ = (
        "_remote",
        "_logical_path",
        "_size",
        "_link",
        "_metadata",
        "_asset_type",
        "_local_path_or_data",
    )

    def __init__(
        self,
        remote: bool,
        logical_path: str,
        size: int,
        link: Optional[str],
        metadata: Optional[Dict[Any, Any]],
        asset_type: Optional[str],
        local_path_or_data: Optional[Any],
    ) -> None:
        self._remote = remote
        self._logical_path = logical_path
        self._size = size
        self._link = link
        self._metadata = metadata
        self._asset_type = asset_type
        self._local_path_or_data = local_path_or_data

    @property
    def remote(self):
        """Is the asset a remote asset or not, boolean"""
        return self._remote

    @property
    def logical_path(self):
        """Asset relative logical_path, str or None"""
        return self._logical_path

    @property
    def size(self):
        """Asset size if the asset is a non-remote asset, int"""
        return self._size

    @property
    def link(self):
        """Asset remote link if the asset is remote, str or None"""
        return self._link

    @property
    def metadata(self):
        """Asset metadata, dict"""
        return self._metadata

    @property
    def asset_type(self):
        """Asset type, str"""
        return self._asset_type

    @property
    def local_path_or_data(self):
        """Asset local path or in-memory file if the asset is non-remote, str, memory-file or None"""
        return self._local_path_or_data

    def __repr__(self):
        return (
            "%s(remote=%r, logical_path=%r, size=%r, link=%r, metadata=%r, asset_type=%r, local_path_or_data=%r)"
            % (
                self.__class__.__name__,
                self._remote,
                self._logical_path,
                self._size,
                self._link,
                self._metadata,
                self._asset_type,
                self._local_path_or_data,
            )
        )

    def __eq__(self, other):
        return (
            self._remote == other._remote
            and self._logical_path == other._logical_path
            and self._size == other._size
            and self._link == other._link
            and self._metadata == other._metadata
            and self._asset_type == other._asset_type
            and self._local_path_or_data == other._local_path_or_data
        )

    def __lt__(self, other):
        return self._logical_path < other._logical_path


class LoggedArtifactAsset(object):
    """
    Represent assets logged to an Artifact
    """

    __slots__ = (
        "_remote",
        "_logical_path",
        "_size",
        "_link",
        "_metadata",
        "_asset_type",
        "_id",
        "_artifact_version_id",
        "_artifact_id",
        "_source_experiment_key",
        "_rest_api_client",
        "_download_timeout",
        "_logged_artifact_repr",
        "_logged_artifact_str",
        "_experiment_key",
        "_verify_tls",
    )

    def __init__(
        self,
        remote: bool,
        logical_path: str,
        size: int,
        link: str,
        metadata: Dict[str, Any],
        asset_type: str,
        id: str,
        artifact_version_id: str,
        artifact_id: str,
        source_experiment_key: str,
        verify_tls: bool,
        rest_api_client: RestApiClient = None,
        download_timeout: float = None,
        logged_artifact_repr: str = None,
        logged_artifact_str: str = None,
        experiment_key: str = None,
    ) -> None:
        self._remote = remote
        self._logical_path = logical_path
        self._size = size
        self._link = link
        self._metadata = metadata
        self._asset_type = asset_type
        self._id = id
        self._artifact_version_id = artifact_version_id
        self._artifact_id = artifact_id
        self._source_experiment_key = source_experiment_key

        self._rest_api_client = rest_api_client
        self._download_timeout = download_timeout
        self._verify_tls = verify_tls
        self._logged_artifact_repr = logged_artifact_repr
        self._logged_artifact_str = logged_artifact_str
        self._experiment_key = experiment_key

    @property
    def remote(self):
        "Is the asset a remote asset or not, boolean"
        return self._remote

    @property
    def logical_path(self):
        "Asset relative logical_path, str or None"
        return self._logical_path

    @property
    def size(self):
        "Asset size if the asset is a non-remote asset, int"
        return self._size

    @property
    def link(self):
        "Asset remote link if the asset is remote, str or None"
        return self._link

    @property
    def metadata(self):
        "Asset metadata, dict"
        return self._metadata

    @property
    def asset_type(self):
        "Asset type, str"
        return self._asset_type

    @property
    def id(self):
        "Asset unique id, str"
        return self._id

    @property
    def artifact_version_id(self):
        "Artifact version id, str"
        return self._artifact_version_id

    @property
    def artifact_id(self):
        "Artifact id, str"
        return self._artifact_id

    @property
    def source_experiment_key(self):
        "The experiment key of the experiment that logged this asset, str"
        return self._source_experiment_key

    def __repr__(self):
        return (
            "%s(remote=%r, logical_path=%r, size=%r, link=%r, metadata=%r, asset_type=%r, id=%r, artifact_version_id=%r, artifact_id=%r, source_experiment_key=%r)"
            % (
                self.__class__.__name__,
                self._remote,
                self._logical_path,
                self._size,
                self._link,
                self._metadata,
                self._asset_type,
                self._id,
                self._artifact_version_id,
                self._artifact_id,
                self._source_experiment_key,
            )
        )

    def __eq__(self, other):
        return (
            self._remote == other._remote
            and self._logical_path == other._logical_path
            and self._size == other._size
            and self._link == other._link
            and self._metadata == other._metadata
            and self._asset_type == other._asset_type
            and self._id == other._id
            and self._artifact_version_id == other._artifact_version_id
            and self._artifact_id == other._artifact_id
            and self._source_experiment_key == other._source_experiment_key
        )

    def __lt__(self, other):
        return self._logical_path < other._logical_path

    def download(
        self,
        local_path: str = None,  # if None, downloads to a tmp path
        logical_path: str = None,
        overwrite_strategy=False,
    ) -> ArtifactAsset:
        """
        Download the asset to a given full path or directory

        Returns:
            The artifact asset downloaded

        Args:
            local_path: the root folder to which to download.
                if None, will download to a tmp path if str, will be either a root local path or a
                full local path
            logical_path: the path relative to the root local_path to use. If None and
                local_path==None then no relative path is used, file would just be a tmp path on
                local disk. If None and local_path!=None then the local_path will be treated as a
                root path, and the asset's logical_path will be appended to the root path to form a
                full local path. If "" or False then local_path will be used as a full path
                (local_path can also be None)
            overwrite_strategy: can be False, "FAIL", "PRESERVE" or "OVERWRITE"
                and follows the same semantics for overwrite strategy as artifact.download()
        """
        if local_path is None:
            root_path = tempfile.mkdtemp()
        else:
            root_path = local_path

        if logical_path is None:
            asset_filename = self._logical_path
        else:
            asset_filename = logical_path

        result_asset_path = os.path.join(root_path, asset_filename)

        prepared_request = self._rest_api_client._prepare_experiment_asset_request(
            asset_id=self._id,
            experiment_key=self._experiment_key,
            artifact_version_id=self._artifact_version_id,
        )

        download_artifact_asset(
            prepared_request=prepared_request,
            timeout=self._download_timeout,
            asset_id=self._id,
            artifact_repr=self._logged_artifact_repr,
            artifact_str=self._logged_artifact_str,
            asset_logical_path=asset_filename,
            asset_path=result_asset_path,
            overwrite=_validate_overwrite_strategy(overwrite_strategy),
            verify_tls=self._verify_tls,
        )

        return ArtifactAsset(
            remote=False,
            logical_path=self._logical_path,
            size=self._size,
            link=None,
            metadata=self._metadata,
            asset_type=self._asset_type,
            local_path_or_data=result_asset_path,
        )


class Artifact(object):
    def __init__(
        self,
        name: str,
        artifact_type: Optional[str] = None,
        version: Optional[str] = None,
        aliases: Optional[Iterable[str]] = None,
        metadata: Any = None,
        version_tags: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Comet Artifacts allow keeping track of assets beyond any particular experiment. You can keep
        track of Artifact versions, create many types of assets, manage them, and use them in any
        step in your ML pipelines---from training to production deployment.

        Artifacts live in a Comet Project, are identified by their name and version string number.

        Example how to log an artifact with an asset:

        ```python
        from comet_ml import Artifact, start

        experiment = start()
        artifact = Artifact("Artifact-Name", "Artifact-Type")
        artifact.add("local-file")

        experiment.log_artifact(artifact)
        experiment.end()
        ```

        Example how to get and download an artifact assets:

        ```python
        from comet_ml import start

        experiment = start()
        artifact = experiment.get_artifact("Artifact-Name", WORKSPACE, PROJECT_NAME)

        artifact.download("/data/input")
        ```

        The artifact is created on the frontend only when calling [comet_ml.CometExperiment.log_artifact][]

        Args:
            name: The artifact name. Exceeding 100 characters length will cause an exception.
            artifact_type: The artifact-type, for example `dataset`.
            version: The version number to create. If not provided, a new version number
                will be created automatically.
            aliases: Some aliases to attach to the future Artifact
                Version. The aliases list is converted into a set for de-duplication.
            metadata (dict): Some additional data to attach to the future Artifact Version. Must
                be a JSON-encodable dict.
        """

        # Artifact fields
        self.name = name

        # Upsert fields
        if artifact_type is None:
            self.artifact_type = "data"
        else:
            self.artifact_type = artifact_type

        if version is None:
            self.version = None
        else:
            self.version = semantic_version.Version(version)

        self.version_tags: Set[str] = set()
        if version_tags is not None:
            self.version_tags = set(version_tags)

        self.aliases: Set[str] = set()
        if aliases is not None:
            self.aliases = set(aliases)

        self.metadata = validate_metadata(metadata, raise_on_invalid=True)

        self._assets: Dict[str, PreprocessedAsset] = {}

        # The set of assets IDs that was already downloaded through LoggedArtifact.download
        self._downloaded_asset_ids: Set[str] = set()

        self._download_local_path: Optional[str] = None

    @classmethod
    def _from_logged_artifact(
        cls,
        name: str,
        artifact_type: str,
        assets: Dict[str, PreprocessedAsset],
        root_path: str,
        asset_ids: Set[str],
    ) -> "Artifact":
        new_artifact = cls(name, artifact_type)
        new_artifact._assets = assets
        new_artifact._download_local_path = root_path
        new_artifact._downloaded_asset_ids = asset_ids

        return new_artifact

    def add(
        self,
        local_path_or_data: Any,
        logical_path: Optional[str] = None,
        overwrite: bool = False,
        copy_to_tmp: bool = True,  # if local_path_or_data is a file pointer
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a local asset to the current pending artifact object.

        Args:
            local_path_or_data (str | File-like):Either a file/directory path of the files you want
                to log, or a file-like asset.
            logical_path (str): A custom file name to be displayed. If not
                provided the filename from the `local_path_or_data` argument will be used.
            overwrite (bool): If True will overwrite all existing assets with the same name.
            copy_to_tmp (bool): If `local_path_or_data` is a file-like object, then this flag determines
                if the file is first copied to a temporary file before upload. If
                `copy_to_tmp` is False, then it is sent directly to the cloud.
            metadata (dict): Some additional data to attach to the the audio asset. Must be a
                JSON-encodable dict.
        """
        if local_path_or_data is None:
            raise TypeError("local_path_or_data cannot be None")

        dispatched = dispatch_user_file_upload(local_path_or_data)

        if not isinstance(dispatched, (FileUpload, FolderUpload, MemoryFileUpload)):
            raise ValueError(
                "Invalid file_data %r, must either be a valid file-path or an IO object"
                % local_path_or_data
            )

        if isinstance(dispatched, FileUpload):
            asset_id = generate_guid()
            preprocessed = preprocess_asset_file(
                dispatched=dispatched,
                upload_type=ASSET_TYPE_DEFAULT,
                file_name=logical_path,
                metadata=metadata,
                overwrite=overwrite,
                asset_id=asset_id,
                copy_to_tmp=copy_to_tmp,
            )
        elif isinstance(dispatched, FolderUpload):
            preprocessed = preprocess_asset_folder(
                dispatched=dispatched,
                upload_type=ASSET_TYPE_DEFAULT,
                logical_path=logical_path,
                metadata=metadata,
                overwrite=overwrite,
                copy_to_tmp=copy_to_tmp,
            )
        else:
            preprocessed = preprocess_asset_memory_file(
                dispatched=dispatched,
                upload_type=ASSET_TYPE_DEFAULT,
                file_name=logical_path,
                metadata=metadata,
                overwrite=overwrite,
                copy_to_tmp=copy_to_tmp,
            )

        if isinstance(preprocessed, PreprocessedAssetFolder):
            self._add_preprocessed_folder(preprocessed)
        else:
            self._add_preprocessed(preprocessed)

    def add_remote(
        self,
        uri: str,
        logical_path: Optional[str] = None,
        overwrite: bool = False,
        asset_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync_mode: bool = True,
        max_synced_objects: int = 10000,
    ) -> None:
        """
        Add a remote asset to the current pending artifact object. A Remote Asset is an asset but
        its content is not uploaded and stored on Comet. Rather a link for its location is stored, so
        you can identify and distinguish between two experiment using different version of a dataset
        stored somewhere else.

        Args:
            uri (str): The remote asset location, there is no imposed format and it could be a
                private link.
            logical_path (str): The "name" of the remote asset, could be a dataset
                name, a model file name.
            overwrite (bool): If True will overwrite all existing assets with the same name.
            asset_type (str): Define the type of the asset - Deprecated.
            metadata (dict): Some additional data to attach to the remote asset.
                Must be a JSON-encodable dict.
            sync_mode (bool): If True and the URI begins with s3://, Comet attempts to list all
                objects in the given bucket and path. Each object will be logged as a separate
                remote asset. If object versioning is enabled on the S3 bucket, Comet also logs each
                object version to be able to download the exact version. If False, Comet just logs a
                single remote asset with the provided URI as the remote URI. Default is True.
            max_synced_objects (int): When sync_mode is True and the URI begins with s3://, set the
                maximum number of S3 objects to log. If there are more matching S3 objects than
                max_synced_objects, a warning will be displayed and the provided URI will be logged
                as a single remote asset.
        """
        if asset_type:
            debug_helpers.log_warning_or_raise(
                ARTIFACT_ASSET_TYPE_DEPRECATED_WARNING, logger=LOGGER
            )

        asset_type = None

        if sync_mode is True:
            url_scheme = None
            try:
                o = urlparse(uri)
                url_scheme = o.scheme
            except Exception as e:
                debug_helpers.log_warning_or_raise(
                    ARTIFACT_FAILED_TO_PARSE_REMOTE_ASSET_URI_WARNING,
                    uri,
                    exc_info=True,
                    logger=LOGGER,
                    original_exception=e,
                )

            error_message = None
            success = False
            if url_scheme == "s3":
                success, error_message = self._add_s3_assets(
                    uri=uri,
                    max_synced_objects=max_synced_objects,
                    logical_path=logical_path,
                    overwrite=overwrite,
                    asset_type=asset_type,
                    metadata=metadata,
                )
            elif url_scheme == "gs":
                success, error_message = self._add_gs_assets(
                    uri=uri,
                    max_synced_objects=max_synced_objects,
                    logical_path=logical_path,
                    overwrite=overwrite,
                    asset_type=asset_type,
                    metadata=metadata,
                )
            else:
                # log debug warning
                LOGGER.debug(SYNC_MODE_IS_NOT_SUPPORTED_FOR_STRING_REMOTE_ARTIFACT, uri)

            if success is True:
                # to avoid logging this artifact as plain artifact beneath
                return

            # append error message to the metadata
            if error_message is not None:
                # add to metadata
                if metadata is None:
                    metadata = dict()
                metadata[META_ERROR_MESSAGE] = error_message
                metadata[META_SYNCED] = False

        # process asset as usually
        preprocessed = preprocess_remote_asset(
            remote_uri=uri,
            logical_path=logical_path,
            overwrite=overwrite,
            upload_type=asset_type,
            metadata=metadata,
        )
        self._add_preprocessed(preprocessed)

    def _add_gs_assets(
        self,
        uri: str,
        max_synced_objects: int,
        logical_path: Optional[str],
        overwrite: bool,
        asset_type: Optional[str],
        metadata: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        try:
            preprocessed_assets = preprocess_remote_gs_assets(
                remote_uri=uri,
                logical_path=logical_path,
                overwrite=overwrite,
                upload_type=asset_type,
                metadata=metadata,
                max_synced_objects=max_synced_objects,
            )
            for asset in preprocessed_assets:
                self._add_preprocessed(asset)

            # success - no error
            return True, None

        except LogAssetException as lax:
            debug_helpers.log_warning_or_raise(
                lax.backend_err_msg,
                logger=LOGGER,
                original_exception=lax,
                exc_info=True,
            )
            error_message = lax.backend_err_msg
        except Exception as e:
            debug_helpers.log_warning_or_raise(
                FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET,
                uri,
                exc_info=True,
                logger=LOGGER,
                original_exception=e,
            )
            error_message = FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET % uri

        return False, error_message

    def _add_s3_assets(
        self,
        uri: str,
        max_synced_objects: int,
        logical_path: Optional[str],
        overwrite: bool,
        asset_type: Optional[str],
        metadata: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        try:
            preprocessed_assets = preprocess_remote_s3_assets(
                remote_uri=uri,
                logical_path=logical_path,
                overwrite=overwrite,
                upload_type=asset_type,
                metadata=metadata,
                max_synced_objects=max_synced_objects,
            )
            for asset in preprocessed_assets:
                self._add_preprocessed(asset)

            # success - no error
            return True, None

        except LogAssetException as lax:
            debug_helpers.log_warning_or_raise(
                lax.backend_err_msg,
                logger=LOGGER,
                original_exception=lax,
                exc_info=True,
            )
            error_message = lax.backend_err_msg
        except Exception as e:
            debug_helpers.log_warning_or_raise(
                FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET,
                uri,
                logger=LOGGER,
                original_exception=e,
                exc_info=True,
            )
            error_message = FAILED_TO_ADD_ARTIFACT_REMOTE_SYNC_ASSET % uri

        return False, error_message

    def _preprocessed_user_input(self, preprocessed: PreprocessedAsset) -> Any:
        if isinstance(preprocessed, PreprocessedRemoteAsset):
            return preprocessed.remote_uri
        else:
            return preprocessed.local_path_or_data

    def _add_preprocessed(self, preprocessed: PreprocessedAsset) -> None:
        preprocessed_logical_path = preprocessed.logical_path

        if preprocessed_logical_path in self._assets:
            # Allow the overriding of an asset inherited from a downloaded version
            if (
                self._assets[preprocessed_logical_path].asset_id
                in self._downloaded_asset_ids
            ):
                self._downloaded_asset_ids.remove(
                    self._assets[preprocessed_logical_path].asset_id
                )
                self._assets[preprocessed_logical_path] = preprocessed
            else:
                raise ArtifactConflictingAssetLogicalPath(
                    self._preprocessed_user_input(
                        self._assets[preprocessed_logical_path]
                    ),
                    self._preprocessed_user_input(preprocessed),
                    preprocessed_logical_path,
                )
        else:
            self._assets[preprocessed_logical_path] = preprocessed

    def _add_preprocessed_folder(
        self, preprocessed_folder: PreprocessedAssetFolder
    ) -> None:

        for preprocessed_asset_file in preprocessed_folder:
            self._add_preprocessed(preprocessed_asset_file)

    def __str__(self) -> str:
        return "%s(%r, artifact_type=%r)" % (
            self.__class__.__name__,
            self.name,
            self.artifact_type,
        )

    def __repr__(self) -> str:
        return (
            "%s(name=%r, artifact_type=%r, version=%r, aliases=%r, version_tags=%s)"
            % (
                self.__class__.__name__,
                self.name,
                self.artifact_type,
                self.version,
                self.aliases,
                self.version_tags,
            )
        )

    @property
    def assets(self) -> List[ArtifactAsset]:
        """
        The list of `ArtifactAssets` that have been logged with this `Artifact`.
        """
        artifact_version_assets = []

        for asset in self._assets.values():

            if isinstance(asset, PreprocessedRemoteAsset):
                artifact_version_assets.append(
                    ArtifactAsset(
                        remote=True,
                        logical_path=asset.logical_path,
                        # Semantically remote files have a 0 size, but we are still counting
                        # the size for upload progress
                        size=0,
                        link=asset.remote_uri,
                        metadata=asset.metadata,
                        asset_type=asset.upload_type,
                        local_path_or_data=None,
                    )
                )
            elif isinstance(asset, PreprocessedSyncedRemoteAsset):
                artifact_version_assets.append(
                    ArtifactAsset(
                        remote=True,
                        logical_path=asset.logical_path,
                        size=asset.size,
                        link=asset.remote_uri,
                        metadata=asset.metadata,
                        asset_type=asset.upload_type,
                        local_path_or_data=asset.local_path,
                    )
                )
            else:
                artifact_version_assets.append(
                    ArtifactAsset(
                        remote=False,
                        logical_path=asset.logical_path,
                        size=asset.size,
                        link=None,
                        metadata=asset.metadata,
                        asset_type=None,
                        local_path_or_data=asset.local_path_or_data,
                    )
                )

        return artifact_version_assets

    @property
    def download_local_path(self) -> Optional[str]:
        """If the Artifact object was returned by `LoggedArtifact.download`, returns the root path
        where the assets have been downloaded. Else, returns None.
        """
        return self._download_local_path


class LoggedArtifact(object):
    def __init__(
        self,
        artifact_name: str,
        artifact_type: str,
        artifact_id: str,
        artifact_version_id: str,
        workspace: str,
        rest_api_client: RestApiClient,
        experiment_key: str,
        version: str,
        aliases: List[str],
        artifact_tags: List[str],
        version_tags: List[str],
        size: int,
        metadata: Dict[str, Any],
        source_experiment_key: str,
        summary: Summary,
        config: Config,
    ) -> None:
        """
        You shouldn't try to create this object by hand, please use
        [comet_ml.Experiment.get_artifact][] instead to
        retrieve an artifact.
        """
        # Artifact fields
        self._artifact_type = artifact_type
        self._name = artifact_name
        self._artifact_id = artifact_id
        self._artifact_version_id = artifact_version_id

        self._version = semantic_version.Version(version)
        self._aliases = frozenset(aliases)
        self._rest_api_client = rest_api_client
        self._workspace = workspace
        self._artifact_tags = frozenset(artifact_tags)
        self._version_tags = frozenset(version_tags)
        self._size = size
        self._source_experiment_key = source_experiment_key
        self._experiment_key = experiment_key  # TODO: Remove ME
        self._summary = summary
        self._config = config

        if metadata is not None:
            self._metadata = ImmutableDict(metadata)
        else:
            self._metadata = ImmutableDict()

    def _raw_assets(self):
        """Returns the artifact version ID assets"""
        return self._rest_api_client.get_artifact_files(
            workspace=self._workspace,
            name=self._name,
            version=str(self.version),
        )["files"]

    def _to_logged_artifact(
        self, raw_artifact_asset: Dict[str, Any]
    ) -> LoggedArtifactAsset:
        if "remote" in raw_artifact_asset:
            remote = raw_artifact_asset["remote"]
        else:
            remote = (
                raw_artifact_asset["link"] is not None
            )  # TODO: Remove me after October 1st

        return LoggedArtifactAsset(
            remote,
            raw_artifact_asset["fileName"],
            raw_artifact_asset["fileSize"],
            raw_artifact_asset["link"],
            raw_artifact_asset["metadata"],
            raw_artifact_asset["type"],
            raw_artifact_asset["assetId"],
            self._artifact_version_id,
            self._artifact_id,
            self._source_experiment_key,
            verify_tls=self._config.get_bool(
                None, "comet.internal.check_tls_certificate"
            ),
            rest_api_client=self._rest_api_client,
            download_timeout=self._config.get_int(None, "comet.timeout.file_download"),
            logged_artifact_repr=self.__repr__(),
            logged_artifact_str=self.__str__(),
            experiment_key=self._experiment_key,
        )

    @property
    def assets(self) -> List[LoggedArtifactAsset]:
        """
        The list of `LoggedArtifactAsset` that have been logged with this `LoggedArtifact`.
        """
        artifact_version_assets = []

        for asset in self._raw_assets():
            artifact_version_assets.append(self._to_logged_artifact(asset))

        return artifact_version_assets

    @property
    def remote_assets(self) -> List[LoggedArtifactAsset]:
        """
        The list of remote `LoggedArtifactAsset` that have been logged with this `LoggedArtifact`.
        """
        artifact_version_assets = []

        for asset in self._raw_assets():
            if "remote" in asset:
                remote = asset["remote"]
            else:
                remote = asset["link"] is not None  # TODO: Remove me after October 1st

            if not remote:
                continue

            artifact_version_assets.append(self._to_logged_artifact(asset))

        return artifact_version_assets

    def get_asset(self, asset_logical_path) -> LoggedArtifactAsset:
        """
        Returns the LoggedArtifactAsset object matching the given asset_logical_path or raises an Exception
        """
        for asset in self._raw_assets():
            if asset["fileName"] == asset_logical_path:
                return self._to_logged_artifact(asset)

        raise ArtifactAssetNotFound(asset_logical_path, self)

    def download(
        self,
        path: Optional[str] = None,
        overwrite_strategy: Union[bool, str] = False,
        sync_mode: bool = True,
    ) -> Artifact:
        """
        Download the current Artifact Version assets to a given directory (or the local directory by
        default).

        This method downloads assets and remote assets that were synced from a compatible cloud
        object storage (AWS S3 or GCP GCS). Other non-remote assets are not downloaded and you can
        access their link with the `artifact.assets` property.

        Args:
            path (str): Where to download artifact version assets. If not provided,
                a temporary path will be used, the root path can be accessed through the Artifact
                object which is returned by download under the `.download_local_path` attribute.
            overwrite_strategy (bool | str): One of the three possible strategies to handle
                conflict when trying to download an artifact version asset to a path with an
                existing file. See below for allowed values. Default is False or "FAIL".
            sync_mode (bool): Enables download of remote assets from the cloud storage platforms
                (AWS S3, GCP GS).

        Returns:
            Artifact: The Artifact downloaded object

        Note:
            Overwrite strategy allowed values:

            - False or "FAIL": If a file already exists and its content is different, raise the
                `comet_ml.exceptions.ArtifactDownloadException`.
            - "PRESERVE": If a file already exists
                and its content is different, show a WARNING but preserve the existing content.
            - True or "OVERWRITE": If a file already exists and its content is different, replace it by the
                asset version asset.
        """

        if path is None:
            root_path = tempfile.mkdtemp()
        else:
            root_path = path

        overwrite_strategy = _validate_overwrite_strategy(overwrite_strategy)

        new_artifact_assets: Dict[str, PreprocessedAsset] = {}
        new_artifact_asset_ids = set()

        try:
            raw_assets = self._raw_assets()
        except Exception:
            raise ArtifactDownloadException(
                ARTIFACT_DOWNLOAD_CANNOT_GET_ASSETS_LIST_EXCEPTION % self
            )

        worker_cpu_ratio = self._config.get_int(
            None, "comet.internal.file_upload_worker_ratio"
        )
        worker_count = self._config.get_raw(None, "comet.internal.worker_count")
        download_manager = FileDownloadManager(
            worker_cpu_ratio=worker_cpu_ratio, worker_count=worker_count
        )

        file_download_timeout = self._config.get_int(
            None, "comet.timeout.file_download"
        )
        verify_tls = get_check_tls_certificate(self._config)

        download_result_holder = namedtuple(
            "_download_result_holder",
            [
                "download_result",
                "asset_filename",
                "asset_path",
                "asset_metadata",
                "asset_id",
                "asset_synced",
                "asset_type",
                "asset_overwrite_strategy",
                "asset_remote_uri",
            ],
        )
        results: List[download_result_holder] = list()

        self_repr = repr(self)
        self_str = str(self)

        for asset in raw_assets:
            asset_metadata = asset["metadata"]
            if asset_metadata is not None:
                asset_metadata = json.loads(asset["metadata"])

            if "remote" in asset:
                asset_remote = asset["remote"]
            else:
                asset_remote = (
                    asset["link"] is not None
                )  # TODO: Remove me after October 1st

            remote_uri = asset.get("link", None)
            asset_filename = asset["fileName"]
            asset_id = asset["assetId"]
            asset_path = os.path.join(root_path, asset_filename)
            asset_synced = False
            asset_sync_error = None
            asset_type = asset.get("type", ASSET_TYPE_DEFAULT)
            if asset_metadata is not None:
                if META_SYNCED in asset_metadata:
                    asset_synced = asset_metadata[META_SYNCED]
                if META_ERROR_MESSAGE in asset_metadata:
                    asset_sync_error = asset_metadata[META_ERROR_MESSAGE]

            if asset_remote is True:
                # check if sync_mode is not enabled or asset was not synced properly
                if sync_mode is False or asset_synced is False:
                    # check if error is in metadata - failed to sync during upload due cloud storage error
                    if asset_sync_error is not None and sync_mode is True:
                        # raise error only if sync_mode==True
                        raise ArtifactDownloadException(
                            ASSET_DOWNLOAD_FAILED_WITH_ERROR
                            % (asset_filename, asset_sync_error)
                        )

                    # We don't download plain remote assets
                    new_artifact_assets[asset_filename] = PreprocessedRemoteAsset(
                        remote_uri=remote_uri,
                        overwrite=False,
                        upload_type=asset_type,
                        metadata=asset_metadata,
                        step=None,
                        asset_id=asset_id,
                        logical_path=asset_filename,
                        size=len(asset["link"]),
                    )
                    new_artifact_asset_ids.add(asset_id)
                    self._summary.increment_section("downloads", "artifact assets")
                else:
                    # check that asset is from supported cloud storage if sync_mode enabled
                    # and asset was synced during Artifact upload
                    o = urlparse(remote_uri)
                    if o.scheme == "s3" or o.scheme == "gs":
                        # register download from AWS S3 or GCS
                        if META_FILE_SIZE in asset_metadata:
                            asset_file_size = asset_metadata[META_FILE_SIZE]
                        else:
                            asset_file_size = 0

                        version_id = None
                        if META_VERSION_ID in asset_metadata:
                            version_id = asset_metadata[META_VERSION_ID]

                        if o.scheme == "s3":
                            data_writer = AssetDataWriterFromS3(
                                s3_uri=remote_uri, version_id=version_id
                            )
                        else:
                            data_writer = AssetDataWriterFromGCS(
                                gs_uri=remote_uri, version_id=version_id
                            )

                        result = download_manager.download_file_async(
                            download_cloud_storage_artifact_asset,
                            data_writer=data_writer,
                            estimated_size=asset_file_size,
                            asset_id=asset_id,
                            artifact_repr=self_repr,
                            artifact_str=self_str,
                            asset_logical_path=asset_filename,
                            asset_path=asset_path,
                            overwrite=overwrite_strategy,
                        )
                        results.append(
                            download_result_holder(
                                download_result=result,
                                asset_filename=asset_filename,
                                asset_path=asset_path,
                                asset_metadata=asset_metadata,
                                asset_id=asset_id,
                                asset_synced=asset_synced,
                                asset_type=asset_type,
                                asset_overwrite_strategy=overwrite_strategy,
                                asset_remote_uri=remote_uri,
                            )
                        )
                    else:
                        # unsupported URI scheme for synced asset
                        raise ArtifactDownloadException(
                            UNSUPPORTED_URI_SYNCED_REMOTE_ASSET % remote_uri
                        )
            else:
                prepared_request = (
                    self._rest_api_client._prepare_experiment_asset_request(
                        asset_id=asset_id,
                        experiment_key=self._experiment_key,
                        artifact_version_id=asset["artifactVersionId"],
                    )
                )

                # register asset to be downloaded
                result = download_manager.download_file_async(
                    download_artifact_asset,
                    prepared_request=prepared_request,
                    timeout=file_download_timeout,
                    verify_tls=verify_tls,
                    asset_id=asset_id,
                    artifact_repr=self_repr,
                    artifact_str=self_str,
                    asset_logical_path=asset_filename,
                    asset_path=asset_path,
                    overwrite=overwrite_strategy,
                    estimated_size=asset["fileSize"],
                )

                results.append(
                    download_result_holder(
                        download_result=result,
                        asset_filename=asset_filename,
                        asset_path=asset_path,
                        asset_metadata=asset_metadata,
                        asset_id=asset_id,
                        asset_synced=asset_synced,
                        asset_type=asset_type,
                        asset_overwrite_strategy=overwrite_strategy,
                        asset_remote_uri=remote_uri,
                    )
                )

        # Forbid new usage
        download_manager.close()

        # Wait for download manager to complete registered file downloads
        if not download_manager.all_done():
            monitor = FileDownloadManagerMonitor(download_manager)

            LOGGER.info(
                ARTIFACT_DOWNLOAD_START_MESSAGE,
                self._workspace,
                self._name,
                self._version,
            )

            wait_for_done(
                check_function=monitor.all_done,
                timeout=self._config.get_int(None, "comet.timeout.artifact_download"),
                progress_callback=monitor.log_remaining_downloads,
                sleep_time=15,
            )

        # iterate over download results and create file assets descriptors
        try:
            for result in results:
                try:
                    result.download_result.get(file_download_timeout)

                    new_asset_size = os.path.getsize(result.asset_path)
                except Exception:
                    # display failed message
                    LOGGER.error(
                        ARTIFACT_ASSET_DOWNLOAD_FAILED,
                        result.asset_filename,
                        self._workspace,
                        self._name,
                        self._version,
                        exc_info=True,
                    )

                    raise ArtifactDownloadException(
                        ARTIFACT_DOWNLOAD_CANNOT_DOWNLOAD_ASSET_EXCEPTION
                        % (result.asset_filename, self_repr)
                    )

                self._summary.increment_section(
                    "downloads",
                    "artifact assets",
                    size=new_asset_size,
                )

                if result.asset_synced is False:
                    # downloaded local asset
                    new_artifact_assets[result.asset_filename] = PreprocessedFileAsset(
                        local_path_or_data=result.asset_path,
                        upload_type=result.asset_type,
                        logical_path=result.asset_filename,
                        metadata=result.asset_metadata,
                        overwrite=result.asset_overwrite_strategy,
                        step=None,
                        asset_id=result.asset_id,
                        grouping_name=None,  # TODO: FIXME?
                        extension=None,  # TODO: FIXME?
                        size=new_asset_size,
                        copy_to_tmp=False,
                    )
                else:
                    # downloaded synced remote asset from cloud storage (AWS S3, GCS)
                    new_artifact_assets[result.asset_filename] = (
                        PreprocessedSyncedRemoteAsset(
                            remote_uri=result.asset_remote_uri,
                            overwrite=result.asset_overwrite_strategy,
                            upload_type=result.asset_type,
                            metadata=result.asset_metadata,
                            step=None,
                            asset_id=result.asset_id,
                            logical_path=result.asset_filename,
                            size=new_asset_size,
                            local_path=result.asset_path,
                        )
                    )

                new_artifact_asset_ids.add(result.asset_id)

            # display success message
            LOGGER.info(
                ARTIFACT_DOWNLOAD_FINISHED, self._workspace, self._name, self._version
            )
        finally:
            download_manager.join()

        return Artifact._from_logged_artifact(
            name=self._name,
            artifact_type=self._artifact_type,
            assets=new_artifact_assets,
            root_path=root_path,
            asset_ids=new_artifact_asset_ids,
        )

    def get_source_experiment(
        self,
        api_key: Optional[str] = None,
        cache: bool = True,
    ) -> APIExperiment:
        """
        Returns an APIExperiment object pointing to the experiment that created this artifact version, assumes that the API key is set else-where.
        """
        return APIExperiment(
            api_key=api_key,
            cache=cache,
            previous_experiment=self._source_experiment_key,
        )

    def update_artifact_tags(self, new_artifact_tags: Sequence[str]) -> None:
        """
        Update the logged artifact tags
        """
        new_artifact_tags_list = list(new_artifact_tags)

        self._rest_api_client.update_artifact(
            self._artifact_id,
            tags=new_artifact_tags,
        )

        self._artifact_tags = frozenset(new_artifact_tags_list)

    def update_version_tags(self, new_version_tags: Sequence[str]) -> None:
        """
        Update the logged artifact version tags
        """
        new_version_tags_list = list(new_version_tags)

        self._rest_api_client.update_artifact_version(
            self._artifact_version_id,
            version_tags=new_version_tags_list,
        )

        self._version_tags = frozenset(new_version_tags_list)

    def update_aliases(self, new_aliases: Sequence[str]) -> None:
        """
        Update the logged artifact tags
        """
        new_aliases_list = list(new_aliases)

        self._rest_api_client.update_artifact_version(
            self._artifact_version_id,
            version_aliases=new_aliases_list,
        )

        self._aliases = frozenset(new_aliases_list)

    # Public properties
    @property
    def name(self):
        """
        The logged artifact name.
        """
        return self._name

    @property
    def artifact_type(self):
        """
        The logged artifact type.
        """
        return self._artifact_type

    @property
    def version(self) -> semantic_version.Version:
        """
        The logged artifact version, as a SemanticVersion. See
        https://python-semanticversion.readthedocs.io/en/latest/reference.html#semantic_version.Version
        for reference
        """
        return self._version

    @property
    def workspace(self):
        """
        The logged artifact workspace name.
        """
        return self._workspace

    @property
    def aliases(self):
        """
        The set of logged artifact aliases.
        """
        return self._aliases

    @property
    def metadata(self):
        """
        The logged artifact metadata.
        """
        return self._metadata

    @property
    def version_tags(self):
        """
        The set of logged artifact version tags.
        """
        return self._version_tags

    @property
    def artifact_tags(self):
        """
        The set of logged artifact tags.
        """
        return self._artifact_tags

    @property
    def size(self):
        """
        The total size of logged artifact version; it is the sum of all the artifact version assets.
        """
        return self._size

    @property
    def source_experiment_key(self):
        """
        The experiment key of the experiment that created this LoggedArtifact.
        """
        return self._source_experiment_key

    def __str__(self):
        return "<%s '%s/%s:%s'>" % (
            self.__class__.__name__,
            self._workspace,
            self._name,
            self._version,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(artifact_name=%r, artifact_type=%r, workspace=%r, version=%r, aliases=%r, artifact_tags=%r, version_tags=%r, size=%r, source_experiment_key=%r)"
            % (
                self._name,
                self._artifact_type,
                self._workspace,
                self._version,
                self._aliases,
                self._artifact_tags,
                self._version_tags,
                self._size,
                self._source_experiment_key,
            )
        )
