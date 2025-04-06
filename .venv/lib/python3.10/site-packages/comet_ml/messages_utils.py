# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This source code is licensed under the MIT license.
# *******************************************************
from .convert_utils import data_to_fp
from .file_uploader import is_user_text
from .file_utils import write_file_like_to_tmp_file
from .messages import UploadFileMessage, UploadInMemoryMessage


def convert_upload_in_memory_to_file_message(
    message: UploadInMemoryMessage, tmp_dir: str
) -> UploadFileMessage:
    if is_user_text(message.file_like):
        file_like = data_to_fp(message.file_like)
    else:
        file_like = message.file_like

    tmp_file = write_file_like_to_tmp_file(file_like, tmp_dir)

    return UploadFileMessage(
        file_path=tmp_file,
        upload_type=message.upload_type,
        additional_params=message.additional_params,
        metadata=message.metadata,
        clean=True,
        size=message._size,
        critical=message._critical,
        message_id=message.message_id,
        on_asset_upload=message._on_asset_upload,
        on_failed_asset_upload=message._on_failed_asset_upload,
    )
