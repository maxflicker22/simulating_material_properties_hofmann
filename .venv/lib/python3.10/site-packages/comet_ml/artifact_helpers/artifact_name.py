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
from typing import Optional, Tuple


def parse_artifact_name(
    artifact_name: str,
) -> Tuple[Optional[str], str, Optional[str]]:
    """Parse an artifact_name, potentially a fully-qualified name"""

    name_parts = artifact_name.split("/")

    # First parse the workspace
    if len(name_parts) == 1:
        workspace = None
        artifact_name_version = name_parts[0]
    else:
        workspace = name_parts[0]
        artifact_name_version = name_parts[1]

    name_version_parts = artifact_name_version.split(":", 1)

    if len(name_version_parts) == 1:
        artifact_name = name_version_parts[0]
        version_or_alias = None
    else:
        artifact_name = name_version_parts[0]
        version_or_alias = name_version_parts[1]

    return workspace, artifact_name, version_or_alias
