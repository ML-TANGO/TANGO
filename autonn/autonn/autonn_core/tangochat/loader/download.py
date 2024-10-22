# This source code is from Meta Platforms, Inc. and affiliates.
# ETRI modified it for TANGO project.

import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional
import time

from loader.convert_hf_checkpoint import convert_hf_checkpoint
from common.models.model_config import (
    load_model_configs,
    ModelConfig,
    ModelDistributionChannel,
    resolve_model_config,
)
import streamlit as st

import logging
logger = logging.getLogger(__name__)

def _get_diretory_size(diretory, unit='GB'):
    total_size = 0.0
    with os.scandir(diretory) as it:
        for entry in it:
            if entry.is_file():
                total_size += entry.stat().st_size
    if   unit in ['GB', 'G', 'gb', 'g']:
        power = 3
    elif unit in ['MB', 'M', 'mb', 'm']:
        power = 2
    elif unit in ['kB', 'K', 'kb', 'k']:
        power = 1
    else:
        power = 0
    total_size /= (1024**power)
    return total_size


def _download_hf_snapshot(
    model_config: ModelConfig, artifact_dir: Path, cache_dir: Path, hf_token: Optional[str]
):
    from huggingface_hub import snapshot_download
    from requests.exceptions import HTTPError

    # Download and store the HF model artifacts.
    logger.info(f"Downloading {model_config.name} from HuggingFace...") #, file=sys.stderr)

    with st.status("Downloading... ", expanded=True) as sts:
        try:
            start = time.time()
            result = snapshot_download(
                model_config.distribution_path,
                cache_dir=artifact_dir, #cache_dir,
                local_dir=artifact_dir,
                # local_dir_use_symlinks=False,
                token=hf_token,
                local_files_only=False,
                ignore_patterns="*safetensors*",

            )
            elapsed_time = time.time()-start
            total_size = _get_diretory_size(artifact_dir, unit='GB')
            st.write(f"Downloaded as {result}({total_size:.1f} GB). Elapsed time {elapsed_time:.2f} sec")
        except HTTPError as e:
            logger.info(f"{e}, {type(e)}")
            # if e.response.status == 401:  # Missing HuggingFace CLI login.
            #     logger.warning(
            #         "Access denied. Create a HuggingFace account and run 'pip3 install huggingface_hub' and 'huggingface-cli login' to authenticate.",
            #         # file=sys.stderr,
            #     )
            #     st.warning("Access denied. Create a HuggingFace account and run 'pip3 install huggingface_hub' and 'huggingface-cli login' to authenticate.")
            #     exit(1)
            # elif e.response.status == 403:  # No access to the specific model.
            #     # The error message includes a link to request access to the given model. This prints nicely and does not include
            #     # a traceback.
            #     logger.warning(e) #str(e), file=sys.stderr)
            #     st.warning(e)
            #     exit(1)
            # else:
            #     st.warning(e)
            #     raise e


        # Convert the model to the torchchat format.
        logger.info(f"Converting {model_config.name} to PyTorch format...") #, file=sys.stderr)
        start_convert = time.time()
        convert_hf_checkpoint(
            model_dir=artifact_dir, model_name=model_config.name, remove_bin_files=True
        )
        convert_time = time.time()-start_convert
        st.write(f"Converted as {artifact_dir}/model.pth. Elapsed time {convert_time:.2f} sec")
        sts.update(label=f"Done.", state="complete")


def _download_direct(
    model_config: ModelConfig,
    artifact_dir: Path,
):
    for url in model_config.distribution_path:
        filename = url.split("/")[-1]
        local_path = artifact_dir / filename
        print(f"Downloading {url}...", file=sys.stderr)
        urllib.request.urlretrieve(url, str(local_path.absolute()))


def is_model_downloaded(model: str, models_dir: Path) -> bool:
    model_config = resolve_model_config(model)
    is_downloaded = False
    # Check if the model directory exists and is not empty.
    model_dir = models_dir / model_config.name
    # if os.path.isdir(model_dir):
    #     is_downloaded = os.listdir(model_dir)
    # else:
    #     is_downloaded = False
    # is_downloaded = os.path.isdir(model_dir) and os.listdir(model_dir)

    # Check if the model.pth exists
    model_path = model_dir / 'model.pth'
    if os.path.isfile(model_path):
        is_downloaded = True

    # Clean if downloading has been incompleted before
    # if not is_downloaded:
    #     if os.path.isdir(model_dir):
    #         shutil.rmtree(model_dir)
    #     tmp_dir = models_dir / 'downloads' / model_config.name
    #     if os.path.isdir(tmp_dir):
    #         shutil.rmtree(tmp_dir)

    return is_downloaded, model_dir


def list_model(model_directory) -> None:
    model_configs = load_model_configs()

    # Build the table in-memory so that we can align the text nicely.
    name_col = []
    aliases_col = []
    installed_col = []
    return_model_dir, return_model_name = [], []
    for name, config in model_configs.items():
        is_downloaded, model_dir = is_model_downloaded(name, model_directory)

        name_col.append(name)
        aliases_col.append(", ".join(config.aliases))
        installed_col.append("Yes" if is_downloaded else " - ")

        if is_downloaded:
            return_model_dir.append(model_dir)
            return_model_name.append(config.aliases[0] if isinstance(config.aliases, list) else config.aliases)

    cols = {"Model": name_col, "Aliases": aliases_col, "Downloaded": installed_col}

    # Find the length of the longest value in each column.
    col_widths = {
        key: max(*[len(s) for s in vals], len(key)) + 1 for (key, vals) in cols.items()
    }

    # Display header.
    header = ''
    line = 0
    for val, width in col_widths.items():
        header += (' '*(width-len(val))+val)
        line += width
    logger.info(header)
    logger.info('-'*line)

    # Display content.
    for i in range(len(name_col)):
        row = [col[i] for col in cols.values()]
        content = ''
        for val, width in zip(row, col_widths.values()):
            content += (' '*(width-len(val))+val)
        logger.info(content)

    # logger.info(return_model_dir)
    return return_model_name, return_model_dir


def remove_model(args) -> None:
    # TODO It would be nice to have argparse validate this. However, we have
    # model as an optional named parameter for all subcommands, so we'd
    # probably need to move it to be registered per-command.
    if not args.model:
        print("Usage: torchchat.py remove <model-or-alias>")
        return

    model_config = resolve_model_config(args.model)
    model_dir = args.model_directory / model_config.name

    if not os.path.isdir(model_dir):
        print(f"Model {args.model} has no downloaded artifacts.")
        return

    print(f"Removing downloaded model artifacts for {args.model}...")
    shutil.rmtree(model_dir)
    print("Done.")


def where_model(args) -> None:
    # TODO It would be nice to have argparse validate this. However, we have
    # model as an optional named parameter for all subcommands, so we'd
    # probably need to move it to be registered per-command.
    if not args.model:
        print("Usage: torchchat.py where <model-or-alias>")
        return

    model_config = resolve_model_config(args.model)
    model_dir = args.model_directory / model_config.name

    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Model {args.model} has no downloaded artifacts.")

    print(str(os.path.abspath(model_dir)))
    exit(0)


def download_model(model, models_dir, hf_token) -> None:
    model_config = resolve_model_config(model)
    model_dir = models_dir / model_config.name

    # Download into a temporary directory. We'll move to the final
    # location once the download and conversion is complete. This
    # allows recovery in the event that the download or conversion
    # fails unexpectedly.
    temp_dir = models_dir / "downloads" / model_config.name
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    logger.info(f"Downloading this model into a temp dir {temp_dir}...")
    try:
        if (
            model_config.distribution_channel
            == ModelDistributionChannel.HuggingFaceSnapshot
        ):
            _download_hf_snapshot(model_config, temp_dir, model_dir, hf_token)
        elif (
            model_config.distribution_channel == ModelDistributionChannel.DirectDownload
        ):
            _download_direct(model_config, temp_dir)
        else:
            logger.warning(f"Unknown distribution channel {model_config.distribution_channel}.")
            st.warning(f"Unknown distribution channel {model_config.distribution_channel}.")
            raise RuntimeError(
                f"Unknown distribution channel {model_config.distribution_channel}."
            )

        # Move from the temporary directory to the intended location,
        # overwriting if necessary.
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        shutil.move(temp_dir, model_dir)
        logger.info(f"Moving model to {model_dir}")
        # st.success(f"Moving model to {model_dir}")

    # [tenace] if we try to re-download, huggingface api wants to check cache file.
    finally:
        logger.info(f"Do Nothing") #Delete temporary directory {temp_dir}")
        # if os.path.isdir(temp_dir):
        #     shutil.rmtree(temp_dir)
