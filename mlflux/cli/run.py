import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import List
from tempfile import TemporaryDirectory

import docker
import mlflow
import typer
import yaml
from loguru import logger
from mlflow.utils.logging_utils import MLFLOW_LOGGING_STREAM


# region : copied from mlflow.cli
def eprint(*args, **kwargs):
    print(*args, file=MLFLOW_LOGGING_STREAM, **kwargs)


def _user_args_to_dict(arguments, argument_type="P"):
    user_dict = {}
    for arg in arguments:
        split = arg.split("=", maxsplit=1)
        # Docker arguments such as `t` don't require a value -> set to True if specified
        if len(split) == 1 and argument_type == "A":
            name = split[0]
            value = True
        elif len(split) == 2:
            name = split[0]
            value = split[1]
        else:
            eprint(
                "Invalid format for -%s parameter: '%s'. "
                "Use -%s name=value." % (argument_type, arg, argument_type)
            )
            sys.exit(1)
        if name in user_dict:
            eprint("Repeated parameter: '%s'" % name)
            sys.exit(1)
        user_dict[name] = value
    return user_dict


# endregion


def _copy_from_uri(project_path: Path, uri: str):
    # TODO: git uri
    # TODO: discard stuff from gitignore
    src_path = Path(uri)
    shutil.copytree(src_path, project_path, dirs_exist_ok=True)


@contextmanager
def _docker_image(project_path: Path):
    ml_project_path = project_path / "MLproject"

    with ml_project_path.open() as f:
        ml_project = yaml.safe_load(f)

    tag = ml_project["docker_env"]["image"]
    try:
        dockerfile = project_path / ml_project["docker_env"]["Dockerfile"]
    except KeyError:
        dockerfile = project_path / "Dockerfile"

    if (":" not in tag) and dockerfile.exists():
        tag += f":{uuid.uuid4()}"

        logger.info(f"Building docker image {tag} from {dockerfile}")
        docker_client = docker.from_env()
        img, docker_logs = docker_client.images.build(
            path=project_path.as_posix(),
            dockerfile=dockerfile.relative_to(project_path).as_posix(),
            tag=tag,
        )
        logger.info(f"Docker image {tag} built")

        ml_project["docker_env"]["image"] = tag
        with ml_project_path.open("w") as f:
            yaml.safe_dump(ml_project, f)

        yield docker_logs

        docker_client.images.remove(tag)
    else:
        yield None


def run(
    *,
    param_list: List[str] = typer.Option([], "-P", "--param-list"),
    entry_point: str = typer.Option("main", "-e", "--entry-point"),
    uri: str,
):
    params_dict = _user_args_to_dict(param_list)

    with TemporaryDirectory() as tmp_dir:
        project_path = Path(tmp_dir)
        _copy_from_uri(project_path, uri)

        with _docker_image(project_path) as docker_logs:
            mlflow_run = mlflow.projects.run(
                uri=project_path.as_posix(),
                entry_point=entry_point,
                parameters=params_dict,
            )
