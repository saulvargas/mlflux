import os
import shutil
import sys
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import docker
import mlflow
import typer
import yaml
from gitignore_parser import parse_gitignore
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


def _copy_from_uri(uri: str, project_path: Path, output_path) -> Optional[Path]:
    # TODO: git uri
    src_path = Path(uri)

    gitignore_path = src_path / ".gitignore"
    if gitignore_path.exists():
        matches = parse_gitignore(gitignore_path.as_posix())

        def ignore(src, names):
            return [f for f in names if matches(os.path.join(src, f)) or (f == ".git")]

    else:
        ignore = None

    shutil.copytree(src_path, project_path, ignore=ignore, dirs_exist_ok=True)

    base_path = output_path / "src"
    src_zip_path = Path(shutil.make_archive(base_path, "zip", project_path))

    return src_zip_path


def _setup_docker_image(project_path: Path, output_path: Path) -> Optional[Path]:
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

        docker_logs_path = output_path / "docker.stdout.txt"
        with docker_logs_path.open("w") as f:
            for log in docker_logs:
                try:
                    f.write(log["stream"])
                except KeyError:
                    pass

        return docker_logs_path


def _store_docker_logs(docker_logs, output_path):
    docker_logs_path = output_path / "docker.stdout.txt"
    with docker_logs_path.open("w") as f:
        for log in docker_logs:
            f.write(log)


def run(
    *,
    param_list: List[str] = typer.Option([], "-P", "--param-list"),
    entry_point: str = typer.Option("main", "-e", "--entry-point"),
    uri: str,
):
    params_dict = _user_args_to_dict(param_list)

    with TemporaryDirectory() as project_dir, TemporaryDirectory() as output_dir:
        project_path = Path(project_dir)
        output_path = Path(output_dir)

        src_zip_path = _copy_from_uri(uri, project_path, output_path)
        docker_logs_path = _setup_docker_image(project_path, output_path)

        mlflow_run = mlflow.projects.run(
            uri=project_path.as_posix(),
            entry_point=entry_point,
            parameters=params_dict,
        )

        with mlflow.start_run(mlflow_run.run_id):
            mlflow.log_artifact(src_zip_path.as_posix())
            mlflow.log_artifact(docker_logs_path.as_posix(), "logs")
