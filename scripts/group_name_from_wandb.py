import logging

import click

from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


@click.command()
@click.argument(
    "swanlab_run_path",
    type=str,
)
def main(swanlab_run_path: str):
    import swanlab

    api = swanlab.Api()
    run = api.run(swanlab_run_path)
    print(run.group)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
