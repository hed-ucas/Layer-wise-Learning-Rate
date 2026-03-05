import logging

import click

from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


@click.command()
@click.argument(
    "swanlab_run_path",
    type=str,
)
@click.argument(
    "new_group_name",
    type=str,
)
def main(
    swanlab_run_path: str,
    new_group_name: str,
):
    import swanlab

    api = swanlab.Api()
    run = api.run(swanlab_run_path)
    run.group = new_group_name
    run.update()


if __name__ == "__main__":
    prepare_cli_environment()
    main()
