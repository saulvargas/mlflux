import typer

from mlflux.cli.hello import hello
from mlflux.cli.run import run

app = typer.Typer()

app.command()(hello)
app.command()(run)


if __name__ == '__main__':
    app()
