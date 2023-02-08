import typer
import requests

import echo_client.functions as f

app = typer.Typer()


@app.command()
def status(host="localhost", port=8000):
    f.status(host, port)


@app.command()
def echo(text="Hello, world!", host="localhost", port=8000):
    f.echo(text, host, port)


@app.command()
def rot(msg="Uryyb Penml Jbeyq Bs 2022", rotation=13, host="localhost", port=8000):
    f.rot(msg, rotation, host, port)


if __name__ == "__main__":
    app()
