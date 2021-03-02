import click
import os

print(__name__)

@click.command()
@click.option("--verbose", is_flag=True)
def main(verbose):
    if verbose:
        print('process: %i; verbose flag on' % os.getpid())


if __name__ == '__main__':
    main()