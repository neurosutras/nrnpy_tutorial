import click
import os


@click.command()
@click.option("--a", type=int, default=10)
@click.option("--verbose", is_flag=True)
def main(a, verbose):
    if verbose:
        print('process: %i; value of a: %i' % (os.getpid(), a))
        print('locals(): ', locals())


if __name__ == '__main__':
    main()