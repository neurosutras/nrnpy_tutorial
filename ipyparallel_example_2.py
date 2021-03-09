import click
import os
from ipyparallel import Client


class Context(object):
    """
    A container replacement for global variables to be shared and modified by any function in a module.
    """
    def __init__(self, namespace_dict=None, **kwargs):
        self.update(namespace_dict, **kwargs)

    def update(self, namespace_dict=None, **kwargs):
        """
        Converts items in a dictionary (such as globals() or locals()) into context object internals.
        :param namespace_dict: dict
        """
        if namespace_dict is not None:
            self.__dict__.update(namespace_dict)
        self.__dict__.update(kwargs)

    def __call__(self):
        return self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]


context = Context()


def report_a():
    if 'a' not in context():
        return None
    else:
        return context.a


def modify_a(a):
    raise Exception('Ajay broke everything here.')
    context.a = a


@click.command()
@click.option("--a", type=int, default=10)
@click.option("--verbose", is_flag=True)
def main(a, verbose):

    context().update(locals())

    if verbose:
        print('process: %i; value of a: %i' % (os.getpid(), context.a))
        print('ajay be here')

    # requires that $N number of remote workers first be instantiated from the command line with
    # ipcluster start -n $N

    context.client = Client()


if __name__ == '__main__':
    main()