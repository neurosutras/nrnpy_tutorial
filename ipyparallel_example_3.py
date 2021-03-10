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
    context.a = a


@click.command()
@click.option("--a", type=int, default=10)
@click.option("--verbose", is_flag=True)
def main(a, verbose):

    context().update(locals())

    if verbose:
        print('process: %i; value of a: %i' % (os.getpid(), context.a))

    # requires that $N number of remote workers first be instantiated from the command line with
    # ipcluster start -n $N
    context.client = Client()

    # syntax for importing python modules onto the remote worker processes
    context.client[:].execute('from ipyparallel_example_3 import *', block=True)

    # syntax for executing a function on all workers
    result = context.client[:].apply(report_a)
    if verbose:
        print('Before calling modify_a, calling report_a on all workers returns:\n%s' % str(result.get()))

    # syntax for calling a function with the same input arguments to all workers
    result = context.client[:].apply(modify_a, context.a)
    result.get()  # discard the result, modify doesn't return anything
    result = context.client[:].apply(report_a)
    if verbose:
        print('After calling modify_a with an apply operation, calling report_a on all workers returns:\n%s' %
              str(result.get()))

    # syntax for calling a function with different input arguments to each worker
    result = context.client[:].map(modify_a, range(len(context.client)))
    result.get()
    result = context.client[:].apply(report_a)
    if verbose:
        print('After calling modify_a with a map operation, calling report_a on all workers returns:\n%s' %
              str(result.get()))

    # syntax for requesting the value of a variable defined in the local namespace on each remote worker
    result = context.client[:].get('context.a')
    if verbose:
        print('Get operation returns:\n%s' % str(result))

    # Don't forget to shut down the ipcluster when you are done!


if __name__ == '__main__':
    main()