import click
import os
from mpi4py import MPI


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

    # execute on $N number of processes from the command line with:
    # mpirun -n $N python mpi4py_example_3.py

    context.comm = MPI.COMM_WORLD

    if verbose:
        print('process: %i; MPI rank/size: %i/%i; value of a: %i' %
              (os.getpid(), context.comm.rank, context.comm.size, context.a))

    result = context.comm.gather(context.a, root=0)
    if verbose and context.comm.rank == 0:
        print('Before scatter, gather returns: %s' % str(result))

    context.a = context.comm.scatter(range(context.comm.size), root=0)

    result = context.comm.gather(context.a, root=0)
    if verbose and context.comm.rank == 0:
        print('After scatter, gather returns: %s' % str(result))

    modify_a(context.a * 2)
    result = report_a()
    if verbose:
        print('After calling modify_a, calling report_a on rank %i returns:\n%s' % (context.comm.rank, str(result)))

    context.a = context.comm.bcast(context.a, root=0)

    result = context.comm.gather(context.a, root=0)
    if verbose and context.comm.rank == 0:
        print('After broadcast, gather returns: %s' % str(result))


if __name__ == '__main__':
    main()