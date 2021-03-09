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


@click.command()
@click.option("--a", type=int, default=10)
@click.option("--verbose", is_flag=True)
def main(a, verbose):
    context().update(locals())

    # execute on $N number of processes from the command line with:
    # mpirun -n $N python mpi4py_example_1.py

    context.comm = MPI.COMM_WORLD

    if verbose:
        print('process: %i; MPI rank/size: %i/%i; value of a: %i' %
              (os.getpid(), context.comm.rank, context.comm.size, context.a))


if __name__ == '__main__':
    main()