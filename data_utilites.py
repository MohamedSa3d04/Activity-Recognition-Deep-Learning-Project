import argparse

def try_fun(name):
    return name


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Name')
    args = parser.parse_args()

    try_fun(args.name)