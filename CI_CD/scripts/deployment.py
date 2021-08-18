import argparse

parser = argparse.ArgumentParser(description='Parse Model ID.')
parser.add_argument('new_best_run_id', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')

arg = parser.parse_args()

def deployment(new_best_run_id):
    # here you can write you deployment logic!
    print(f'Model with Run ID {new_best_run_id} Deployed succesfully!')


deployment(arg.new_best_run_id)