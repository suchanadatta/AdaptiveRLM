import pyterrier as pt
pt.init()
from pyterrier_t5 import MonoT5ReRanker
import argparse
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('dataset')
    parser.add_argument('output')
    args = parser.parse_args()
    dataset = pt.get_dataset(args.dataset)
    monot5 = MonoT5ReRanker()
    pipeline = pt.text.get_text(dataset, 'text') >> monot5
    print(pipeline)
    input = pt.io.read_results(args.input, dataset=dataset)
    print(input)
    pdb.set_trace()
    output = pipeline(input)
    print(output)
    pt.io.write_results(output, args.output)

if __name__ == '__main__':
  main()