#The development began around 2019-02-21
import os
import json
import pprint as pp
import argparse
import time
from collections import OrderedDict

from tools.spec import SPEC
from model.meta import META
from model.model import MODEL

def main():
    parser = argparse.ArgumentParser(description="PepGo for de novo peptide sequencing")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")
    
    #Global arguments 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('-c', '--config', type=str, dest='config', default= os.path.join(current_dir, 'config.json'),
                        help='Configure file')

    #Converter arguments
    tospec = subparsers.add_parser("tospec", help="Convert input files to .spec format")
    tospec.add_argument("-t", "--type", type=str, dest='type', choices=['mgf', 'msp', 'mzML'], required=True,
                        help="Input file type: mgf, msp, mzML ...")
    tospec.add_argument('input', type=str, default=None, help="Name of the file to input")
    specto = subparsers.add_parser("specto", help="Convert .spec file to other formats")

    #Train arguments
    train = subparsers.add_parser('train', help='Training Transformer models')
    #train.add_argument('input', type=str, default=None, help="Name of the spec file for training")
    train.add_argument('-T', '--Transformers', type=str, dest='Transformers',default=None,
                        help="Directory to save two Transformer models(.ckpt)")
    train.add_argument('-t', '--train', type=str, dest='train',default=None, help="Spec file for training")
    train.add_argument('-v', '--valid', type=str, dest='valid',default=None, help="Spec file for validation")
    
    #Predict arguments
    predict = subparsers.add_parser('predict', help='Predicting peptides from spectra')
    predict.add_argument('-T', '--Transformers', type=str, dest='Transformers',default=None,
                        help="Directory containing the two Transformer models(.ckpt)")
    predict.add_argument('input', type=str, default=None, help="Name of the spec file for prediction")

    args = parser.parse_args()
    print('args:')
    pp.pprint(args)

    configs = read_configs(args.config)
    meta = META(configs)
    model = MODEL(meta, configs)

    if(args.command == "tospec"):
        spec=SPEC(meta)
        if(args.type=='mgf'):
            spec.convert_mgf_to_spec(args.input)
        elif(args.type=='msp'):
            spec.convert_msp_to_spec(args.input)
    elif(args.command == "train"):
        model.initialize_model(mode='train', models_dir=args.Transformers)
        model.train(train_spec=args.train, valid_spec=args.valid)
    elif(args.command == "predict"):
        start=time.time()
        #model.initialize_trainer(train=False)
        model.initialize_model(mode='predict', models_dir=args.Transformers)
        model.predict(args.input)
        end=time.time()
        print('Total_time_consumed in prediction:',end=':')
        print(end-start)
    else:
        pass    

def read_configs(config_file):
    with open(config_file, 'r') as f:
        configs = json.load(f, object_pairs_hook=OrderedDict)
    return(configs)

if __name__ == "__main__":
    main()
