#The development began around 2019-02-21
import os
import sys
import json
import pprint as pp
import argparse
import time
from collections import OrderedDict

from tools.spec import SPEC
from tools.MGFConverter import MGFConverter

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
    mgfconverter = subparsers.add_parser("mgfconvert", help="Convert input mgf files to multiple formats")
    mgfconverter.add_argument("-x", "--xml", type=str, dest='xml',help="XML file along with the mgf file")
    mgfconverter.add_argument('input', type=str, default=None, help="Name of the input mgf file")
    mgfconverter.add_argument('-o', '--output', type=str, dest='output', default=None, help="Name of the output file")
    mgfconverter.add_argument('-s', '--scan', type=str, dest='scan_table', default=None, help="Name of the scan table")
    mgfconverter.add_argument('-p', '--ptm', type=str, dest='ptm', default=None, help="Name of the mass-ptm table")
    mgfconverter.add_argument('-f', '--outformat', type=str, dest='outformat', default=None, help="Format of output file")
    mgfconverter.add_argument('-i', '--informat', type=str, dest='informat', default=None, help="Format of input file")

    #spec arguments
    tospec = subparsers.add_parser("tospec", help="Convert input files to .spec format")
    tospec.add_argument("-t", "--type", type=str, dest='type', choices=['mgf', 'msp', 'mzML'], required=True,
                        help="Input file type: mgf, msp, mzML ...")
    tospec.add_argument('input', type=str, default=None, help="Name of the file to input")
    #specto = subparsers.add_parser("specto", help="Convert .spec file to other formats")

    #Train arguments
    train = subparsers.add_parser('train', help='Training Transformer models')
    #train.add_argument('input', type=str, default=None, help="Name of the spec file for training")
    train.add_argument('-T', '--Transformers', type=str, dest='Transformers',default='ckpt',
                        help="Directory to save two Transformer models(./ckpt)")
    train.add_argument('-t', '--train', type=str, dest='train',default=None, help="Spec file for training")
    train.add_argument('-v', '--valid', type=str, dest='valid',default=None, help="Spec file for validation")

    #Predict arguments
    predict = subparsers.add_parser('predict', help='Predicting peptides from spectra')
    predict.add_argument('-T', '--Transformers', type=str, dest='Transformers',default=None,
                        help="Directory containing the two Transformer models(.ckpt)")
    predict.add_argument('input', type=str, default=None, help="Name of the spec file for prediction")

    args = parser.parse_args()
    #print('args:')
    #pp.pprint(args)

    configs = read_configs(args.config)
    meta = META(configs)
    model = MODEL(meta, configs)

    if(args.command == 'mgfconvert'):
        mgf_converter = MGFConverter(meta, args.informat, args.outformat)
        #mgf_converter.index_mgf(args.input, args.xml)
        #mgf_converter.extract_ptms(args.input, 'tmp.ptm')
        if(args.ptm is not None):
            mgf_converter.readin_mass_ptm_dicts(args.ptm)
        if(args.scan_table is not None):
            mgf_converter.readin_mass_scan_table(args.scan_table)

        if(args.informat=='MassIVE_KB' and args.outformat=='Casanovo'):
            casanovomgf_file = mgf_converter.convert_MassiveMGF_to_CasanovoMGF(args.input, casanovomgf_file=args.output)
        elif(args.informat=='MassIVE_KB' and args.outformat=='PointNovo'):
            PointNovo_mgf_file, PointNovo_csv_file = mgf_converter.convert_MassiveMGF_to_PointNovo(args.input)
        elif (args.informat=='MassIVE_KB' and args.outformat == 'PrimeNovo'):
            PointNovo_mgf_file = mgf_converter.convert_MassiveMGF_to_PrimeNovo(mgf_file=args.input, PrimeNovo_mgf_file=args.output)
        elif(args.informat=='MassIVE_KB' and args.outformat=='PepGo'):
            spec_file = mgf_converter.convert_MassiveKBmgf_to_PepGo(args.input, preprocess=True)
            #mgf_converter.convert_spec_to_h5(spec_file)
        elif(args.informat=='9species' and args.outformat=='PepGo'):
            spec_file = mgf_converter.convert_9SpeciesMGF_to_PepGo(args.input, preprocess=True, spec_file=args.output)
            mgf_converter.convert_spec_to_h5(spec_file)
        elif (args.informat == 'PepGo' and args.outformat == 'H5'):
            mgf_converter.convert_spec_to_h5(args.input)
        else:
            raise ValueError('Nothing converted')
        '''
        mgf_converter.convert_mgf_to_spec(args.input)
        mgf_converter.convert_mgf_to_Casanovo(args.input)
        mgf_converter.convert_mgf_to_Pointnovo(args.input)
        mgf_converter.convert_mgf_to_PrimeNovo(args.input)
        mgf_converter.convert_mgf_to_InstaNovo(args.input)
        '''

    if(args.command == "tospec"):
        spec=SPEC(meta)
        if(args.type=='mgf'):
            spec.convert_mgf_to_spec(args.input)
        elif(args.type=='msp'):
            spec.convert_msp_to_spec(args.input)
    elif(args.command == "train"):
        model.initialize_models(mode='train', models_dir=args.Transformers)
        model.train(train_spec=args.train, valid_spec=args.valid)
    elif(args.command == "predict"):
        start=time.time()
        model.initialize_models(mode='predict', models_dir=args.Transformers)
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
