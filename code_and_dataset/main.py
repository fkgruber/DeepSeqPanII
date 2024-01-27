import os
import torch
import sys
from seq_encoding import one_hot_PLUS_blosum_encode
from config_parser import Config
from model import Model
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def read_hla_sequences():
    """Read hla sequences from [CLUATAL_OMEGA_B_chains_aligned_FLATTEN.txt]
    and [CLUATAL_OMEGA_A_chains_aligned_FLATTEN.txt]
    """
    def read(f, d):
        file_path = os.path.join(BASE_DIR, 'dataset', f)
        with open(file_path, 'r') as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                d[info[0]] = info[1]

    hla_sequence_A = {}
    hla_sequence_B = {}
    read('CLUATAL_OMEGA_A_chains_aligned_FLATTEN.txt', hla_sequence_A)
    read('CLUATAL_OMEGA_B_chains_aligned_FLATTEN.txt', hla_sequence_B)
    return hla_sequence_A, hla_sequence_B


def run(model_path, hla_a, hla_b, peptide,input_file,output_file):
    """Get ic50
    """
    # load model
    config = Config("config_main.json")
    config.device = 'cpu'
    if torch.cuda.is_available():
        state_dict = torch.load(os.path.join(BASE_DIR, model_path))
    else:
        state_dict = torch.load(os.path.join(BASE_DIR, model_path),map_location=torch.device('cpu'))
    model = Model(config)
    model.load_state_dict(state_dict)
    model.eval()
    # check if peptide is a file
    if os.path.exists(input_file):
        indf=pd.read_csv(input_file)
    else:
        indf=pd.DataFrame({"peptide":[peptide],"hla_a":[hla_a],"hla_b":[hla_b]})
    hla_sequence_A, hla_sequence_B = read_hla_sequences()
    scores=[]
    for (ii,rr) in tqdm(indf.iterrows()):
        peptide=rr["peptide"]
        hla_a=rr["hla_a"]
        hla_b=rr["hla_b"]
        peptide_encoded, pep_mask, pep_len = one_hot_PLUS_blosum_encode(peptide, config.max_len_pep)
        hla_a_seq = hla_sequence_A[hla_a]
        hla_b_seq = hla_sequence_B[hla_b]
        hla_a_encoded, hla_a_mask, hla_a_len = one_hot_PLUS_blosum_encode(hla_a_seq, config.max_len_hla_A)
        hla_b_encoded, hla_b_mask, hla_b_len = one_hot_PLUS_blosum_encode(hla_b_seq, config.max_len_hla_B)
        pred_ic50, _ = model(
            torch.stack([hla_a_encoded], dim=0),
            torch.stack([hla_a_mask], dim=0),
            torch.tensor([hla_a_len]),
            torch.stack([hla_b_encoded], dim=0),
            torch.stack([hla_b_mask], dim=0),
            torch.tensor([hla_b_len]),
            torch.stack([peptide_encoded], dim=0),
            torch.stack([pep_mask], dim=0),
            torch.tensor([pep_len]),
        )
        scores.append(pred_ic50.item())
    indf["DeepSeqPanII_score"]=scores
    ## create output directory if necessary
    outdir=os.path.dirname(output_file)
    if not os.path.exists(outdir) and outdir != "":
        os.makedirs(os.path.dirname(output_file))
    indf.to_csv(output_file,index=False)

if __name__ == '__main__':
    parser=ArgumentParser(
        description="DeepSeqPanII predictor"
    )
    parser.add_argument('--model-path',
                        dest="model_path",
                        type=str,
                        default=None,
                        help="The path to the model file")
    parser.add_argument('--hla-a',
                        dest="hla_a",
                        type=str,
                        default=None,
                        help="Alpha component of the HLA")
    parser.add_argument('--hla-b',
                        dest="hla_b",
                        type=str,
                        default=None,
                        help="Beta component of the HLA")
    parser.add_argument('--peptide',
                        dest="peptide",
                        type=str,
                        default=None,
                        help="Peptide")
    parser.add_argument('--input',
                        dest='input_file',
                        type=str,
                        default="",
                        help='Input file with HLA and peptide. If not provided then you need to provide hla-a, hla-b and peptide flags. If provided it takes precedence over hla-a, hla-b and peptide flags. Input file should be a csv file with columns peptide, hla_a, and hla_b.')
    parser.add_argument('--output',
                        dest='output_file',
                        type=str,
                        default="output.csv",
                        help='Output file with predictions. Default: output.csv')
    args=parser.parse_args()
    if args.input_file=="":
        if args.hla_b==None or args.hla_a==None or args.peptide==None:
            print("Please provide either input file or HLA and peptide")
            parser.print_help()
            sys.exit()
    run(
        model_path=args.model_path,
        hla_a=args.hla_a,
        hla_b=args.hla_b,
        peptide=args.peptide,
        input_file=args.input_file,
        output_file=args.output_file
    )
