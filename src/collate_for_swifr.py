import os
import sys
import argparse
from itertools import chain

import yaml
import h5py
import numpy as np

def get_args():    
    parser = argparse.ArgumentParser(description='Aggregate simulate data for SWIF(r).')
    parser.add_argument('--h5_data','-d',required=True,help='HDF5 data container.')
    parser.add_argument('--data_split','-y',required=True,help='YAML describing data organization.')
    parser.add_argument('--yaml_address','-a',required=True,help='Comma separated list of keys needed to retrieve HDF5 paths.')
    parser.add_argument('--selected_sites','-os',required=True,type=str,help='Output path for loci under selection')
    parser.add_argument('--neutral_sites','-on',required=True,type=str,help='Output path for neutral loci')
    parser.add_argument('--selected_position','-p',type=int,default=1500000,help='position of adaptive variant.')
    parser.add_argument('--flank_count','-c',type=int,default=500,help='Maximum number of neutral valiants to collect from each side of selected variant.')
    parser.add_argument('--header','-r',type=str,help='Optional headers for component scores. Provide as comma delimited list.')
    args = parser.parse_args()
    args = parser.parse_args()
    return args

def main(args):
    H5_D    = args.h5_data
    D_SPLIT = args.data_split
    Y_ADDRS = args.yaml_address
    O_SEL   = args.selected_sites
    O_NEU   = args.neutral_sites
    SEL_POS = args.selected_position
    F_CT    = args.flank_count
    HEAD    = args.header
    
    ###############################################
    ##                                           ##
    ## Get list of HDF5 addresses from YAML file ##
    ##                                           ##
    ###############################################
    
    with open(D_SPLIT,'r') as f:
        h5_info = yaml.load(f) # YAML is loaded as a dictionary
    layer_ref = h5_info
    for layer_key in Y_ADDRS.split(','): # Step through dictionary layers
        layer_ref = layer_ref[layer_key]
        
    print("HDF5 keys loaded",file=sys.stderr)
        
    ####################################################
    ##                                                ##
    ## Iter through HDF5 addresses and print to files ##
    ##                                                ##
    ####################################################
        
    with h5py.File(H5_D, 'r') as data, open(O_SEL, 'w') as oh_s, open(O_NEU, 'w') as oh_n:
        if HEAD: # Print a header if we've defined it
            print("\t".join(['SNP_ID']+HEAD.split(',')), file=oh_s)
            print("\t".join(['SNP_ID']+HEAD.split(',')), file=oh_n)
        n_keys = len(layer_ref)
        for i, h5_path in enumerate(layer_ref):
            if i % (n_keys//10) == 0:
                print('Printed [{}/{}] regions.'.format(i,n_keys),file=sys.stderr)
            ## Check for a selected allele
            if np.sum(data['positions/{}'.format(h5_path)][:] == SEL_POS) == 1:
                sel_idx = np.argmax(data['positions/{}'.format(h5_path)][:] == SEL_POS)
            else:
                sel_idx = None
                
            ## Pull data for selected allele if exists, and print
            if sel_idx is not None:
                line = [ '-998' if np.isnan(x) else str(x) 
                         for x in data['components/{}'.format(h5_path)][sel_idx] ]
                line = ['{}:{}'.format(h5_path,sel_idx)] + line
                print( '\t'.join(line), file=oh_s )
            ## If no selected allele, notify user and use the middle index a placeholder
            else:
                print('No selected locus in {}'.format(h5_path), file=sys.stderr)
                sel_idx = len( data['positions/{}'.format(h5_path)] ) // 2
            
            ## Move through positions flanking selected allele and print to alternate file
            slice_start = max(sel_idx-F_CT,0)
            slice_end   = min(sel_idx+1+F_CT,data['components/{}'.format(h5_path)].shape[0])
            slice_end   = max(sel_idx+1,slice_end)
            for row_idx in chain(range(slice_start,sel_idx), range(sel_idx+1,slice_end)):
                line = [ '-998' if np.isnan(x) else str(x) 
                         for x in data['components/{}'.format(h5_path)][row_idx] ]
                line = ['{}:{}'.format(h5_path,row_idx)] + line
                print( '\t'.join(line), file=oh_n )
                
    print("Done.",file=sys.stderr)

if __name__ == '__main__':
    args = get_args()
    main(args)