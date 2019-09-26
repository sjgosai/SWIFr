#python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a defdef15,train -os data/defdef15__train__selected.txt -on data/defdef15__train__neutral.txt -c 500 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a defdef15,valid -os data/defdef15__valid__selected.txt -on data/defdef15__valid__neutral.txt -c 25000 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF


#python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a gradient15,train -os data/gradient15__train__selected.txt -on data/gradient15__train__neutral.txt -c 500 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

#python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a gradient15,extra -os data/gradient15__extra__selected.txt -on data/gradient15__extra__neutral.txt -c 500 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a gradient15,valid -os data/gradient15__valid__selected.txt -on data/gradient15__valid__neutral.txt -c 25000 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF


#python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a gravel,train -os data/gravel__train__selected.txt -on data/gravel__train__neutral.txt -c 500 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

#python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a gravel,extra -os data/gravel__extra__selected.txt -on data/gravel__extra__neutral.txt -c 500 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a gravel,valid -os data/gravel__valid__selected.txt -on data/gravel__valid__neutral.txt -c 25000 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF


#python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a nd,train -os data/nd__train__selected.txt -on data/nd__train__neutral.txt -c 500 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

#python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a nd,extra -os data/nd__extra__selected.txt -on data/nd__extra__neutral.txt -c 500 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a nd,valid -os data/nd__valid__selected.txt -on data/nd__valid__neutral.txt -c 25000 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF


#python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a ndcs,train -os data/ndcs__train__selected.txt -on data/ndcs__train__neutral.txt -c 500 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

#python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a ndcs,extra -os data/ndcs__extra__selected.txt -on data/ndcs__extra__neutral.txt -c 500 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

python src/collate_for_swifr.py -d /n/local/cms/data/organized_data.h5 -y /n/local/cms/data/organized_data__shuffle.yml -a ndcs,valid -os data/ndcs__valid__selected.txt -on data/ndcs__valid__neutral.txt -c 25000 -r maf,ihs,delihh,nsl,h12,h2h1,iSAFE,XPEHH,Fst,delDAF

