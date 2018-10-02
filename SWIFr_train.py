class AODE_train():
	'''
	Given directory with component_stats.txt, scenarios.txt, and subdirectory simulations/
	containing subdirectories for each classification scenario with training examples, choose
	number of Gaussian mixture components for all distributions based on BIC minima, then
	save all necessary parameters in AODE_params/ and illustrations of learned distributions
	in component_statistic_distributions/
	'''

	def __init__(self,args):
		self.retrain = args.retrain
		self.readpkl = args.readpkl
		self.savepkl = args.savepkl
		self.singles_dict = {}
		self.tuples_dict  = {}
		self.path2allstats = args.path2files
		if self.path2allstats != '' and self.path2allstats[-1] != '/':
			self.path2allstats += '/'
		self.path2allstats += 'simulations/'
		self.path2files = args.path2files
		if self.path2files != '' and self.path2files[-1] != '/':
			self.path2files += '/'
		self.path2AODE = self.path2files+'AODE_params/'


		if os.path.isdir(self.path2AODE) == False:
			os.mkdir(self.path2AODE)

		file = open(self.path2files+'classes.txt','r')
		f = file.read()
		file.close()
		self.scenarios = [x.strip() for x in f.strip().splitlines()]



		file = open(self.path2files+'component_stats.txt','r')
		f = file.read()
		file.close()
		f = f.strip().splitlines()
		allstats = [x.strip() for x in f]
		if len(args.stats2use) == 0:
			self.statlist = allstats
		else:
			self.statlist = [x for x in allstats if x in args.stats2use]

		self.minscores = [[np.nan] for i in range(len(self.statlist))]
		self.maxscores = [[np.nan] for i in range(len(self.statlist))]

		self.num2stat = {i:self.statlist[i] for i in range(len(self.statlist))}
		self.stat2num = {y:x for x,y in self.num2stat.items()}		

		self.colors = ['blue','red','green','purple','orange'] #would need to add more colors for 6+ scenarios
		self.colorspectra = [cm.Blues,cm.Reds,cm.Greens,cm.Purples,cm.Oranges] #would need to add more colors for 6+ scenarios

		self.component_nums_1D = [['n/a' for s in self.scenarios] for x in self.statlist]
		self.component_nums_2D = [[['n/a' for s in self.scenarios] for y in self.statlist] for x in self.statlist]

		if os.path.isdir(self.path2files+'BIC_plots/') == False:
			os.mkdir(self.path2files+'BIC_plots/')
		if os.path.isdir(self.path2files+'component_statistic_distributions/') == False:
			os.mkdir(self.path2files+'component_statistic_distributions')
			os.mkdir(self.path2files+'component_statistic_distributions/marginals')
			os.mkdir(self.path2files+'component_statistic_distributions/joints')
		if self.readpkl == False:
			self.read_in_all()


	def tuples(self,stat1,stat2,scenario,from_pkl=False,save_pkl=False):
		stat = "{}-{}".format(stat1,stat2)
		try: # If you hate this, replace with "if stat in self.singles_dict.keys():"
			scores = self.tuples_dict[stat]
		except KeyError: # Change this to an "else:" However, querrying a dict is fast, checking a list is slow
			if from_pkl == True:
				print 'reading from PKL: '+scenario+' joint distributions for '+stat1+' and '+stat2+'...'
				self.tuples_dict[stat] = pd.read_pickle(self.path2AODE+stat1+'_'+stat2+'_'+scenario+'_tuples.p')
			else:
				mask   = (self.train_data[scenario][stat1] != -998) & (self.train_data[scenario][stat2] != -998)
				self.tuples_dict[stat] = self.train_data[scenario].loc[mask, (stat1,stat2)]
				if save_pkl == True:
					print 'saving '+scenario+' joint distributions for '+stat1+' and '+stat2+'...'
					self.tuples_dict[stat].to_pickle(self.path2AODE+stat1+'_'+stat2+'_'+scenario+'_tuples.p')
			scores = self.tuples_dict[stat]
		return np.array(scores)

	def singles(self,stat,scenario,from_pkl=False,save_pkl=False):
		try: # If you hate this, replace with "if stat in self.singles_dict.keys():"
			scores = self.singles_dict[stat]
		except KeyError: # Change this to an "else:" However, querrying a dict is fast, checking a list is slow
			if from_pkl == True:
				print 'reading from PKL: '+scenario+' marginal distributions for '+stat+'...'
				self.singles_dict[stat] = pd.read_pickle(self.path2AODE+stat+'_'+scenario+'_singles.p')
			else:
				mask   = self.train_data[scenario][stat] != -998
				self.singles_dict[stat] = self.train_data[scenario].loc[mask, stat]
				if save_pkl == True:
					print 'saving '+scenario+' marginal distributions for '+stat+'...'
					self.singles_dict[stat].to_pickle(self.path2AODE+stat+'_'+scenario+'_singles.p')
			scores = self.singles_dict[stat]
		SCORES = np.expand_dims(np.array(scores), axis=1)
		minscore = min(SCORES)[0]
		maxscore = max(SCORES)[0]
		RANGE = maxscore-minscore
		self.minscores[self.stat2num[stat]][0] = minscore
		self.maxscores[self.stat2num[stat]][0] = maxscore
		return np.expand_dims(np.array(scores), axis=1)

	def plot_bic(self,stat1,stat2,scenario):
		S = self.tuples(stat1,stat2,scenario,from_pkl=self.readpkl)
		BICs_full = []
		for n in range(1,11):
			H = mixture.GMM(n_components=n,covariance_type='full')
			H.fit(S)
			#AICs_full.append(H.aic(S))
			BICs_full.append(H.bic(S))
		minbic = min(BICs_full)
		argminbic = BICs_full.index(minbic)+1
		print 'number of components for '+scenario+': '+str(argminbic)
		plt.plot(range(1,11),BICs_full,'o-',color='darkblue',ms=5,markeredgecolor='none')
		plt.plot(argminbic,minbic,'o-',color='coral',ms=5,markeredgecolor='red')
		plt.xlabel('number of Gaussian mixture components')
		plt.ylabel('BIC')
		plt.savefig(self.path2files+'BIC_plots/'+stat1+'_'+stat2+'_'+scenario+'_BIC.pdf')
		plt.clf()

		self.component_nums_2D[self.stat2num[stat1]][self.stat2num[stat2]][self.scenarios.index(scenario)] = argminbic

	def plot_bic_1D(self,stat,scenario):
		S = self.singles(stat,scenario,from_pkl=self.readpkl)
		BICs = []
		for n in range(1,11):
			H = mixture.GMM(n_components=n)
			H.fit(S)
			#AICs.append(H.aic(S))
			BICs.append(H.bic(S))

		minbic = min(BICs)
		argminbic = BICs.index(minbic)+1
		print 'number of components for '+scenario+': '+str(argminbic)			
		plt.xlabel('number of Gaussian mixture components')
		plt.ylabel('BIC')
		plt.plot(range(1,11),BICs,'o-',color='darkblue',ms=5,markeredgecolor='none')
		plt.plot(argminbic,minbic,'o-',color='coral',ms=5,markeredgecolor='red')
		plt.savefig(self.path2files+'BIC_plots/'+stat+'_'+scenario+'_BIC.pdf')
		plt.clf()
		self.component_nums_1D[self.stat2num[stat]][self.scenarios.index(scenario)] = argminbic

	def read_in_all(self):
		print 'Begin reading all training data'
		self.train_data = { s_:pd.concat([ pd.read_table(os.path.join(self.path2allstats+s_+'/',f),
                                                         header=0, index_col=False) 
                                           for f in os.listdir(self.path2allstats+s_+'/') 
                                           if f[0] != '.' ], 
                                         ignore_index=True) 
                           for s_ in self.scenarios }
		print 'Finished loading data'
		if self.savepkl == False:
			print 'skip distribution pickling'
		for stat in self.statlist:
			for scenario in self.scenarios:
				self.singles(stat,scenario,from_pkl=False,save_pkl=self.savepkl)

		for i in range(len(self.statlist)-1):
			for j in range(i+1,len(self.statlist)):
				for scenario in self.scenarios:
					self.tuples(self.statlist[i],self.statlist[j],scenario,from_pkl=False,save_pkl=self.savepkl)

	def run_bic(self):
		for stat in self.statlist:
			print 'learning number of Gaussian mixture components for '+stat
			for scenario in self.scenarios:
				self.plot_bic_1D(stat,scenario)

		for i in range(len(self.statlist)-1):
			for j in range(i+1,len(self.statlist)):
				print 'learning number of Gaussian mixture components for joint '+self.statlist[i]+', '+self.statlist[j]
				for scenario in self.scenarios:
					self.plot_bic(self.statlist[i],self.statlist[j],scenario)

		#write marginal component_nums file
		out = open(self.path2AODE+'marginal_component_nums','w')
		header = 'statistic\t'
		for scenario in self.scenarios:
			header += scenario+'\t'
		header = header.strip()
		out.write(header+'\n')
		for i in range(len(self.component_nums_1D)):
			line = self.num2stat[i]+'\t'
			for scenario in self.scenarios:
				line += str(self.component_nums_1D[i][self.scenarios.index(scenario)])+'\t'
			line = line.strip()
			out.write(line+'\n')
		out.close()

		#write joint component_nums file
		out = open(self.path2AODE+'joint_component_nums','w')
		header = 'stat1\tstat1\t'
		for scenario in self.scenarios:
			header += scenario+'\t'
		header = header.strip()
		out.write(header+'\n')
		for i in range(len(self.component_nums_2D)-1):
			for j in range(i+1,len(self.component_nums_2D[i])):
				line = self.num2stat[i]+'\t'+self.num2stat[j]+'\t'
				for scenario in self.scenarios:
					line += str(self.component_nums_2D[i][j][self.scenarios.index(scenario)])+'\t'
				line = line.strip()
				out.write(line+'\n')
		out.close()

	def gmm_fit(self,stat1,stat2,scenario):
		S = self.tuples(stat1,stat2,scenario,from_pkl=self.readpkl)
		G = mixture.GMM(n_components=self.component_nums_2D[self.stat2num[stat1]][self.stat2num[stat2]][self.scenarios.index(scenario)],covariance_type='full')
		G.fit(S)
		pickle.dump(G,open(self.path2AODE+stat1+'_'+stat2+'_'+scenario+'_GMMparams.p','wb'))
		return G

	def gmm_fit_1D(self,stat,scenario):
		S = self.singles(stat,scenario,from_pkl=self.readpkl)
		G = mixture.GMM(n_components=self.component_nums_1D[self.stat2num[stat]][self.scenarios.index(scenario)])
		G.fit(S)
		pickle.dump(G,open(self.path2AODE+stat+'_'+scenario+'_1D_GMMparams.p','wb'))
		return G

	def plot_gmm_marginals(self,stat):
		fig = plt.figure()
		for scenario in self.scenarios:
			G = self.gmm_fit_1D(stat,scenario)
			mu = G.means_
			sigma = G.covars_
			w = G.weights_
			minscore = min(self.minscores[self.stat2num[stat]])
			maxscore = max(self.maxscores[self.stat2num[stat]])
			x = np.linspace(minscore,maxscore,100)
			Z = 0
			for i in range(len(w)):
				Z = Z + w[i]*normpdf(x,mu[i],sigma[i])
			plt.plot(x,Z,self.colors[self.scenarios.index(scenario)%len(self.colors)])
		plt.xlabel(stat)
		plt.ylabel('frequency')
		plt.legend(self.scenarios)
		plt.savefig(self.path2files+'component_statistic_distributions/marginals/'+stat+'_marginal.pdf')
		plt.clf()

	def plot_gmm_contour(self,stat1,stat2):

		fig = plt.figure()
		legendlocations = {x:[] for x in self.scenarios}
		for scenario in self.scenarios:
			G = self.gmm_fit(stat1,stat2,scenario)
			mu = G.means_
			sigma = G.covars_
			legendlocations[scenario] = [mu[0][0]+.2*sigma[0][0][0],mu[0][1]+.2*sigma[0][1][1]]			
			w = G.weights_
			minscore1 = min(self.minscores[self.stat2num[stat1]])
			maxscore1 = max(self.maxscores[self.stat2num[stat1]])
			minscore2 = min(self.minscores[self.stat2num[stat2]])
			maxscore2 = max(self.maxscores[self.stat2num[stat2]])							
			x = np.linspace(minscore1,maxscore1,100)
			y = np.linspace(minscore2,maxscore2,100)
			X,Y = np.meshgrid(x,y)
			Z = 0
			for i in range(len(w)):
				Z = Z + w[i]*bivariate_normal(X,Y,mux=mu[i][0],muy=mu[i][1],sigmax=math.sqrt(sigma[i][0][0]),sigmay=math.sqrt(sigma[i][1][1]),sigmaxy=sigma[i][0][1])
			C = plt.contour(X,Y,Z,10,cmap=self.colorspectra[self.scenarios.index(scenario)%len(self.colorspectra)])

		plt.xlabel(stat1)
		plt.ylabel(stat2)
		for scenario in self.scenarios:
			plt.text(legendlocations[scenario][0],legendlocations[scenario][1],scenario,color=self.colors[self.scenarios.index(scenario)%len(self.colors)])
		plt.savefig(self.path2files+'component_statistic_distributions/joints/'+stat1+'_'+stat2+'_joint.pdf')
		plt.clf()

	def plot_contours(self):
		for i in range(len(self.statlist)-1):
			for j in range(i+1,len(self.statlist)):
				stat1 = self.statlist[i]
				stat2 = self.statlist[j]
				self.plot_gmm_contour(stat1,stat2)
		for stat in self.statlist:
			self.plot_gmm_marginals(stat)

	def retrain_classifier(self):
		file = open(self.path2AODE+'marginal_component_nums','r')
		f = file.read()
		file.close()
		f = f.strip().splitlines()[1:]
		for i in range(len(f)):
			f[i] = f[i].strip().split('\t')
			stat = f[i][0]
			if stat in self.statlist:
				for j in range(len(self.scenarios)):
					self.component_nums_1D[self.stat2num[stat]][j] = int(f[i][j+1])

		file = open(self.path2AODE+'joint_component_nums','r')
		f = file.read()
		file.close()
		f = f.strip().splitlines()[1:]
		for i in range(len(f)):
			f[i] = f[i].strip().split('\t')
			stat1 = f[i][0]
			stat2 = f[i][1]
			if stat1 in self.statlist and stat2 in self.statlist:
				for j in range(len(self.scenarios)):
					self.component_nums_2D[self.stat2num[stat1]][self.stat2num[stat2]][j] = int(f[i][j+2])

		self.plot_contours()

if __name__ == '__main__':
	from scipy.stats import norm
	import math,sys,argparse, pickle, os
	sys.path.append(os.getcwd())	
	import matplotlib
	matplotlib.use('agg')
	from matplotlib import pyplot as plt
	import numpy as np
	import pandas as pd
	from sklearn import mixture
	from matplotlib.mlab import bivariate_normal
	from matplotlib.mlab import normpdf
	import matplotlib.cm as cm 	

	parser = argparse.ArgumentParser()
	parser.add_argument('--path',action='store',dest='path2files',default='') #path to all input files (simulations in a 'simulations' directory, and compstats, scenarios files)
	parser.add_argument('--retrain',action='store_true',dest='retrain')
	parser.add_argument('--readpkl',action='store_true',dest='readpkl')
	parser.add_argument('--savepkl',action='store_true',dest='savepkl',help="Save distributions as PKLs.")
	parser.add_argument('--stats2use',action='store',nargs='+',default=[]) #use to split training into parallel runs, only with --retrain

	args = parser.parse_args()
	A = AODE_train(args)

	if args.retrain:
		A.retrain_classifier()
		print 'Training complete. Run SWIFr.py with --path2trained '+args.path2files
	else:
		A.run_bic()
		A.plot_contours()
		print 'Training complete. Run SWIFr.py with --path2trained '+args.path2files

