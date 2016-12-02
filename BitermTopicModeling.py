import os
from collections import Counter, defaultdict

class BBTM:

	def __init__(self):
		# self.docs_dir = docs_dir 	# ../reckful/cleaned_logs/
		# self.res_dir = res_dir		# ../reckful/output/   ( dwid_dir )
		self.word2id = {}   # store  { word: [id, freq], ... }
		self.dwid_file = ''

	def File_indeXing(self, docs_dir, res_dir):
		"""
			Map each word to a unique ID (starts from 0) in the documents.
			Input Dir: 	docs_dir  e.g. ../reckful/cleaned_logs_dir/
			Output Dir: res_dir	  e.g. ../reckful/output/
		"""
		for fn in os.listdir(docs_dir):
			fname = os.path.join(docs_dir, fn)
			fout = os.path.join(res_dir, fn)
			self._index(fname, fout)

	def _index(self, doc, res_file):
		wf = open(res_file, 'w')
		with open(doc, 'r') as f:
			for line in f:
				tokens = line.strip().split()
				for token in tokens:
					if token in self.word2id:
						self.word2id[token][1] += 1
					else:
						self.word2id[token] = [len(self.word2id), 1]  

				# Get all word's ids in sentence 
				word_ids = [self.word2id[token][0] for token in tokens] 
				ids = ' '.join(map(str, word_ids))
				wf.write(ids+'\n')

		self.save_word2id(res_dir+'/vocabulary.txt')
		wf.close()

	def save_word2id(self, file):
		with open(file, 'w') as f:
			for word, id_f in sorted(self.word2id.items(), key=lambda d:d[1]):
				f.write('%d\t%s\t%d\n' % (id_f[0], word, id_f[1]))

	def proc_indexfile(self, filename):
		# self.res_dir+'/biterm_dayfreq.txt'
		biterm_freq = defaultdict(str)
		fn = res_dir + '/indexed_log.txt'


	def bitermFreq(self, file):
	    bf = Counter()
	    with open(file, 'r') as f:
	    	for l in f:
		        ws = map(int, l.strip().split())
		        bs = self.genBiterms(ws)
		        bf.update(bs)
		        
	    return bf

	def genBiterms(ws):
		bs = []
		# Since somebody may run this code on normal texts, 
		# I add a window size.
		win = 15  
		for i in range(len(ws)-1):
		    for j in range(i+1, min(i+win+1, len(ws))):
		        wi = min(ws[i], ws[j])
		        wj = max(ws[i], ws[j])
		        #if wi == wj: continue
		        b = '%d %d' % (wi, wj)
		        bs.append(b)
		return bs 
