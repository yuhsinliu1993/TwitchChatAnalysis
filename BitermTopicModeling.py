import os


class BBTM:

	def __init__(self, docs_dir, res_dir):
		self.docs_dir = docs_dir
		self.res_dir = res_dir
		self.word2id = {}   # store  { word: [id, freq], ... }

	def indeXing(self):
		for f in os.listdir(self.docs_dir):
			self.index_file(self.docs_dir+'/'+f, self.res_dir+'/indexed_log.txt')

	def index_file(self, doc, res_file):
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

		self.save_word2id(self.res_dir+'/vocabulary.txt')
		wf.close()

	def save_word2id(self, file):
		with open(file, 'w') as f:
			for word, id_f in sorted(self.word2id.items(), key=lambda d:d[1]):
				f.write('%d\t%s\t%d\n' % (id_f[0], word, id_f[1]))
