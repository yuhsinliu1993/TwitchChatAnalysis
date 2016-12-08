import os
from collections import Counter, defaultdict


class BTM:

	def __init__(self, num_topics):
		self.word2id = {}
		self.num_topics = num_topics
		self.topics_dict = {}

	def FileIndeXing(self, in_file, out_dir):
		wf = open(os.path.join(out_dir, 'doc_wids.txt'), 'w')
		with open(in_file, 'r') as f:
			for line in f:
				tokens = line.strip().split()
				for token in tokens:
					if token not in self.word2id:
						self.word2id[token] = len(self.word2id)

				word_ids = [self.word2id[token] for token in tokens]
				ids = ' '.join(list(map(str, word_ids)))
				wf.write(ids+'\n')

		wf.close()

		print('[*] Write indexed file: %s' % os.path.join(out_dir, 'doc_wids.txt'))

		self.save_word2id(os.path.join(out_dir, 'vocabulary.txt'))

	def save_word2id(self, out_file):
		with open(out_file, 'w') as f:
			for word, _id in sorted(self.word2id.items(), key=lambda d:d[1]):
				f.write('%d\t%s\n' % (_id, word))		
		print('[*] Write vocabulary file: %s' % out_file)

	def load_vocabulary(self, path_dir):
		voca = {}
		for l in open(os.path.join(path_dir, 'output', 'vocabulary.txt')):
			wid, w = l.strip().split('\t')[:2]
			voca[int(wid)] = w

	def get_topics_distributions(self, output_dir, show=False, save=True):
		print('Topics: %d\tn(W): %d' % (self.num_topics, len(self.word2id)))

		voca_pt = os.path.join(output_dir, 'vocabulary.txt')
		pz_pt = os.path.join(output_dir, 'model', 'k%d.pz' % self.num_topics)
		zw_pt = os.path.join(output_dir, 'model', 'k%d.pw_z' % self.num_topics)
		pzd_pt = os.path.join(output_dir, 'model', 'k%d.pz_d' % self.num_topics)

		voca = {}
		for l in open(voca_pt, 'r'):
			wid, w = l.strip().split('\t')[:2]
			voca[int(wid)] = w

		pz = [float(p) for p in open(pz_pt, 'r').readline().split()]

		k = 0
		topics = []
		for l in open(zw_pt, 'r'):
			vs = [float(v) for v in l.split()]
			wvs = zip(range(len(vs)), vs)
			wvs = sorted(wvs, key=lambda d:d[1], reverse=True)
			#tmps = ' '.join(['%s' % voca[w] for w,v in wvs[:10]])
			tmps = ' '.join(['%s:%f' % (voca[w],v) for w, v in wvs[:15]])
			topics.append((pz[k], tmps))
			k += 1

		for i in range(len(topics)):
			self.topics_dict[i] = topics[i]

		t_distributions = []
		for line in open(pzd_pt, 'r'):
			topic_distributions = [float(prob) for prob in line.split()]
			maximum_p = sorted(topic_distributions)[-1]
			for i in range(len(topic_distributions)):
				if maximum_p == topic_distributions[i]:
					t_distributions.append(i+1)

		if show:
			print("================ Topic Display ================")
			print("K\tp(z)\t\tTop words")
			for key in self.topics_dict:
				print('%d\t%f\t%s' % (key+1, self.topics_dict[key][0], self.topics_dict[key][1]))

		if save:
			saved = os.path.join(output_dir, 'topics.txt')
			print("[*] Saving topics distributions to %s" % saved)
			self._save_topics(saved, 0.01)

		return t_distributions
		
	def _save_topics(self, out_file, threshold=0.02):
		print('[*] Save topics to %s' % out_file)
		with open(out_file, 'w') as wf:
			for key, val in self.topics_dict.items():
			    topics = ' '.join([t_p.rsplit(':', 1)[0] for t_p in val[1].split() if float(t_p.rsplit(':', 1)[-1]) >= threshold])
			    wf.write(topics+'\n')


class BBTM:

	def __init__(self):
		self.word2id = {}   # store  { word: [id, freq], ... }

	def FileIndeXing(self, docs_dir, res_dir):
		"""
			Map each word to a unique ID (starts from 0) in the documents.
			Input Dir: 	docs_dir  e.g. ../reckful/cleaned_logs_dir/
			Output Dir: res_dir	  e.g. ../reckful/output/
		"""
		for fn in os.listdir(docs_dir):
			fname = os.path.join(docs_dir, fn)
			fout = os.path.join(res_dir, 'doc_wids', fn)
			self._index(fname, fout)

		self.save_word2id(os.path.join(res_dir, 'vocabulary.txt'))

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
				ids = ' '.join(list(map(str, word_ids)))
				wf.write(ids+'\n')

		wf.close()

	def save_word2id(self, file):
		with open(file, 'w') as f:
			f.write('ID\tWORD\tFREQ\n')
			for word, id_f in sorted(self.word2id.items(), key=lambda d:d[1]):
				f.write('%d\t%s\t%d\n' % (id_f[0], word, id_f[1]))

	def BitermFreq(self, output_dir):
		bf = self.proc_dir(os.path.join(output_dir, 'doc_wids'))
		self.save_bf(bf, output_dir)

	def proc_dir(self, dwid_dir):
		biterm_freq = defaultdict(str)
		for fn in sorted(os.listdir(dwid_dir), key=lambda d:d.split('.')[0]):
			bf = self._biterm_freq(os.path.join(dwid_dir, fn))
			for b, f in bf.items():
				biterm_freq[b] += '%s:%d ' % (fn.split('.')[0], f)
		return biterm_freq

	def _biterm_freq(self, file):
		bf = Counter()
		with open(file, 'r') as f:
			for l in f:
				ws = list(map(int, l.strip().split()))
				bs = self._generate_biterms(ws)
				bf.update(bs)
				
		return bf

	def _generate_biterms(self, ws):
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

	def save_bf(self, bf, output_dir):
		print('[*] Save the "bitermFreq.txt"')
		save_f = os.path.join(output_dir, 'biterm_freq.txt')
		with open(save_f, 'w') as f:
			for b, s in bf.items():
				f.write('%s\t%s\n' % (b, s))

